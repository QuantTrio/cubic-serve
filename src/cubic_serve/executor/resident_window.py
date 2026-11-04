"""Startup-only composition of sealed owned-decode graphs and guarded publication.

Model and format arithmetic remains in the original captured model and sampler.
The temporal window shares their topology lease; this module adds only bounded
integer control, granted page transitions and immutable step receipts.
"""

from __future__ import annotations

from cubic_serve.ops.cuda.collective import all_reduce, barrier
from cubic_serve.executor.readiness import execution_ready
from cubic_serve.executor.conditional_graph import add_conditional_child

import ctypes
import logging
import os
import time
from multiprocessing import shared_memory
from pathlib import Path

import torch
import torch.distributed as dist
from cuda.bindings import driver as cuda_driver, runtime as rt

from cubic_serve.ops.cuda.native import CudaScalar
from cubic_serve.ops.cuda.native_compiler import compile_native
from cubic_serve.runtime.continuous_batch import (
    RESIDENT_STEP_CAPACITY,
    resident_gpu_step_intervals,
)

_LOGGER = logging.getLogger(__name__)
SOURCE = (Path(__file__).parents[1] / "ops/cuda/sampling/resident_window.cu").read_bytes()


class SharedResidentJournal:
    """Worker-owned CUDA-registered journal with a CPU-shared stop word.

    The existing coordinator reads its stop word at byte offset eight. A
    controller can revoke an active window without waiting for its publishing
    thread. Graphs and exported tensor views must be retired before close.
    """

    def __init__(self, elements: int, device: torch.device):
        if elements < 64:
            raise ValueError("resident journal requires its complete header")
        self.device = torch.device(device)
        page_bytes = os.sysconf("SC_PAGE_SIZE")
        size = ((elements * 8 + page_bytes - 1) // page_bytes) * page_bytes
        self.memory = shared_memory.SharedMemory(create=True, size=size)
        self.address = ctypes.addressof(ctypes.c_char.from_buffer(self.memory.buf))
        self.tensor = None
        self._registered = False
        self._closed = False
        try:
            with torch.cuda.device(device):
                flags = int(rt.cudaHostRegisterMapped) | int(rt.cudaHostRegisterPortable)
                check(rt.cudaHostRegister(self.address, size, flags))
                self._registered = True
                self.device_address = int(check(rt.cudaHostGetDevicePointer(self.address, 0))[0])
                self.tensor = torch.frombuffer(self.memory.buf, dtype=torch.int64, count=elements)
                self.tensor.zero_()
                if not self.tensor.is_pinned():
                    raise RuntimeError("resident journal registration is not CUDA-accessible")
        except BaseException:
            self.close()
            raise

    @property
    def name(self) -> str:
        return self.memory.name

    def close(self) -> None:
        if self._closed:
            return
        if self._registered:
            with torch.cuda.device(self.device):
                check(rt.cudaHostUnregister(self.address))
            self._registered = False
        self.tensor = None
        self.memory.close()
        self.memory.unlink()
        self._closed = True


def resident_publication_names(executors) -> tuple[str, ...]:
    """Export only actual final-portfolio controller-writable journals."""
    names = set()
    for executor in executors.values():
        rounds = getattr(executor, "_pipelined_draft_rounds", {}).values()
        if not rounds:
            rounds = (getattr(executor, "_pipelined_draft_round", None),)
        windows = [getattr(executor, "_resident_window", None),
                   getattr(executor, "_resident_stochastic_window", None)]
        windows.extend(getattr(round_, attribute, None) for round_ in rounds
                       for attribute in ("_resident_window", "_candidate_window",
                                         "_budget_resident_window", "_budget_candidate_window"))
        for window in windows:
            journal = getattr(window, "_shared_journal", None)
            if journal is not None:
                names.add(journal.name)
    return tuple(sorted(names))


def resident_input_program_capacities(executors, *, stochastic: bool = False) -> tuple[int, ...]:
    """Advertise the sampling policy's completed single-token input program."""
    attribute = "_resident_stochastic_window" if stochastic else "_resident_window"
    return tuple(sorted(
        capacity for capacity, executor in executors.items()
        if execution_ready(getattr(executor, attribute, None), "program_qualified")
        and executor._target_hidden_slots is None
    ))


def resident_segment_program_capacities(executors) -> tuple[int, ...]:
    """Expose a segment window only after its actual model/state witness."""
    return tuple(sorted(
        capacity for capacity, executor in executors.items()
        if execution_ready(getattr(executor, "_resident_window", None),
                           "segmented_program_qualified")
        and (executor._target_hidden_slots is None
             or getattr(executor._resident_window, 'draft_maintenance', None) is not None)
    ))


class CapturedRowCompaction:
    """Startup-captured metadata gather with an explicit live-row map.

    Fields are (source, destination, row axis, inactive value). All gathers
    finish before any destination write, including aliased in-place fields.
    This copies metadata only; model/KV/recurrent tensors are never fields.
    """

    def __init__(self, fields, activity, *, stream, scratch=None):
        if not fields:
            raise ValueError("row compaction requires metadata fields")
        self.fields = tuple(
            (source, target, axis, padding if isinstance(padding, torch.Tensor)
             else torch.tensor(padding, dtype=target.dtype, device=target.device))
            for source, target, axis, padding in fields
        )
        self.activity = activity
        source, target, axis, _ = self.fields[0]
        self.source_capacity = source.shape[axis]
        self.capacity = target.shape[axis]
        self.device = target.device
        if self.device.type != "cuda" or activity.shape != (self.capacity,):
            raise ValueError("row compaction requires caller-owned CUDA metadata")
        for source, target, axis, padding in self.fields:
            if (source.device != self.device or target.device != self.device
                    or source.dtype != target.dtype or source.ndim != target.ndim
                    or not 0 <= axis < target.ndim
                    or source.shape[axis] != self.source_capacity
                    or target.shape[axis] != self.capacity
                    or any(a != b for i, (a, b) in enumerate(zip(source.shape, target.shape, strict=True))
                           if i != axis)):
                raise ValueError("row compaction field geometry differs")
            if isinstance(padding, torch.Tensor) and (
                padding.device != self.device or padding.dtype != target.dtype
            ):
                raise ValueError("row compaction padding differs from field storage")
        self.indices_host = torch.zeros(self.capacity, dtype=torch.int64, pin_memory=True)
        self.indices = torch.zeros(self.capacity, dtype=torch.int64, device=self.device)
        self.count_host = torch.zeros((), dtype=torch.int32, pin_memory=True)
        self.count = torch.zeros((), dtype=torch.int32, device=self.device)
        self.ordinal = torch.arange(self.capacity, dtype=torch.int32, device=self.device)
        self.mask = torch.empty(self.capacity, dtype=torch.bool, device=self.device)
        self.scratch = tuple(scratch) if scratch is not None else tuple(
            torch.empty_like(target, memory_format=torch.contiguous_format)
            for _, target, _, _ in self.fields
        )
        if len(self.scratch) != len(self.fields) or any(
            temporary.shape != target.shape or temporary.dtype != target.dtype
            or temporary.device != target.device
            for temporary, (_, target, _, _) in zip(self.scratch, self.fields, strict=True)
        ):
            raise ValueError("row compaction scratch differs from field storage")
        self.masks = tuple(
            self.mask.view(tuple(self.capacity if dim == axis else 1
                                 for dim in range(target.ndim)))
            for _, target, axis, _ in self.fields
        )
        self.graph = None
        self.bind(tuple(range(self.capacity)))
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for _ in range(2):
                self._launch()
        stream.synchronize()
        self.graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(self.graph, stream=stream):
            self._launch()
        self.graph.instantiate()
        stream.synchronize()

    def _launch(self):
        torch.lt(self.ordinal, self.count, out=self.mask)
        for (source, _, axis, _), temporary in zip(self.fields, self.scratch, strict=True):
            torch.index_select(source, axis, self.indices, out=temporary)
        for (_, target, _, padding), temporary, mask in zip(
            self.fields, self.scratch, self.masks, strict=True,
        ):
            torch.where(mask, temporary, padding, out=target)
        self.activity.copy_(self.mask)

    def bind(self, rows, *, source_limit=None):
        limit = self.source_capacity if source_limit is None else source_limit
        if (not 0 < limit <= self.source_capacity or not 0 < len(rows) <= self.capacity
                or len(set(rows)) != len(rows)
                or any(type(row) is not int or not 0 <= row < limit for row in rows)):
            raise ValueError("row compaction map differs from its owned source rows")
        values = self.indices_host.numpy()
        values.fill(0)
        values[:len(rows)] = rows
        self.count_host.fill_(len(rows))
        self.indices.copy_(self.indices_host, non_blocking=True)
        self.count.copy_(self.count_host, non_blocking=True)

    def replay(self):
        if self.graph is None:
            raise RuntimeError("row compaction graph is not captured")
        self.graph.replay()

    def close(self):
        if self.graph is not None:
            self.graph.reset()
            self.graph = None
        self.fields = self.scratch = self.masks = ()
        self.activity = self.indices_host = self.indices = None
        self.count_host = self.count = self.ordinal = self.mask = None


def prepare_resident_compactions(executors, lifetime, *, qualification=None):
    """Qualify same-bucket survivor gathers using existing CUDA copy kernels."""
    for capacity, executor in executors.items():
        window = getattr(executor, "_resident_window", None)
        if (capacity < 4 or window is None or not execution_ready(window)
                or executor._target_hidden_slots is not None):
            continue
        b = executor.body.buffers
        layers = executor._local_state_layers.numel()
        reserved = set(getattr(executor, "_reserved_scratch_states", ()))
        scratch = tuple(slot for slot in executor._scratch_states if slot not in reserved)
        if ((b.state_slot_maps and len(scratch) < capacity)
                or len(executor._scratch_pages) < capacity):
            continue
        device = executor.body.hidden.device
        pages = torch.tensor(executor._scratch_pages[:capacity], dtype=b.page_source.dtype,
                             device=device)
        state_padding = None
        if b.state_slot_maps:
            states = torch.tensor(scratch[:capacity], dtype=b.state_slot_maps[0].dtype,
                                  device=device)
            slots = states.unsqueeze(0) * layers + torch.arange(
                layers, dtype=states.dtype, device=device,
            ).unsqueeze(1)
            state_padding = slots.unsqueeze(0).expand(2, -1, -1).contiguous()
        fields = []

        def field(tensor, axis, padding, fields=fields):
            fields.append((tensor, tensor, axis, padding))

        field(b.position_ids, 1, 0)
        field(b.page_source, 0, pages)
        field(b.page_destination, 0, pages)
        field(b.token_offsets, 0, 0)
        field(b.sequence_lengths, 0, 1)
        field(b.segment_lengths, 0, 0)
        field(b.page_table, 0, pages.unsqueeze(1))
        for mapping in b.state_slot_maps:
            field(mapping, 2, state_padding)
        if executor.body.stage == 0:
            field(b.token_ids, 0, 0)
            field(executor.body.embedding_mask, 0, 0)
        field(executor._token_ids, 0, 0)
        for name in ("temperatures", "top_ps", "top_ks", "seeds", "ordinals"):
            field(getattr(executor._sampling, name), 0, 0)
        compaction = CapturedRowCompaction(fields, b.row_activity,
                                          stream=executor.slot.stream)
        lifetime.callback(compaction.close)
        # Copies have no floating arithmetic. Independently witness every
        # producer field, including padded rows and >2**54 RNG storage.
        from cubic_serve.runtime.performance import _logical_m_values, _prepare_audit
        from cubic_serve.runtime.quantized_qualification import _has_typed_boundary
        from cubic_serve.runtime.text_qualification import prepare_rows

        domain = getattr(executor.body, "bucket_logical_ms", None)
        domain = tuple(domain) if domain is not None else _logical_m_values(capacity)
        def qualify_compaction():
            for count in resident_audit_rows(capacity, domain):
                for reverse in (False, True):
                    if _has_typed_boundary(executor.body):
                        prepare_rows(executor.body, capacity)
                    else:
                        _prepare_audit(executor.body, executor.resource,
                                       poison=0, logical_m=capacity)
                    executor._token_ids.copy_(
                        (17 + torch.arange(capacity, device=device)) % executor.vocab_size
                    )
                    executor._sampling.seeds.copy_(
                        2**54 + torch.arange(capacity, device=device, dtype=torch.int64)
                    )
                    executor._sampling.ordinals.copy_(
                        31 + torch.arange(capacity, device=device, dtype=torch.int64)
                    )
                    source = tuple(value.detach().cpu().clone() for value, _, _, _ in fields)
                    selected = tuple(range(capacity - count, capacity))
                    if reverse:
                        selected = selected[::-1]
                    compaction.bind(selected, source_limit=capacity)
                    compaction.replay()
                    torch.cuda.current_stream(device).synchronize()
                    for (_, target, axis, padding), previous in zip(fields, source, strict=True):
                        expected = (padding.detach().cpu().expand(target.shape).clone()
                                    if isinstance(padding, torch.Tensor)
                                    else torch.full(target.shape, padding, dtype=target.dtype,
                                                    device="cpu"))
                        expected.narrow(axis, 0, count).copy_(
                            previous.index_select(axis, torch.tensor(selected, device="cpu"))
                        )
                        if not torch.equal(target.detach().cpu().contiguous().view(torch.uint8),
                                           expected.contiguous().view(torch.uint8)):
                            raise RuntimeError("resident metadata compaction differs from row map")
                    if b.row_activity.cpu().tolist() != [1] * count + [0] * (capacity - count):
                        raise RuntimeError("resident metadata compaction changed row activity")
                    barrier(group=executor._world_group)
        if qualification is None:
            qualify_compaction()
            compaction.qualified = True
        else:
            qualification.compaction(compaction, (capacity, domain,
                tuple((tuple(t.shape), str(t.dtype), axis) for t, _, axis, _ in fields)),
                qualify_compaction)
        # The first ordinary bind must clear the entire old active span.
        executor._bound_active_rows = capacity
        executor._resident_compaction = compaction
        _LOGGER.info("resident metadata compaction qualified rank=%d M%d",
                     executor._rank, capacity)


def resident_compaction_capacities(executors):
    return tuple(sorted(capacity for capacity, executor in executors.items()
                        if execution_ready(getattr(executor, "_resident_compaction", None))))


def prepare_resident_communication():
    """The leased replay protocol serializes communicator use explicitly.

    NCCL's implicit graph-mixing events cannot run outside a conditional body
    on behalf of its repeated collectives. The common worker initializes that
    serialized contract before creating any communicators. Explicit incompatible
    overrides are diagnosed when the resident graph is constructed.
    """
    os.environ.setdefault("NCCL_GRAPH_MIXING_SUPPORT", "0")
    os.environ.setdefault("NCCL_GRAPH_STREAM_ORDERING", "0")


def append_captured_graph(graph: torch.cuda.CUDAGraph) -> None:
    """Compose one retained body at the current startup-capture dependency tip.

    No graph is launched or captured here. CUDA copies the retained child DAG;
    the caller continues capture after that child, preserving incoming edges.
    The owning stage retains its original graph and tensor storage lifetime.
    """
    from cuda.bindings import runtime as rt

    stream = torch.cuda.current_stream()
    status, _, parent, dependencies, edges, *_ = check(
        rt.cudaStreamGetCaptureInfo(stream.cuda_stream))
    if status != rt.cudaStreamCaptureStatus.cudaStreamCaptureStatusActive:
        raise ValueError("graph composition requires an active startup capture")
    params = rt.cudaGraphNodeParams()
    params.type = rt.cudaGraphNodeType.cudaGraphNodeTypeGraph
    params.graph.graph = graph.raw_cuda_graph()
    node = check(rt.cudaGraphAddNode(parent, dependencies, edges, len(dependencies), params))[0]
    check(rt.cudaStreamUpdateCaptureDependencies(stream.cuda_stream, [node], None, 1,
        int(rt.cudaStreamUpdateCaptureDependenciesFlags.cudaStreamSetCaptureDependencies)))


def check(result):
    if int(result[0]):
        raise RuntimeError("resident CUDA graph operation failed: " + str(result[0]))
    return result[1:]


def conditional_handle(graph, initial):
    """Bind control to the same explicit driver context as native kernels."""
    context = check(cuda_driver.cuCtxGetCurrent())[0]
    if not int(context):
        raise RuntimeError("conditional graph requires a current CUDA context")
    return check(cuda_driver.cuGraphConditionalHandleCreate(
        graph, context, initial, cuda_driver.CU_GRAPH_COND_ASSIGN_DEFAULT,
    ))[0]


def add_conditional_body(graph, dependencies, handle, *, while_loop):
    """Create the node and its body with CUDA 13.0's explicit context contract."""
    context = check(cuda_driver.cuCtxGetCurrent())[0]
    if not int(context):
        raise RuntimeError("conditional graph requires a current CUDA context")
    params = cuda_driver.CUgraphNodeParams()
    params.type = cuda_driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_CONDITIONAL
    params.conditional.handle = handle
    params.conditional.ctx = context
    params.conditional.type = (
        cuda_driver.CUgraphConditionalNodeType.CU_GRAPH_COND_TYPE_WHILE if while_loop else
        cuda_driver.CUgraphConditionalNodeType.CU_GRAPH_COND_TYPE_IF
    )
    params.conditional.size = 1
    node = check(cuda_driver.cuGraphAddNode(
        graph, dependencies, None, len(dependencies), params,
    ))[0]
    return node, params.conditional.phGraph_out[0]


def resident_audit_rows(capacity, logical_domain):
    """Use the bucket's qualified lower, interior and upper logical rows."""
    domain = tuple(logical_domain)
    if not domain or any(type(m) is not int or not 1 <= m <= capacity for m in domain):
        raise ValueError("resident audit requires a valid bucket logical domain")
    if len(set(domain)) != len(domain) or tuple(sorted(domain)) != domain:
        raise ValueError("resident audit logical domain must be ordered and unique")
    return tuple(dict.fromkeys((domain[0], domain[len(domain) // 2], domain[-1])))


class ResidentDecodeWindow:
    """One bounded, startup-instantiated temporal extension of an existing bucket."""

    def __init__(self, e, sampler, *, journal=None, draft_provider=None, stochastic=False):
        if (
            os.environ.get("NCCL_GRAPH_MIXING_SUPPORT") != "0"
            or os.environ.get("NCCL_GRAPH_STREAM_ORDERING") != "0"
        ):
            raise RuntimeError("resident communication requires serialized graph ordering")
        self.e = e
        self.sampler = sampler
        if type(stochastic) is not bool or (stochastic and draft_provider is not None):
            raise ValueError("resident sampling strategy has an unsupported maintenance contract")
        self.stochastic = stochastic
        self.root_sampler = e._sampling.root_sampler if stochastic else None
        if stochastic and e._rank == e._sampler_rank and self.root_sampler is None:
            raise RuntimeError("stochastic resident sampling has no bound root primitive")
        self.draft_maintenance = None
        device = e.body.hidden.device
        self.policy = torch.zeros(3, dtype=torch.uint64, device=device)
        self.update_activity = torch.zeros(e.captured_m, dtype=torch.bool, device=device)
        self.policy_host = torch.zeros(3, dtype=torch.uint64, pin_memory=True)
        self.meta = torch.zeros(5, dtype=torch.int32, device=device)
        self.meta_host = torch.zeros(5, dtype=torch.int32, pin_memory=True)
        self.eos_offsets = torch.zeros(e.captured_m + 1, dtype=torch.int32, device=device)
        self.eos_offsets_host = torch.zeros_like(self.eos_offsets, device="cpu", pin_memory=True)
        self.eos_values = torch.empty(e.vocab_size, dtype=torch.int64, device=device)
        self.eos_values_host = torch.empty_like(self.eos_values, device="cpu", pin_memory=True)
        self.prompt_steps = torch.zeros(e.captured_m, dtype=torch.int32, device=device)
        self.prompt_steps_host = torch.zeros_like(self.prompt_steps, device="cpu", pin_memory=True)
        self.output_rows = torch.arange(e.captured_m, dtype=torch.int32, device=device)
        self.output_rows_host = torch.arange(e.captured_m, dtype=torch.int32, pin_memory=True)
        self.logical_steps = torch.ones(e.captured_m, dtype=torch.int32, device=device)
        self.logical_steps_host = torch.ones(e.captured_m, dtype=torch.int32, pin_memory=True)
        self._output_map_bound = False
        self._logical_steps_bound = False
        self.page_program = torch.full(
            (RESIDENT_STEP_CAPACITY, e.captured_m, 4), -1, dtype=torch.int32, device=device)
        self.page_program_host = torch.full_like(
            self.page_program, -1, device="cpu", pin_memory=True)
        self.owner_starts = torch.zeros(e.captured_m, dtype=torch.int32, device=device)
        self.owner_starts_host = torch.zeros_like(self.owner_starts, device="cpu", pin_memory=True)
        self.input_tokens = torch.zeros(
            (RESIDENT_STEP_CAPACITY, e.captured_m), dtype=torch.int64, device=device
        )
        self.input_tokens_host = torch.zeros_like(self.input_tokens, device="cpu", pin_memory=True)
        self.input_positions = torch.zeros(
            (RESIDENT_STEP_CAPACITY, e.body.buffers.position_ids.shape[0], e.captured_m),
            dtype=e.body.buffers.position_ids.dtype, device=device,
        )
        self.input_positions_host = torch.zeros_like(
            self.input_positions, device="cpu", pin_memory=True
        )
        self.output = torch.empty(
            (RESIDENT_STEP_CAPACITY, e.captured_m), dtype=torch.int64, device=device
        )
        self.clocks = torch.empty((RESIDENT_STEP_CAPACITY, 2), dtype=torch.int64, device=device)
        host_elements = 64 + RESIDENT_STEP_CAPACITY * (e.captured_m + 3)
        self._shared_journal = journal
        if journal is None:
            self.host = torch.zeros(host_elements, dtype=torch.int64, pin_memory=True)
        else:
            if (journal.device != device or journal.tensor is None
                    or journal.tensor.numel() != host_elements):
                raise RuntimeError("resident publication resource differs from its graph")
            self.host = journal.tensor
        self.host_array = self.host.numpy()
        self.host_flags = self.host_array.view("int32")
        mapped = check(rt.cudaHostGetDevicePointer(self.host.data_ptr(), 0))[0]
        self.host_address = torch.tensor([int(mapped)], dtype=torch.uint64, device=device)
        self.page_schedule = torch.full(
            (RESIDENT_STEP_CAPACITY, e.captured_m, 3), -1, dtype=torch.int32, device=device
        )
        self.page_schedule_host = torch.full_like(
            self.page_schedule, -1, device="cpu", pin_memory=True
        )
        # Fixed views share the existing pinned buffers. Fill integer metadata
        # without creating one temporary Tensor and dispatching a copy per grant.
        self._host_staging_arrays = {
            name: getattr(self, name).numpy()
            for name in (
                "policy_host", "meta_host", "page_schedule_host",
                "eos_offsets_host", "eos_values_host",
                "prompt_steps_host", "input_tokens_host", "input_positions_host",
                "output_rows_host", "logical_steps_host",
                "page_program_host", "owner_starts_host",
            )
        }
        self.graph = check(rt.cudaGraphCreate(0))[0]
        self.exec = None
        handle = conditional_handle(self.graph, 1)
        page_handle = conditional_handle(self.graph, 0)
        model_graph = e.slot.owned_decode_graph
        segment_graph = e.slot.graph
        distinct_models = model_graph is not segment_graph
        segment_handle = (conditional_handle(self.graph, 0)
                          if distinct_models else 0)
        decode_handle = (conditional_handle(self.graph, 0)
                         if distinct_models else 0)
        module, entry = compile_native((("entry.cu", SOURCE),), device, "resident_window")
        args = (
            self.meta,
            e._token_ids,
            sampler.final_valid,
            self.eos_offsets,
            self.eos_values,
            self.output,
            self.clocks,
            self.host_address,
            self.page_schedule,
            e._continuation_updates_device,
            self.policy,
            e.body.buffers.row_activity,
            self.prompt_steps,
            self.output_rows,
            self.update_activity,
            self.page_program,
            self.owner_starts,
            e.body.buffers.sequence_lengths,
            e.body.buffers.segment_lengths,
            e.body.buffers.page_source,
            e.body.buffers.page_destination,
            e.body.buffers.page_table,
            e.body.buffers.token_offsets,
            CudaScalar("i32", e._block_size),
            CudaScalar("i32", e._table_pages),
            CudaScalar("i32", e.captured_m),
            CudaScalar("i32", e.vocab_size),
            CudaScalar("i32", int(e._rank == e._sampler_rank)),
            CudaScalar("i64", int(handle)),
            CudaScalar("i64", int(page_handle)),
            CudaScalar("i64", int(segment_handle)),
            CudaScalar("i64", int(decode_handle)),
        )
        self.calls = tuple(
            module.bind(
                entry,
                (*args, CudaScalar("i32", phase)),
                operator_id="runtime.resident_window_coordinate",
                grid=(1, 1, 1),
                block=(128, 1, 1),
            )
            for phase in range(5)
        )
        from cubic_serve.ops.cuda.pointwise.qwen_decode import (
            advance_resident_inputs,
            update_resident_pages,
        )

        b = e.body.buffers
        first = e.body.stage == 0
        span = getattr(e.body, "embedding_input_span", e.body.embedding_span) if first else None
        mask = e.body.embedding_mask if first else None
        self.control = self.meta[4:5]
        all_reduce(self.control, op=dist.ReduceOp.MAX, group=e._world_group)
        torch.cuda.current_stream(device).synchronize()

        def pages():
            update_resident_pages(
                e._continuation_updates_device,
                e.body.arena.key,
                e.body.arena.value,
                e.body.arena.written,
                b.token_offsets,
                b.page_source,
                b.page_destination,
                b.page_table,
                page_fields=tuple(e.body.arena.page_fields.values()),
                update_activity=self.update_activity,
            )

        update_resident_pages(
            e._continuation_updates_device,
            e.body.arena.key,
            e.body.arena.value,
            e.body.arena.written,
            b.token_offsets,
            b.page_source,
            b.page_destination,
            b.page_table,
            page_fields=tuple(e.body.arena.page_fields.values()),
            update_activity=self.update_activity,
            compile_only=True,
        )
        maintenance_graph = None
        if draft_provider is not None:
            from cubic_serve.executor.draft_graph import ResidentDraftMaintenance
            self.draft_maintenance = ResidentDraftMaintenance(e, draft_provider, self)
            e.body.prepare_scratch(e.resource)
            # No prompt owners are bound during construction. Stage scratch
            # policies may exercise active dummy rows, but provider metadata
            # must never interpret unbound owner columns as request identity.
            e.body.buffers.row_activity.zero_()
            e.body.buffers.segment_lengths.zero_()
            draft_provider.body.prepare_scratch()
            e.slot.stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(e.slot.stream):
                self.draft_maintenance.launch()
            e.slot.stream.synchronize()
            maintenance_graph = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(maintenance_graph, stream=e.slot.stream):
                self.draft_maintenance.launch()
            e.slot.stream.synchronize()

        if self.stochastic:
            # Materialize the existing top-k and root kernels before capture,
            # using only the caller's private cold-start buffers.
            with torch.cuda.stream(e.slot.stream):
                sampler.prepare_candidates()
                sampler.sample_candidates(self.root_sampler)
                sampler._broadcast()
            e.slot.stream.synchronize()

        ingress = torch.cuda.CUDAGraph(keep_graph=True)
        page_graph = torch.cuda.CUDAGraph(keep_graph=True)
        begin_graph = torch.cuda.CUDAGraph(keep_graph=True)
        finish_graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(ingress, stream=e.slot.stream):
            advance_resident_inputs(
                b.position_ids,
                b.token_offsets,
                b.sequence_lengths,
                e._token_ids,
                b.token_ids,
                e._sampling.ordinals,
                b.row_activity,
                mask,
                vocab_start=span.start if span else 0,
                vocab_stop=span.stop if span else 0,
                input_tokens=self.input_tokens,
                input_positions=self.input_positions,
                prompt_steps=self.prompt_steps,
                step=self.meta[2:3],
                logical_steps=self.logical_steps,
            )
            self.calls[3].launch()
        with torch.cuda.graph(page_graph, stream=e.slot.stream):
            pages()
        with torch.cuda.graph(begin_graph, stream=e.slot.stream):
            self.calls[0].launch()
        with torch.cuda.graph(finish_graph, stream=e.slot.stream):
            self.calls[1].launch()
            if self.stochastic:
                sampler.prepare_candidates()
                sampler.sample_candidates(self.root_sampler)
            else:
                sampler._prepare()
            sampler._broadcast()
            self.calls[2].launch()
            all_reduce(self.control, op=dist.ReduceOp.MAX, group=e._world_group)
            self.calls[4].launch()
        # Use the existing actual sealed model graph rather than recapturing its body.
        e.slot.select_decode_overlap(True)
        self.owned_graphs = (ingress, page_graph, begin_graph,
                            *(() if maintenance_graph is None else (maintenance_graph,)), finish_graph)
        for child in self.owned_graphs:
            child.instantiate()
        model_children = (segment_graph, model_graph) if distinct_models else (model_graph,)
        self.children = (ingress, page_graph, begin_graph, *model_children,
                         *(() if maintenance_graph is None else (maintenance_graph,)), finish_graph)
        _, body = add_conditional_body(self.graph, [], handle, while_loop=True)
        first_node = add_conditional_child(body, [], ingress.raw_cuda_graph())
        page_node, page_body = add_conditional_body(body, [first_node], page_handle, while_loop=False)
        add_conditional_child(page_body, [], page_graph.raw_cuda_graph())
        previous = add_conditional_child(body, [page_node], begin_graph.raw_cuda_graph())
        if distinct_models:
            # Preserve the sealed monolithic segment graph and the measured
            # independent-row decode graph under their existing device controls.
            for child, conditional in ((segment_graph, segment_handle),
                                       (model_graph, decode_handle)):
                previous, choice_body = add_conditional_body(
                    body, [previous], conditional, while_loop=False)
                add_conditional_child(choice_body, [], child.raw_cuda_graph())
        else:
            previous = add_conditional_child(body, [previous], model_graph.raw_cuda_graph())
        if maintenance_graph is not None:
            previous = add_conditional_child(body, [previous], maintenance_graph.raw_cuda_graph())
        add_conditional_child(body, [previous], finish_graph.raw_cuda_graph())
        self.exec = check(cuda_driver.cuGraphInstantiate(self.graph, 0))[0]
        self.qualified = False
        self.segmented_program_qualified = False

    def close(self):
        self.qualified = False
        if self.exec is not None:
            check(rt.cudaGraphExecDestroy(self.exec))
            self.exec = None
        if self.graph is not None:
            check(rt.cudaGraphDestroy(self.graph))
            self.graph = None
        for child in getattr(self, "owned_graphs", ()):
            child.reset()
        if getattr(self, "_shared_journal", None) is not None:
            # The owning ExitStack unregisters and closes storage after this
            # callback. Release every exported CPU view before that happens.
            self.host_flags = None
            self.host_array = None
            self.host = None

    def bind(
        self, rows, steps, eos, page_updates=(), duration=1000.0, *,
        prompt_remaining=(), prompt_tokens=(), prompt_positions=(),
        output_rows=(), physical_rows=None, logical_steps=(),
        page_program=(), source_pages=(), owner_offsets=(), owner_page_counts=(),
    ):
        if not 0 < rows <= self.e.captured_m or not 0 < steps <= self.output.shape[0]:
            raise RuntimeError("resident window outside fixed resources")
        physical_rows = rows if physical_rows is None else physical_rows
        if type(physical_rows) is not int or not rows <= physical_rows <= self.e.captured_m:
            raise RuntimeError("resident token rows differ from the captured domain")
        output_rows = tuple(range(rows)) if not output_rows else output_rows
        logical_steps = (1,) * physical_rows if not logical_steps else logical_steps
        if (type(output_rows) is not tuple or len(output_rows) != rows
                or any(type(row) is not int or not 0 <= row < physical_rows for row in output_rows)
                or len(set(output_rows)) != rows
                or type(logical_steps) is not tuple or len(logical_steps) != physical_rows
                or any(type(value) is not int or not 1 <= value < 2**31
                       for value in logical_steps)):
            raise RuntimeError("resident owner rows or logical steps are invalid")
        if any(value > 1 for value in logical_steps) and (
            len(prompt_remaining) != physical_rows
            or any(value > 1 and (type(prompt_remaining[row]) is not int
                                  or prompt_remaining[row] <= steps)
                   for row, value in enumerate(logical_steps))
        ):
            raise RuntimeError("resident segments must finish before their prompt boundary")
        program_resets = []
        owner_starts = []
        widths = []
        first = 0
        for last in output_rows:
            width = logical_steps[last]
            if (last - width + 1 != first
                    or any(value != width for value in logical_steps[first:last + 1])):
                raise RuntimeError("resident owner partition differs from token rows")
            widths.append(width)
            owner_starts.extend((first,) * width)
            first = last + 1
        if first != physical_rows:
            raise RuntimeError("resident owner partition does not cover token rows")
        if any(value > 1 for value in logical_steps) and not page_program:
            raise RuntimeError("resident segments require an owned page program")
        if page_program:
            from cubic_serve.runtime.continuous_batch import valid_owner_page_program

            if page_updates or not valid_owner_page_program(
                tuple(widths), source_pages, page_program, owner_offsets, owner_page_counts,
                steps=steps, capacity=self.e.captured_m, block_size=self.e._block_size,
                page_limit=self.e.body.arena.spec.page_capacity, table_limit=self.e._table_pages,
            ):
                raise RuntimeError("resident page program differs from its owned resources")
            program_resets = [
                (step, row, index, source)
                for step, coordinates in enumerate(page_program)
                for row, (source, _destination, index, reset) in enumerate(coordinates) if reset
            ]
        elif source_pages or owner_offsets or owner_page_counts:
            raise RuntimeError("resident page ownership has no program")
        offsets = [0]
        values = []
        for ids in eos:
            values.extend(ids)
            offsets.append(len(values))
        if len(eos) != rows or len(values) > self.e.vocab_size:
            raise RuntimeError("resident EOS list outside owned resource")
        self.host_flags[0] = 0
        # After readiness the controller owns a shared journal's stop word.
        # In particular, an admission/cancellation between dispatch and bind
        # must survive this fill. Startup qualification still owns its header.
        if not (getattr(self, "_shared_journal", None) is not None
                and execution_ready(self)):
            self.host_flags[2] = 0
        arrays = self._host_staging_arrays
        mapped = output_rows != tuple(range(rows))
        if mapped or self._output_map_bound:
            arrays["output_rows_host"][:] = range(self.e.captured_m)
            arrays["output_rows_host"][:rows] = output_rows
            self.output_rows.copy_(self.output_rows_host, non_blocking=True)
        self._output_map_bound = mapped
        strided = any(value != 1 for value in logical_steps)
        if strided or self._logical_steps_bound:
            arrays["logical_steps_host"].fill(1)
            arrays["logical_steps_host"][:physical_rows] = logical_steps
            self.logical_steps.copy_(self.logical_steps_host, non_blocking=True)
        self._logical_steps_bound = strided
        if page_program:
            arrays["page_program_host"][:steps, :physical_rows] = page_program
            self.page_program[:steps].copy_(self.page_program_host[:steps], non_blocking=True)
            arrays["owner_starts_host"][:physical_rows] = owner_starts
            self.owner_starts.copy_(self.owner_starts_host, non_blocking=True)
        prompt = arrays["prompt_steps_host"]
        prompt.fill(0)
        if prompt_remaining:
            from cubic_serve.runtime.continuous_batch import ContinueBatchCommand

            descriptor = ContinueBatchCommand(
                0, 0, self.e.captured_m, (), max_steps=steps,
                prompt_remaining=prompt_remaining, prompt_tokens=prompt_tokens,
                prompt_positions=prompt_positions,
            )
            if not descriptor._valid_prompt_program(physical_rows):
                raise RuntimeError("resident input program differs from its row domain")
            prompt[:physical_rows] = prompt_remaining
            tokens = arrays["input_tokens_host"]
            positions = arrays["input_positions_host"]
            tokens[:steps].fill(0)
            positions[:steps].fill(0)
            axes = positions.shape[1]
            for physical, (ids, points) in enumerate(zip(
                prompt_tokens, prompt_positions, strict=True,
            )):
                if ids:
                    tokens[:len(ids), physical] = ids
                    for axis in range(axes):
                        positions[:len(ids), axis, physical] = tuple(p[axis] for p in points)
            self.input_tokens[:steps].copy_(self.input_tokens_host[:steps], non_blocking=True)
            self.input_positions[:steps].copy_(self.input_positions_host[:steps], non_blocking=True)
        self.prompt_steps.copy_(self.prompt_steps_host, non_blocking=True)
        arrays["policy_host"][:] = (int(duration * 1e9), 0, int(bool(page_program)))
        self.policy.copy_(self.policy_host, non_blocking=True)
        schedule = arrays["page_schedule_host"]
        schedule[:steps].fill(-1)
        for step, physical, page_index, page in page_updates:
            schedule[step, physical] = (physical, page_index, page)
        for step, physical, page_index, page in program_resets:
            schedule[step, physical] = (physical, page_index, page)
        self.page_schedule[:steps].copy_(self.page_schedule_host[:steps], non_blocking=True)
        arrays["meta_host"][:] = (rows, steps, 0, 0, 0)
        self.meta.copy_(self.meta_host, non_blocking=True)
        arrays["eos_offsets_host"][: rows + 1] = offsets
        self.eos_offsets[: rows + 1].copy_(self.eos_offsets_host[: rows + 1], non_blocking=True)
        if values:
            arrays["eos_values_host"][: len(values)] = values
            self.eos_values[: len(values)].copy_(
                self.eos_values_host[: len(values)], non_blocking=True
            )
        self.sampler.metadata.copy_(self.sampler._metadata_hosts[physical_rows, 0], non_blocking=True)
        self.sampler._metadata_key = (physical_rows, False)

    def replay(self):
        check(
            rt.cudaGraphLaunch(
                self.exec, torch.cuda.current_stream(self.e.body.hidden.device).cuda_stream
            )
        )

    def consume(self, rows):
        torch.cuda.current_stream(self.e.body.hidden.device).synchronize()
        values = self.meta.cpu().tolist()
        count = values[2]
        if values[3] or not bool(self.sampler.final_valid.item()) or not 0 < count <= values[1]:
            raise RuntimeError("resident window rejected invalid token/validity/count")
        outputs = self.output[:count, :rows].cpu().tolist()
        for index, tokens in enumerate(outputs):
            begin = 64 + index * (self.e.captured_m + 3)
            if self.host_array[begin : begin + rows].tolist() != tokens:
                raise RuntimeError("mapped immutable publication differs from GPU token log")
        if int(self.host_flags[0]) != count:
            raise RuntimeError("mapped publication generation differs")
        clocks = self.clocks[:count].cpu().tolist()
        self._gpu_step_intervals_ms = resident_gpu_step_intervals(clocks)
        return outputs, [(b - a) / 1e6 for a, b in clocks]

    def qualify(self):
        """Prove unchanged execution/transport, not independent floating arithmetic.

        The original model's independent numerical receipts remain mandatory.
        This witness checks that composition adds no step, changes no state,
        and never publishes an invalid step or diverges across ranks.
        """
        from cubic_serve.runtime.performance import (
            _active_snapshot,
            _exact_snapshot,
            _logical_m_values,
            _prepare_audit,
        )
        from cubic_serve.runtime.quantized_qualification import _has_typed_boundary
        from cubic_serve.runtime.text_qualification import boundary_snapshot, prepare_rows

        e = self.e
        typed = _has_typed_boundary(e.body)

        def reset(rows):
            if typed:
                prepare_rows(e.body, rows)
            else:
                _prepare_audit(e.body, e.resource, poison=0, logical_m=rows)
            e._sampling.temperatures.fill_(0.7 if self.stochastic else 0.0)
            e._sampling.top_ps.fill_(0.95)
            e._sampling.top_ks.fill_(min(20, e.vocab_size, e._sampling.parameters.max_top_k))
            e._sampling.seeds.fill_(1)
            if self.stochastic:
                e._sampling.temperatures[1:rows:2].zero_()
                e._sampling.seeds.copy_(torch.arange(
                    e._sampling.seeds.numel(), device=e._sampling.seeds.device,
                    dtype=e._sampling.seeds.dtype) + 1)
            e._sampling.ordinals.zero_()
            e._token_ids.fill_(17 % e.vocab_size)

        def snapshot(rows):
            result = boundary_snapshot(e.body, rows) if typed else _active_snapshot(e.body, rows)
            return dict(result, sampling_seeds=e._sampling.seeds.cpu().clone(),
                        sampling_ordinals=e._sampling.ordinals.cpu().clone())

        def sample_reference(rows, validity_out):
            if self.stochastic:
                return self.sampler._reference_stochastic(
                    rows, self.root_sampler, validity_out=validity_out)
            return self.sampler.replay(rows, False, validity_out=validity_out)

        def reference_log(rows, steps):
            # Immutable per-step device records retain the entire journal. No
            # token/guard is dropped and no CPU read interrupts a serial step.
            device = e.body.hidden.device
            return (torch.empty((steps, rows), dtype=torch.int64, device=device),
                    torch.empty(steps, dtype=torch.bool, device=device))

        def record_reference(log, guards, step, rows):
            sample_reference(rows, guards[step])
            log[step].copy_(e._token_ids[:rows])
            # A failed deferred guard must never feed an unsafe embedding index
            # during subsequent private audit work. The original invalid token
            # remains in log and every guard is consumed before accepting it.
            e._token_ids[:rows].clamp_(0, e.vocab_size - 1)

        def consume_reference(log, guards):
            e.slot.synchronize()
            torch.cuda.current_stream(e.body.hidden.device).synchronize()
            barrier(group=e._world_group)
            if not bool(guards.all()):
                raise RuntimeError("serial resident witness rejected invalid sampling")
            return log.cpu().tolist()

        def serial(rows, steps):
            log, guards = reference_log(rows, steps)
            for step in range(steps):
                e._input_update_graph.replay()
                current = torch.cuda.current_stream(e.body.hidden.device)
                e.slot.stream.wait_stream(current)
                e.slot.replay()
                current.wait_stream(e.slot.stream)
                record_reference(log, guards, step, rows)
            return consume_reference(log, guards)

        domain = getattr(e.body, "bucket_logical_ms", None)
        if domain is None:
            domain = _logical_m_values(e.captured_m)
        witnesses = resident_audit_rows(e.captured_m, domain)
        for rows in witnesses:
            reset(rows)
            self.bind(rows, 3, ((),) * rows)
            expected = serial(rows, 3)
            anchor = snapshot(rows)
            reset(rows)
            self.bind(rows, 3, ((),) * rows)
            self.replay()
            actual, _ = self.consume(rows)
            e.slot.synchronize()
            barrier(group=e._world_group)
            if actual != expected or not _exact_snapshot(anchor, snapshot(rows)):
                raise RuntimeError("resident composition differs from its sealed steps")

        # Witness the new integer ingress against ordinary serial preparation,
        # not by running the same ingress kernel on both sides. Model floating
        # arithmetic still requires its separate independent numerical oracle.
        for rows in witnesses:
            remaining = tuple(3 if row == 0 else row % 4 for row in range(rows))
            inputs = tuple(tuple((17 + row + step * 7) % e.vocab_size
                                 for step in range(count))
                           for row, count in enumerate(remaining))
            points = tuple(tuple((step + 2, step * 3 + 5, step * 2 + 7)
                                 for step in range(count))
                           for count in remaining)
            results = []
            serial_tokens = None
            for composed in (False, True):
                reset(rows)
                b = e.body.buffers
                b.page_source.copy_(b.page_destination)
                if typed:
                    for mapping in b.state_slot_maps:
                        mapping[0].copy_(mapping[1])
                else:
                    b.conv_slots[0].copy_(b.conv_slots[1])
                    b.recurrent_slots[0].copy_(b.recurrent_slots[1])
                eos = ((),) * rows
                if composed:
                    # A mid-prompt sampled value must not terminate its row.
                    eos = ((serial_tokens[0][0],),) + ((),) * (rows - 1)
                self.bind(
                    rows, 3, eos, prompt_remaining=remaining,
                    prompt_tokens=inputs, prompt_positions=points,
                )
                if composed:
                    self.replay()
                    tokens, _ = self.consume(rows)
                else:
                    log, guards = reference_log(rows, 3)
                    for step in range(3):
                        e._input_update_graph.replay()
                        for row, count in enumerate(remaining):
                            if step >= count:
                                continue
                            for axis in range(b.position_ids.shape[0]):
                                b.position_ids[axis, row].fill_(points[row][step][axis])
                            e._sampling.ordinals[row].zero_()
                            if e.body.stage == 0:
                                span = getattr(
                                    e.body, "embedding_input_span", e.body.embedding_span,
                                )
                                token = inputs[row][step]
                                owned = span.start <= token < span.stop
                                b.token_ids[row].fill_(token - span.start if owned else 0)
                                e.body.embedding_mask[row].fill_(int(owned))
                        current = torch.cuda.current_stream(e.body.hidden.device)
                        e.slot.stream.wait_stream(current)
                        e.slot.replay()
                        current.wait_stream(e.slot.stream)
                        record_reference(log, guards, step, rows)
                    tokens = consume_reference(log, guards)
                    serial_tokens = tokens
                e.slot.synchronize()
                barrier(group=e._world_group)
                result = snapshot(rows)
                result.update({
                    "input_position_ids": b.position_ids.cpu().clone(),
                    "input_token_ids": b.token_ids.cpu().clone(),
                    "input_ordinals": e._sampling.ordinals.cpu().clone(),
                    "tokens": torch.tensor(tokens, dtype=torch.int64),
                })
                results.append(result)
            if not _exact_snapshot(results[0], results[1]):
                raise RuntimeError("resident input program differs from serial preparation")
            _LOGGER.info(
                "resident input program composition qualified rank=%d M%d logical=%d",
                e._rank, e.captured_m, rows,
            )

        rows = witnesses[0]
        reset(rows)
        self.bind(rows, 1, ((),) * rows)
        expected = serial(rows, 1)
        anchor = snapshot(rows)
        for case in ("budget", "eos", "deadline", "interrupt"):
            reset(rows)
            self.bind(
                rows,
                1 if case == "budget" else 3,
                ((expected[0][0],),) + ((),) * (rows - 1) if case == "eos" else ((),) * rows,
                duration=1e-9 if case == "deadline" else 1000.0,
            )
            if case == "interrupt":
                self.host_flags[2] = 1
            self.replay()
            actual, _ = self.consume(rows)
            e.slot.synchronize()
            barrier(group=e._world_group)
            if actual != expected or not _exact_snapshot(anchor, snapshot(rows)):
                raise RuntimeError("resident stopping boundary differs: " + case)

        reset(rows)
        self.bind(rows, 3, ((),) * rows)
        if e._rank == dist.get_world_size(e._world_group) - 1:
            self.meta[3].fill_(1)
        self.replay()
        torch.cuda.current_stream(e.body.hidden.device).synchronize()
        e.slot.synchronize()
        if self.meta.cpu().tolist()[2:5] != [1, 1, 2]:
            raise RuntimeError("resident invalidity did not stop all ranks uniformly")
        barrier(group=e._world_group)
        # Discrete multi-page transport proof on the actual sealed arithmetic.
        # This is composition identity, not an independent floating oracle.
        from cubic_serve.ops.cuda.pointwise.qwen_decode import update_resident_pages

        for rows in tuple(dict.fromkeys((witnesses[0], witnesses[-1]))):
            # Numerical/state propagation has its independent oracle. This
            # composition witness spans two complete page transitions. Every
            # journal address and the terminal model/control boundary are proved
            # separately below, without repeating the model for integer stores.
            steps = min(self.output.shape[0], 2 * e._block_size + 1)
            reset(rows)
            current_pages = set(e.body.buffers.page_destination[:rows].cpu().tolist())
            available = [
                page for page in range(e.body.arena.spec.page_capacity) if page not in current_pages
            ]
            crossing_steps = tuple(range(e._block_size - 1, steps, e._block_size))
            if len(available) < len(crossing_steps) * rows:
                raise RuntimeError("multi-page audit lacks privately owned pages")
            plan = tuple(
                (step, row, index + 1, available[index * rows + row])
                for index, step in enumerate(crossing_steps)
                for row in range(rows)
            )
            results = []
            for composed in (False, True):
                reset(rows)
                b = e.body.buffers
                b.page_source.copy_(b.page_destination)
                if typed:
                    for mapping in b.state_slot_maps:
                        mapping[0].copy_(mapping[1])
                else:
                    b.conv_slots[0].copy_(b.conv_slots[1])
                    b.recurrent_slots[0].copy_(b.recurrent_slots[1])
                self.bind(rows, steps, ((),) * rows, plan)
                if composed:
                    self.replay()
                    tokens, _ = self.consume(rows)
                else:
                    log, guards = reference_log(rows, steps)
                    for step in range(steps):
                        e._input_update_graph.replay()
                        entries = tuple(
                            (row, index, page) for at, row, index, page in plan if at == step
                        )
                        if entries:
                            # This pinned staging row is shared across pages;
                            # wait before reuse, rather than after every token.
                            torch.cuda.current_stream(e.body.hidden.device).synchronize()
                            e._continuation_updates_host_array[: len(entries)] = entries
                            e._continuation_updates_device[: len(entries)].copy_(
                                e._continuation_updates_host[: len(entries)], non_blocking=True
                            )
                            update_resident_pages(
                                e._continuation_updates_device[: len(entries)],
                                e.body.arena.key,
                                e.body.arena.value,
                                e.body.arena.written,
                                b.token_offsets,
                                b.page_source,
                                b.page_destination,
                                b.page_table,
                                page_fields=tuple(e.body.arena.page_fields.values()),
                            )
                        current = torch.cuda.current_stream(e.body.hidden.device)
                        e.slot.stream.wait_stream(current)
                        e.slot.replay()
                        current.wait_stream(e.slot.stream)
                        record_reference(log, guards, step, rows)
                    tokens = consume_reference(log, guards)
                torch.cuda.current_stream(e.body.hidden.device).synchronize()
                e.slot.synchronize()
                barrier(group=e._world_group)
                result = snapshot(rows)
                result["page_table"] = b.page_table[:rows].cpu().clone()
                result["offsets"] = b.token_offsets[:rows].cpu().clone()
                result["tokens"] = torch.tensor(tokens, dtype=torch.int64)
                results.append(result)
            if not _exact_snapshot(results[0], results[1]):
                raise RuntimeError("multi-page journal differs from serial sealed transitions")
            _LOGGER.info(
                "multi-page transport qualified rank=%d M%d logical=%d", e._rank, e.captured_m, rows
            )
        for rows in tuple(dict.fromkeys((witnesses[0], witnesses[-1]))):
            reset(rows)
            self.bind(rows, self.output.shape[0], ((),) * rows)
            self._qualify_journal_storage(rows)

            # Exercise the real conditional loop at its last output slots.
            # Model state/position and the journal index are independent inputs:
            # both arms start from the identical cold snapshot and execute the
            # same three model/sampler steps, with fixed RNG on every rank.
            steps = min(3, self.output.shape[0])
            begin = self.output.shape[0] - steps
            reset(rows)
            self.bind(rows, steps, ((),) * rows)
            expected = serial(rows, steps)
            anchor = snapshot(rows)
            reset(rows)
            self.bind(rows, self.output.shape[0], ((),) * rows)
            self.output.fill_(-99)
            self.clocks.fill_(-99)
            self.host_array[64:] = -99
            self.meta[2].fill_(begin)
            # The deadline's start tick is normally written at index0. Disable
            # just that policy in this sparse-tail fixture; deadline behavior is
            # independently exercised above on the real loop.
            self.policy[0].zero_()
            self.replay()
            torch.cuda.current_stream(e.body.hidden.device).synchronize()
            e.slot.synchronize()
            barrier(group=e._world_group)
            meta = self.meta.cpu().tolist()
            output = self.output.cpu()
            clocks = self.clocks.cpu()
            mapped = torch.from_numpy(self.host_array[64:]).reshape(-1, e.captured_m + 3)
            if (meta[2:] != [self.output.shape[0], 0, 1]
                    or output[begin:, :rows].tolist() != expected
                    or not _exact_snapshot(anchor, snapshot(rows))
                    or not bool((output[:begin] == -99).all())
                    or not bool((output[begin:, rows:] == -99).all())
                    or not bool((clocks[:begin] == -99).all())
                    or not bool((mapped[:begin] == -99).all())
                    or not torch.equal(mapped[begin:, :rows], output[begin:, :rows])
                    or not bool((mapped[begin:, rows:e.captured_m] == -99).all())
                    or not torch.equal(mapped[begin:, e.captured_m],
                                       clocks[begin:, 1] - clocks[begin:, 0])
                    or mapped[begin:, e.captured_m + 1].tolist() != [0] * (steps - 1) + [1]
                    or not bool((mapped[begin:, e.captured_m + 2] == 0).all())
                    or int(self.host_flags[0]) != self.output.shape[0]):
                raise RuntimeError("terminal journal composition differs from sealed steps")
            _LOGGER.info("resident terminal model composition qualified rank=%d M%d logical=%d "
                         "begin=%d steps=%d", e._rank, e.captured_m, rows, begin, steps)
        self.qualified = True
        self.program_qualified = True
        self.qualify_segments(witnesses)
        _LOGGER.info(
            "resident graph composition and stop controls qualified rank=%d M%d",
            e._rank,
            e.captured_m,
        )

    def _qualify_journal_storage(self, rows):
        """Check every bound native journal address and protected byte directly.

        Phase2 only writes token/control storage and contains no conditional-
        handle call. It can run without a model forward. The actual conditional
        loop, publication fences, counters, RNG and state are checked separately
        at start, page transitions, stop conditions and the final journal slots.
        This is an integer transport witness, never a floating-point oracle.
        """
        e, steps = self.e, self.output.shape[0]
        sentinel = -99
        self.output.fill_(sentinel)
        self.host_array[64:] = sentinel
        self.host_flags[0] = 0
        self.host_flags[2] = 0
        self.policy[0].zero_()
        self.sampler.final_valid.fill_(True)
        expected = (torch.arange(rows, device=e._token_ids.device, dtype=torch.int64)
                    .mul_(7919).add_(17).remainder_(e.vocab_size))
        e._token_ids[:rows].copy_(expected)
        controls = torch.empty(steps, dtype=torch.int32, device=e._token_ids.device)
        for index in range(steps):
            self.meta[2].fill_(index)
            self.calls[2].launch()
            controls[index].copy_(self.meta[4])
        torch.cuda.current_stream(e.body.hidden.device).synchronize()
        e.slot.synchronize()
        barrier(group=e._world_group)
        expected = expected.cpu().expand(steps, -1)
        output = self.output.cpu()
        mapped = torch.from_numpy(self.host_array[64:]).reshape(steps, e.captured_m + 3)
        wanted_control = torch.zeros(steps, dtype=torch.int32)
        wanted_control[-1] = 1
        if (not torch.equal(output[:, :rows], expected)
                or not torch.equal(mapped[:, :rows], expected)
                or not bool((output[:, rows:] == sentinel).all())
                or not bool((mapped[:, rows:] == sentinel).all())
                or not torch.equal(controls.cpu(), wanted_control)
                or int(self.meta[3]) != 0 or int(self.host_flags[0]) != 0):
            raise RuntimeError("full native journal storage or control write-set differs")
        _LOGGER.info("resident full native journal storage qualified rank=%d M%d logical=%d "
                     "journal_steps=%d", e._rank, e.captured_m, rows, steps)

    def qualify_segments(self, witnesses):
        """Prove cross-page composition against ordinary batch preparation.

        The cold prefix and fixed K trace are identical in both arms. This
        witnesses transport/state composition; independent FP audits remain
        mandatory. Every snapshot follows completed forwards and rank barriers.
        """
        from dataclasses import replace
        from cubic_serve.runtime.continuous_batch import BatchRow
        from cubic_serve.runtime.performance import _exact_snapshot

        e, b, arena = self.e, self.e.body.buffers, self.e.body.arena
        maintenance = self.draft_maintenance
        if e._target_hidden_slots is not None and maintenance is None:
            return
        prefix, steps = max(0, e._block_size - 2), 3
        if prefix + 2 * (steps + 1) > e.max_sequence_tokens:
            return
        for physical in witnesses:
            if physical < 2:
                continue
            segment_width = (max(2, (physical + arena.spec.state_slot_capacity - 1)
                                      // arena.spec.state_slot_capacity)
                             if maintenance is not None else 2)
            widths = [segment_width] * (physical // segment_width)
            if physical % segment_width:
                widths.append(physical % segment_width)
            elif physical >= 4 and maintenance is None:
                widths[-1:] = [1, 1]
            if prefix + (steps + 1) * max(widths) > e.max_sequence_tokens:
                return
            owners = len(widths)
            if owners > arena.spec.state_slot_capacity:
                return
            ends, first = [], 0
            for width in widths:
                first += width
                ends.append(first - 1)
            provider_tables = (() if maintenance is None else tuple(
                tuple(owner + j * owners for j in range(
                    (prefix + (steps + 1) * width + e._block_size - 1) // e._block_size))
                for owner, width in enumerate(widths)))
            if provider_tables and max(max(table) for table in provider_tables) >= maintenance.provider.body.arena.spec.page_capacity:
                return
            tables = [[owner] for owner in range(owners)]
            owned = [{owner} for owner in range(owners)]
            next_page = owners
            templates, program = [], []
            initial_tables = None

            def token(owner, position):
                return (17 + owner * 7919 + position * 7) % e.vocab_size

            def allocate(owner):
                nonlocal next_page
                page = next_page
                next_page += 1
                owned[owner].add(page)
                return page

            for step in range(steps + 1):
                rows, coordinates = [], []
                for owner, width in enumerate(widths):
                    begin = prefix + step * width
                    first_page = begin // e._block_size
                    last_page = (begin + width - 1) // e._block_size
                    new = [owner] if step == 0 and not prefix else []
                    while len(tables[owner]) <= last_page:
                        page = allocate(owner)
                        tables[owner].append(page)
                        new.append(page)
                    sources = {j: tables[owner][j] for j in range(first_page, last_page + 1)}
                    targets = {j: allocate(owner) if width > 1 else sources[j]
                               for j in sources}
                    for j, page in targets.items():
                        tables[owner][j] = page
                    seen = set()
                    for local in range(width):
                        position = begin + local
                        j = position // e._block_size
                        source = sources[j] if j not in seen else targets[j]
                        reset = int(j not in seen and source in new)
                        seen.add(j)
                        rows.append(BatchRow(
                            f"segment-witness:{owner}", token(owner, position), position,
                            owner, owner, source, targets[j], tuple(tables[owner]),
                            reset_state=not prefix and step == 0 and local == 0,
                            new_pages=tuple(new) if local == 0 else (),
                            segment_length=width if local == 0 else 0,
                            is_decode=width == 1 and maintenance is None,
                            sample_ordinal=step if width == 1 and maintenance is None else 0,
                            mtp_position=position if maintenance is not None else -1,
                            mtp_page_destination=(provider_tables[owner][position // e._block_size]
                                                  if maintenance is not None else -1),
                            mtp_page_table=provider_tables[owner] if maintenance is not None else (),
                        ))
                        coordinates.append((source, targets[j], j, reset))
                templates.append(tuple(rows))
                if step:
                    program.append(tuple(coordinates))
                else:
                    initial_tables = tuple(tuple(pages) for pages in tables)
            if next_page > arena.spec.page_capacity:
                _LOGGER.info("resident page-program witness lacks private audit pages rank=%d M%d",
                             e._rank, e.captured_m)
                return
            assert initial_tables is not None

            def finish():
                e.slot.synchronize()
                torch.cuda.current_stream(e.body.hidden.device).synchronize()
                barrier(group=e._world_group)

            def replay(rows):
                if maintenance is not None:
                    from cubic_serve.runtime.performance import _replay_target_with_mtp_maintenance
                    _replay_target_with_mtp_maintenance(e, rows, (),
                        {e.captured_m: maintenance.provider}, e._target_hidden_slots)
                else:
                    e.bind_batch(rows)
                    e.slot.select_decode_overlap(False)
                    e.replay_bound()

            def snapshot():
                selected = sorted(set((0, owners // 2, owners - 1)))
                state_ids = torch.tensor(selected, device=arena.device)
                pages = torch.tensor(sorted({page for owner in selected for page in owned[owner]}),
                                     device=arena.device)
                columns = max(map(len, tables))
                table = b.page_table[:physical, :columns].detach().to("cpu", copy=True)
                first = 0
                for width, owner_table in zip(widths, tables, strict=True):
                    table[first:first + width, len(owner_table):] = -1
                    first += width
                values = {"hidden": e.body.hidden[:physical],
                          "position": b.position_ids[:, :physical],
                          "offsets": b.token_offsets[:physical], "table": table,
                          "ordinals": e._sampling.ordinals[:physical]}
                if e.body.logits is not None:
                    values["logits"] = e.body.logits[:physical]
                values.update({"state:" + name: value.index_select(0, state_ids)
                               for name, value in arena.states.items()})
                for name, value in (("key", arena.key), ("value", arena.value),
                                    ("written", arena.written), *arena.page_fields.items()):
                    values["page:" + name] = value.index_select(0, pages)
                if maintenance is not None:
                    provider_arena = maintenance.provider.body.arena
                    provider_ids = torch.tensor(sorted({page for owner in selected for page in provider_tables[owner]}), device=arena.device)
                    values["draft_feedback"] = maintenance.durable.index_select(0, state_ids)
                    pb=maintenance.provider.body.buffers
                    for name in ('token_ids','sequence_lengths','token_offsets','page_source',
                                 'page_destination','segment_lengths','row_activity','target_hidden'):
                        values['provider-input:'+name]=getattr(pb,name)[:physical]
                    values['provider-input:positions']=pb.position_ids[:,:physical]
                    for index,mapping in enumerate(pb.state_slot_maps):
                        values['provider-input:map:'+str(index)]=mapping[:,:,:physical]
                    values.update({"provider-state:"+name: value.index_select(0,state_ids)
                                   for name,value in provider_arena.states.items()})
                    for name,value in (("key",provider_arena.key),("value",provider_arena.value),
                                       ("written",provider_arena.written),*provider_arena.page_fields.items()):
                        values["provider-page:"+name] = value.index_select(0,provider_ids)
                return {name: value.detach().to("cpu", copy=True)
                        for name, value in values.items()}

            results, sequences = [], []
            for composed in (False, True):
                arena.reset_page_rows(torch.arange(next_page, device=arena.device))
                if maintenance is not None:
                    provider_arena=maintenance.provider.body.arena
                    provider_arena.reset_page_rows(torch.tensor(sorted({page for table in provider_tables for page in table}),device=arena.device))
                    provider_arena.reset_state_rows(torch.arange(owners,device=arena.device))
                    maintenance.durable[:owners].zero_()
                # Pre-readiness reference preparation may use a lower logical
                # row count on the same fixed graph; it never exposes a new
                # performance bucket or performs runtime capture.
                for position in range(prefix):
                    replay(tuple(BatchRow(
                        f"segment-witness:{owner}", token(owner, position), position,
                        owner, owner, owner, owner, (owner,),
                        reset_state=position == 0, new_pages=(owner,) if position == 0 else (),
                        mtp_position=position if maintenance is not None else -1,
                        mtp_page_destination=provider_tables[owner][position // e._block_size] if maintenance is not None else -1,
                        mtp_page_table=provider_tables[owner] if maintenance is not None else (),
                    ) for owner in range(owners)))
                initial = templates[0]
                replay(initial)
                e.sample_tokens(initial)
                sampled = e._token_ids[:physical].cpu().tolist()
                previous = tuple(sampled[row] for row in ends)
                finish()
                if composed:
                    remaining, inputs, positions, strides = [], [], [], []
                    for owner, width in enumerate(widths):
                        for local in range(width):
                            remaining.append(steps + 1 if width > 1 or maintenance is not None else 0)
                            inputs.append(tuple(token(owner, prefix + (step + 1) * width + local)
                                                for step in range(steps)) if width > 1 or maintenance is not None else ())
                            positions.append(tuple((prefix + (step + 1) * width + local,) * 3
                                                   for step in range(steps)) if width > 1 or maintenance is not None else ())
                            strides.append(width)
                    if maintenance is not None:
                        maintenance.bind(tuple(widths),provider_tables)
                    self.bind(
                        owners, steps, ((),) * owners,
                        physical_rows=physical, output_rows=tuple(ends),
                        logical_steps=tuple(strides), page_program=tuple(program),
                        source_pages=tuple(pages[-1] for pages in initial_tables),
                        owner_offsets=tuple((prefix + width - 1) % e._block_size for width in widths),
                        owner_page_counts=tuple(map(len, initial_tables)),
                        prompt_remaining=tuple(remaining), prompt_tokens=tuple(inputs),
                        prompt_positions=tuple(positions),
                    )
                    self.replay()
                    outputs, _ = self.consume(owners)
                else:
                    outputs = []
                    for rows in templates[1:]:
                        changed = list(rows)
                        for owner, (width, last) in enumerate(zip(widths, ends, strict=True)):
                            if width == 1 and maintenance is None:
                                changed[last] = replace(changed[last], token_id=previous[owner])
                        rows = tuple(changed)
                        replay(rows)
                        e.sample_tokens(rows)
                        sampled = e._token_ids[:physical].cpu().tolist()
                        previous = tuple(sampled[row] for row in ends)
                        outputs.append(list(previous))
                finish()
                results.append(snapshot())
                sequences.append(outputs)
            if sequences[0] != sequences[1] or not _exact_snapshot(*results):
                differences = [name for name in results[0]
                               if not _exact_snapshot({name: results[0][name]},
                                                      {name: results[1][name]})]
                raise RuntimeError("segment page program differs from ordinary binding: "
                                   + ",".join(differences))
            _LOGGER.info("resident cross-page segment composition qualified rank=%d M%d logical=%d owners=%d",
                         e._rank, e.captured_m, physical, owners)
        if e.captured_m >= 2:
            self.segmented_program_qualified = True
            if maintenance is not None:
                self.qualified = True
                self.program_qualified = True
        e._bound_active_rows = e.captured_m

    def supports(self, command):
        """Select by resident capabilities and explicit resource capacity."""
        return (
            execution_ready(self)
            and self.e._sampling.all_greedy != self.stochastic
            and (not self.stochastic or not command.owner_lengths)
            and (self.e._target_hidden_slots is None or (
                self.draft_maintenance is not None and bool(command.provider_tables)
                and bool(command.owner_lengths) and bool(command.prompt_remaining)
                and all(remaining > command.max_steps * width for remaining, width in zip(
                    command.prompt_remaining, command.owner_lengths, strict=True))))
            and (not command.prompt_remaining or execution_ready(self, "program_qualified"))
            and (not command.owner_lengths
                 or execution_ready(self, "segmented_program_qualified"))
            and sum(map(len, command.eos_token_ids)) <= self.eos_values.numel()
        )

    def run(self, command, **kwargs):
        from cubic_serve.runtime.continuous_batch import (
            BatchProtocolError,
            BatchResult,
            completed_resident_pages,
        )

        if not self.supports(command):
            raise BatchProtocolError("resident graph is not qualified for this transaction")
        if self.draft_maintenance is not None:
            self.draft_maintenance.bind(command.owner_lengths, command.provider_tables)
        rows = len(kwargs["request_ids"])
        owner_lengths = command.owner_lengths
        segment_binding = {}
        if owner_lengths:
            physical_tokens, physical_positions, physical_remaining = [], [], []
            output_rows, logical_steps = [], []
            physical = 0
            for width, remaining, tokens, positions in zip(
                owner_lengths, command.prompt_remaining,
                command.prompt_tokens, command.prompt_positions, strict=True,
            ):
                for offset in range(width):
                    physical_tokens.append(tokens[offset::width])
                    physical_positions.append(positions[offset::width])
                    physical_remaining.append((remaining + width - 1) // width)
                logical_steps.extend((width,) * width)
                physical += width
                output_rows.append(physical - 1)
            segment_binding = dict(
                physical_rows=physical, output_rows=tuple(output_rows),
                logical_steps=tuple(logical_steps), page_program=command.page_program,
                source_pages=command.source_pages,
                owner_offsets=kwargs["offsets"], owner_page_counts=kwargs["page_counts"],
            )
            prompt_remaining = tuple(physical_remaining)
            prompt_tokens = tuple(physical_tokens)
            prompt_positions = tuple(physical_positions)
        else:
            prompt_remaining = command.prompt_remaining
            prompt_tokens = command.prompt_tokens
            prompt_positions = command.prompt_positions
        schedule = (
            tuple((0, *entry) for entry in command.page_updates) + command.future_page_updates
        )
        self.bind(
            rows,
            command.max_steps,
            command.eos_token_ids or ((),) * rows,
            schedule,
            duration=command.max_duration_seconds,
            prompt_remaining=prompt_remaining,
            prompt_tokens=prompt_tokens,
            prompt_positions=prompt_positions,
            **segment_binding,
        )
        started = time.monotonic_ns()
        self.replay()
        index = 0
        published = started
        if kwargs["rank"] == kwargs["sampler_rank"]:
            deadline = time.monotonic() + kwargs["timeout"]
            while True:
                while int(self.host_flags[0]) <= index:
                    if kwargs["interrupt"] is not None and kwargs["interrupt"].is_set():
                        self.host_flags[2] = 1
                    if time.monotonic() > deadline:
                        raise BatchProtocolError("resident publication timed out")
                    time.sleep(0.0001)
                begin = 64 + index * (self.e.captured_m + 3)
                tokens = tuple(int(x) for x in self.host_array[begin : begin + rows])
                duration = int(self.host_array[begin + self.e.captured_m]) / 1e6
                complete = bool(self.host_array[begin + self.e.captured_m + 1])
                if self.host_array[begin + self.e.captured_m + 2]:
                    raise BatchProtocolError("resident sampling rejected its step")
                if complete:
                    break
                now = time.monotonic_ns()
                kwargs["publish"](
                    BatchResult(
                        kwargs["rank"],
                        command.batch_id,
                        kwargs["request_ids"],
                        tokens,
                        target_ms=duration,
                        step_ms=(now - published) / 1e6,
                        continuation_index=index,
                        continuation_complete=False,
                        gpu_step_intervals_ms=(),
                    )
                )
                published = now
                index += 1
        outputs, costs = self.consume(rows)
        self.e.slot.synchronize()
        count = len(outputs)
        self._completed_steps = count
        if owner_lengths:
            advanced = tuple(offset + count * width for offset, width in zip(
                kwargs["offsets"], owner_lengths, strict=True))
            completed = (
                tuple(offset % self.e._block_size for offset in advanced),
                tuple(pages + offset // self.e._block_size for pages, offset in zip(
                    kwargs["page_counts"], advanced, strict=True)),
            )
            self.e._resident_owner_pages = tuple(
                command.page_program[count - 1][row][1] for row in output_rows
            )
        else:
            completed = completed_resident_pages(
                kwargs["offsets"], kwargs["page_counts"], schedule, count
            )
        # Every rank consumes its completed page metadata before acknowledging
        # the topology transaction. The final receipt may release its lease.
        kwargs["completion_barrier"].wait(kwargs["timeout"])
        if kwargs["rank"] == kwargs["sampler_rank"]:
            if index != count - 1:
                raise BatchProtocolError("resident publication ordinal differs from completion")
            kwargs["publish"](
                BatchResult(
                    kwargs["rank"],
                    command.batch_id,
                    kwargs["request_ids"],
                    tuple(outputs[-1]),
                    target_ms=costs[-1],
                    step_ms=(time.monotonic_ns() - published) / 1e6,
                    continuation_index=index,
                    continuation_complete=True,
                    gpu_step_intervals_ms=self._gpu_step_intervals_ms,
                )
            )
        return completed
