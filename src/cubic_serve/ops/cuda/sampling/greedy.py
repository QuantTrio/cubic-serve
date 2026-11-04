"""Startup-owned, independently qualified graphs for temperature-zero sampling.

This operator owns no model, checkpoint-format or hardware-generation policy.
Its explicit active prefix describes sampling row ownership, independently of
sequence segmentation. Non-greedy and speculative candidate paths keep their
existing sampling contracts. Their caller-owned candidate phases reuse the
same preparation math and native final guard inside a combined startup graph.
"""

from __future__ import annotations

from cubic_serve.ops.cuda.collective import all_gather_single, broadcast
from cubic_serve.executor.readiness import execution_ready

from pathlib import Path
from typing import Any, cast

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup

from cubic_serve.generation.tp_sampling import (
    ShardedSamplingParameters,
    ShardedSamplingWorkspace,
    TensorParallelSamplingError,
    _group_ranks,
    _padded_vocab_size,
    _require_buffer,
    _require_sampling_workspace,
)
from cubic_serve.models.parallel_weights import AxisSpan
from cubic_serve.ops.cuda.compiled_artifact import bind_tensor
from cubic_serve.ops.cuda.native import NativeLaunch
from cubic_serve.ops.cuda.native_compiler import compile_native

PRE_METRIC = "greedy_prepare_gather_root_graph_gpu_ms"
POST_METRIC = "greedy_broadcast_guard_graph_gpu_ms"


class CapturedGreedySampler:
    """Two fixed-capacity phases, captured before readiness and leased by caller.

    Candidate preparation/gather/root and world broadcast/final guard are
    separate phases so PP ranks enter collectives in the same startup order.
    Graph destruction must precede process-group destruction.
    """

    def __init__(
        self,
        *,
        local_logits: Tensor | None,
        local_vocab_span: AxisSpan | None,
        final_ranks: tuple[int, ...],
        final_vocab_spans: tuple[AxisSpan, ...],
        true_vocab_size: int,
        token_ids: Tensor,
        parameters: ShardedSamplingParameters,
        workspace: ShardedSamplingWorkspace | None,
        final_tp_group: ProcessGroup | None,
        world_group: ProcessGroup,
        root_valid: Tensor | None,
        capture_stream: torch.cuda.Stream,
        token_mask: Tensor | None = None,
        constrained_rows: Tensor | None = None,
        masked_logits: Tensor | None = None,
        graph_pool: Any = None,
        consume_logits: bool = False,
    ) -> None:
        self.device = token_ids.device
        self.rank = dist.get_rank()
        self.root_rank = final_ranks[0]
        self.capacity = token_ids.numel()
        if (
            self.device.type != "cuda"
            or self.device.index is None
            or token_ids.ndim != 1
            or self.capacity < 1
            or tuple(sorted(set(final_ranks))) != final_ranks
            or len(final_ranks) != len(final_vocab_spans)
            or self.rank not in _group_ranks(world_group, "world")
            or not set(final_ranks).issubset(_group_ranks(world_group, "world"))
            or not 0 < true_vocab_size <= _padded_vocab_size(final_vocab_spans)
        ):
            raise TensorParallelSamplingError("greedy graph topology or capacity is invalid")
        self._call: dict[str, Any] = dict(
            local_logits=local_logits,
            local_vocab_span=local_vocab_span,
            final_ranks=final_ranks,
            final_vocab_spans=final_vocab_spans,
            true_vocab_size=true_vocab_size,
            token_ids=token_ids,
            parameters=parameters,
            workspace=workspace,
            final_tp_group=final_tp_group,
            world_group=world_group,
            device=self.device,
            root_valid=root_valid,
        )
        _require_buffer(
            token_ids,
            shape=(self.capacity,),
            dtype=torch.int64,
            device=self.device,
            label="greedy token buffer",
        )
        # Import locally: generation declares the optional plan using TYPE_CHECKING.
        from cubic_serve.generation.tp_sampling import _require_sampling_parameter_buffers

        _require_sampling_parameter_buffers(parameters, capacity=self.capacity, device=self.device)
        if parameters.seeds is None or parameters.ordinals is None:
            raise TensorParallelSamplingError("greedy graphs require seed and ordinal buffers")
        self.local_logits = local_logits
        self.token_mask = token_mask
        self.constrained_rows = constrained_rows
        self._grammar_operands = (token_mask, constrained_rows)
        if token_mask is not None:
            _require_buffer(
                token_mask,
                shape=(self.capacity, (true_vocab_size + 31) // 32),
                dtype=torch.int32,
                device=self.device,
                label="grammar bitmask",
            )
        self.width = 0
        if self.rank in final_ranks:
            expected = final_vocab_spans[final_ranks.index(self.rank)]
            if (
                local_logits is None
                or local_vocab_span != expected
                or workspace is None
                or final_tp_group is None
                or _group_ranks(final_tp_group, "final TP") != final_ranks
                or local_logits.dtype not in (torch.bfloat16, torch.float16)
            ):
                raise TensorParallelSamplingError("greedy graph logits ownership is invalid")
            _require_buffer(
                local_logits,
                shape=(self.capacity, expected.size),
                dtype=local_logits.dtype,
                device=self.device,
                label="greedy logits",
            )
            _require_sampling_workspace(
                workspace,
                tp_size=len(final_ranks),
                capacity=self.capacity,
                candidate_count=parameters.max_top_k,
                device=self.device,
                dtype=local_logits.dtype,
            )
            self.width = max(0, min(expected.stop, true_vocab_size) - expected.start)
        elif any(
            v is not None for v in (local_logits, local_vocab_span, workspace, final_tp_group)
        ):
            raise TensorParallelSamplingError("non-final ranks cannot own greedy logits")
        if self.rank == self.root_rank:
            if root_valid is None:
                raise TensorParallelSamplingError("greedy root requires validity storage")
            _require_buffer(
                root_valid,
                shape=(self.capacity,),
                dtype=torch.bool,
                device=self.device,
                label="greedy root validity",
            )
        elif root_valid is not None:
            raise TensorParallelSamplingError("only the greedy root owns validity storage")
        self.metadata = torch.zeros(2, dtype=torch.int32, device=self.device)
        # Deferred proposal guards allow several differently sized waves to
        # enqueue before any host wait. Each metadata H2D therefore reads an
        # immutable pinned row, never a two-element buffer rewritten by the
        # following wave. This bounded table adds no execution graph or wait.
        self._metadata_hosts = torch.empty(
            (self.capacity + 1, 2, 2), dtype=torch.int32, pin_memory=True
        )
        self._metadata_hosts[:, :, 0] = torch.arange(self.capacity + 1)[:, None]
        self._metadata_hosts[:, :, 1] = torch.tensor([0, 1])[None, :]
        self._metadata_key: tuple[int, bool] | None = None
        self.final_valid = torch.empty((), dtype=torch.bool, device=self.device)
        self.stream = capture_stream
        if self.stream.device != self.device:
            raise TensorParallelSamplingError("greedy capture stream belongs to another device")
        self.graphs: list[torch.cuda.CUDAGraph] = []
        self.calls: list[NativeLaunch] = []
        self._closed = False
        self.qualified = False
        self.evidence: dict[str, Any] = {}
        tile = 4096
        tiles = max(1, (self.width + tile - 1) // tile)
        self._scratch = tuple(
            torch.empty((self.capacity, tiles), dtype=dtype, device=self.device)
            for dtype in (torch.float32, torch.int32, torch.int32)
        )
        self._partial_scores, self._partial_ids, self._partial_finite = self._scratch
        source = Path(__file__).with_suffix(".cu").read_bytes()
        defines = dict(
            ROWS=self.capacity,
            CAPACITY=self.capacity,
            TP=len(final_ranks),
            K=parameters.max_top_k,
            CANDIDATES=parameters.max_top_k,
            WIDTH=self.width,
            STRIDE=0 if local_logits is None else local_logits.shape[1],
            TILE=tile,
            TILES=tiles,
            VOCAB_START=0 if local_vocab_span is None else local_vocab_span.start,
            VOCAB_SIZE=true_vocab_size,
            ROOT=int(self.rank == self.root_rank),
            SCORE_HALF=int(local_logits is not None and local_logits.dtype == torch.float16),
            HAS_GRAMMAR=int(token_mask is not None),
            MASK_WORDS=(true_vocab_size + 31) // 32,
        )

        def bind(
            phase: int,
            operands: tuple[Tensor | None, ...],
            grid: tuple[int, int, int],
            threads: int,
        ) -> NativeLaunch:
            prefix = "".join(f"#define {k} {v}\n" for k, v in (defines | {"PHASE": phase}).items())
            module, entry = compile_native(
                (("entry.cu", prefix.encode() + source),), self.device, "greedy_sampling"
            )
            call = module.bind(
                entry,
                operands,
                operator_id=f"sampling.greedy.phase{phase}",
                grid=grid,
                block=(threads, 1, 1),
            )
            self.calls.append(call)
            return call

        self._partial = self._merge = self._root = None
        self._candidate_preparation = None
        if workspace is not None:
            count = min(parameters.max_top_k, self.width)
            if count:
                from cubic_serve.ops.cuda.sampling.local_candidates import LocalCandidatePlan

                assert local_logits is not None and local_vocab_span is not None
                self._candidate_preparation = LocalCandidatePlan(
                    local_logits=local_logits,
                    workspace=workspace,
                    active_rows=self.capacity,
                    visible_width=self.width,
                    local_count=count,
                    vocab_start=local_vocab_span.start,
                    metadata=self.metadata,
                    token_mask=token_mask,
                    masked_logits=masked_logits,
                    consume_logits=consume_logits,
                )
            self._partial = bind(
                0,
                (
                    local_logits,
                    self.metadata,
                    self._partial_scores,
                    self._partial_ids,
                    self._partial_finite,
                    *((token_mask,) if token_mask is not None else ()),
                ),
                (self.capacity, tiles, 1),
                256,
            )
            self._merge = bind(
                1,
                (
                    self._partial_scores,
                    self._partial_ids,
                    self._partial_finite,
                    self.metadata,
                    workspace.local_scores,
                    workspace.local_token_ids,
                ),
                (self.capacity, 1, 1),
                256,
            )
        if self.rank == self.root_rank:
            assert workspace is not None
            self._root = bind(
                2,
                (
                    workspace.gathered_scores,
                    workspace.gathered_token_ids,
                    parameters.temperatures,
                    parameters.top_ps,
                    parameters.top_ks,
                    parameters.seeds,
                    parameters.ordinals,
                    self.metadata,
                    token_ids,
                    root_valid,
                ),
                (self.capacity, 1, 1),
                128,
            )
        self._final = bind(
            3,
            (
                token_ids,
                root_valid,
                self.metadata,
                self.final_valid,
                parameters.temperatures,
                parameters.top_ps,
                parameters.top_ks,
                parameters.seeds,
                parameters.ordinals,
            ),
            (1, 1, 1),
            128,
        )
        self._operands = tuple(
            dict.fromkeys(
                [t for call in self.calls for t in call.tensors]
                + ([constrained_rows] if constrained_rows is not None else [])
            )
        )
        self.grammar_candidate_graph = None
        self.grammar_random_graph = None
        if token_mask is not None:
            from cubic_serve.ops.cuda.sampling.topk_topp import RootTopKTopPSamplingPrimitive

            self._grammar_root = None
            if self.rank == self.root_rank:
                assert workspace is not None and root_valid is not None
                self._grammar_root = RootTopKTopPSamplingPrimitive(
                    workspace.gathered_scores,
                    workspace.gathered_token_ids,
                    parameters.temperatures,
                    parameters.top_ps,
                    parameters.top_ks,
                    parameters.seeds,
                    parameters.ordinals,
                    token_ids,
                    root_valid,
                    allow_underfilled=True,
                    constrained_rows=constrained_rows,
                )
        self._bindings = tuple(bind_tensor(t) for t in self._operands)
        if len({t.untyped_storage().data_ptr() for t in self._operands}) != len(self._operands):
            raise TensorParallelSamplingError("greedy graph operands cannot alias")
        try:
            self.stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(self.stream):
                for phase in (self._prepare, self._broadcast):
                    phase()
                    self.stream.synchronize()
                    graph = torch.cuda.CUDAGraph(keep_graph=True)
                    with torch.cuda.graph(graph, stream=self.stream, pool=graph_pool):
                        phase()
                    graph.instantiate()
                    self.graphs.append(graph)
                if token_mask is not None:
                    for phase in (self.prepare_candidates, self._grammar_sample):
                        phase()
                        self.stream.synchronize()
                        graph = torch.cuda.CUDAGraph(keep_graph=True)
                        with torch.cuda.graph(graph, stream=self.stream, pool=graph_pool):
                            phase()
                        graph.instantiate()
                        if self.grammar_candidate_graph is None:
                            self.grammar_candidate_graph = graph
                        else:
                            self.grammar_random_graph = graph
            torch.cuda.current_stream(self.device).wait_stream(self.stream)
        except BaseException:
            self.close()
            raise

    def _prepare(self) -> None:
        workspace = self._call["workspace"]
        if workspace is not None:
            assert self._partial is not None and self._merge is not None
            self._partial.launch()
            self._merge.launch()
            all_gather_single(
                workspace.gathered_scores.view(-1),
                workspace.local_scores.view(-1),
                group=self._call["final_tp_group"],
            )
            all_gather_single(
                workspace.gathered_token_ids.view(-1),
                workspace.local_token_ids.view(-1),
                group=self._call["final_tp_group"],
            )
            if self._root is not None:
                self._root.launch()

    def prepare_candidates(self, *, temperatures: Tensor | None = None) -> None:
        """Device-active top-k candidates for a caller-owned captured transaction.

        Reuses the ordinary sampler's exact local top-k/greedy math and TP
        candidate ABI. Unlike standalone replay, the logical count is device
        metadata, so temporal graph compositions need no host row materialization.
        The enclosing graph owns these operations and their temporary storage.
        """
        workspace = self._call["workspace"]
        if workspace is None:
            return
        parameters = self._call["parameters"]
        temperatures = parameters.temperatures if temperatures is None else temperatures
        if self._candidate_preparation is not None:
            self._candidate_preparation.launch(temperatures=temperatures)
        else:
            workspace.local_scores.fill_(float("nan"))
            workspace.local_token_ids.fill_(-1)
        all_gather_single(
            workspace.gathered_scores.view(-1),
            workspace.local_scores.view(-1),
            group=self._call["final_tp_group"],
        )
        all_gather_single(
            workspace.gathered_token_ids.view(-1),
            workspace.local_token_ids.view(-1),
            group=self._call["final_tp_group"],
        )

    def _grammar_sample(self) -> None:
        self.prepare_candidates()
        self.sample_candidates(self._grammar_root, constrained_rows=self.constrained_rows)

    def replay_candidates(self, count: int) -> None:
        if not execution_ready(self) or self.grammar_candidate_graph is None:
            raise TensorParallelSamplingError("masked candidate graph is not qualified")
        self._set_metadata(count, False)
        self.grammar_candidate_graph.replay()

    def sample_candidates(
        self, root_sampler: Any, *, constrained_rows: Tensor | None = None
    ) -> None:
        """Existing FP32 top-k/top-p/Philox math, with inactive tokens sentinelized."""
        if self.rank != self.root_rank:
            return
        w, p = self._call["workspace"], self._call["parameters"]
        launch_kwargs = {} if constrained_rows is None else {"constrained_rows": constrained_rows}
        root_sampler.launch(
            w.gathered_scores,
            w.gathered_token_ids,
            p.temperatures,
            p.top_ps,
            p.top_ks,
            p.seeds,
            p.ordinals,
            self._call["token_ids"],
            self._call["root_valid"],
            **launch_kwargs,
        )
        active = torch.arange(self.capacity, device=self.device) < self.metadata[0]
        self._call["token_ids"].masked_fill_(~active, -1)

    def _broadcast(self) -> None:
        broadcast(
            self._call["token_ids"], group=self._call["world_group"], group_src=self.root_rank
        )
        self._final.launch()

    def _reference_stochastic(
        self, count: int, root_sampler: Any, *, validity_out: Tensor | None = None,
    ) -> Tensor:
        """Pre-readiness reference, using the canonical uncaptured candidate math.

        Keep the physical top-k geometry fixed on both sides of a temporal
        composition proof. Logical ownership remains an explicit activity mask.
        This method neither grants readiness nor supplies a serving fallback.
        """
        from cubic_serve.generation.tp_sampling import (
            _prepare_local_candidates,
            sample_sharded_broadcast_tp,
        )

        if self._closed or not 0 < count <= self.capacity:
            raise TensorParallelSamplingError("reference sampling has invalid ownership")
        if torch.cuda.is_current_stream_capturing():
            raise TensorParallelSamplingError("reference sampling cannot run in a capture")
        workspace = self._call["workspace"]
        if workspace is not None:
            workspace.local_scores.fill_(float("nan"))
            workspace.local_token_ids.fill_(-1)
            local_count = min(self._call["parameters"].max_top_k, self.width)
            if local_count:
                assert self.local_logits is not None
                _prepare_local_candidates(
                    local_logits=self.local_logits,
                    workspace=workspace,
                    temperatures=self._call["parameters"].temperatures,
                    active_rows=self.capacity,
                    visible_width=self.width,
                    local_count=local_count,
                    vocab_start=self._call["local_vocab_span"].start,
                    row_activity=torch.arange(self.capacity, device=self.device) < count,
                    token_mask=self.token_mask,
                )
            all_gather_single(workspace.gathered_scores.view(-1),
                                   workspace.local_scores.view(-1),
                                   group=self._call["final_tp_group"])
            all_gather_single(workspace.gathered_token_ids.view(-1),
                                   workspace.local_token_ids.view(-1),
                                   group=self._call["final_tp_group"])
        self.invalidate_metadata()
        return sample_sharded_broadcast_tp(
            **self._call, active_rows=count, root_sampler=root_sampler,
            candidates_pre_gathered=True, all_greedy_rows=False,
            validity_out=validity_out,
        )

    def validate_call(self, **kwargs: Any) -> None:
        if self.token_mask is not None and (
            self.token_mask is not self._grammar_operands[0]
            or self.constrained_rows is not self._grammar_operands[1]
        ):
            raise TensorParallelSamplingError("grammar replay operands changed after startup")
        for key, expected in self._call.items():
            actual = kwargs[key]
            if isinstance(
                expected,
                Tensor | ShardedSamplingParameters | ShardedSamplingWorkspace | ProcessGroup,
            ):
                matches = actual is expected
            else:
                matches = actual == expected
            if not matches:
                raise TensorParallelSamplingError("greedy replay differs from its startup binding")
        if tuple(bind_tensor(t) for t in self._operands) != self._bindings:
            raise TensorParallelSamplingError("greedy replay operands changed after startup")

    def invalidate_metadata(self) -> None:
        """Require ingress after a caller graph writes shared sampler metadata."""
        self._metadata_key = None

    def _set_metadata(self, count: int, cancel: bool) -> None:
        if (
            type(count) is not int
            or not 0 < count <= self.capacity
            or type(cancel) is not bool
            or (cancel and self.rank != self.root_rank)
        ):
            raise TensorParallelSamplingError("sampling metadata is invalid")
        key = (count, cancel)
        if key != self._metadata_key:
            self.metadata.copy_(self._metadata_hosts[count, int(cancel)], non_blocking=True)
            self._metadata_key = key

    def _replay(
        self,
        count: int,
        cancel: bool,
        profiler: Any = None,
        *,
        validity_out: Tensor | None = None,
        stochastic: bool = False,
    ) -> Tensor:
        if (
            self._closed
            or len(self.graphs) != 2
            or type(count) is not int
            or not 0 < count <= self.capacity
            or type(cancel) is not bool
            or (cancel and self.rank != self.root_rank)
        ):
            raise TensorParallelSamplingError("greedy graph replay is unavailable")
        if validity_out is not None:
            _require_buffer(
                validity_out,
                shape=(),
                dtype=torch.bool,
                device=self.device,
                label="deferred sampling validity",
            )
            if any(
                validity_out.untyped_storage().data_ptr() == t.untyped_storage().data_ptr()
                for t in (*self._operands, self.final_valid)
            ):
                raise TensorParallelSamplingError("deferred sampling validity must be private")
        self._set_metadata(count, cancel)
        if stochastic and self.grammar_random_graph is None:
            raise TensorParallelSamplingError("masked stochastic graph is not available")
        graphs = (self.grammar_random_graph, self.graphs[1]) if stochastic else self.graphs
        for graph, metric in zip(graphs, (PRE_METRIC, POST_METRIC), strict=True):
            assert graph is not None
            if profiler is not None:
                profiler.begin_cuda(metric)
            graph.replay()
            if profiler is not None:
                profiler.end_cuda(metric)
        if validity_out is None:
            valid = bool(self.final_valid.item())
        else:
            validity_out.copy_(self.final_valid)
            valid = True
        if profiler is not None:
            profiler.flush_cuda((PRE_METRIC, POST_METRIC))
        if not valid:
            raise TensorParallelSamplingError("TP sampling rejected invalid candidates or tokens")
        return cast(Tensor, self._call["token_ids"])

    def replay(
        self,
        count: int,
        cancel: bool,
        profiler: Any = None,
        *,
        validity_out: Tensor | None = None,
        stochastic: bool = False,
    ) -> Tensor:
        if not execution_ready(self):
            raise TensorParallelSamplingError("greedy graph has no independent qualification")
        return self._replay(
            count, cancel, profiler, validity_out=validity_out, stochastic=stochastic
        )

    def qualify(self) -> None:
        """Independent CPU argmax on exact operands, after each completed replay."""
        from cubic_serve.guardrail.quantized_graph import byte_exact

        if self._closed:
            raise TensorParallelSamplingError("closed greedy graph cannot be qualified")
        p = self._call["parameters"]
        logits = self._call["local_logits"]
        protected = [p.temperatures, p.top_ps, p.top_ks, p.seeds, p.ordinals]
        if logits is not None:
            protected.append(logits)
        if self.token_mask is not None:
            protected.append(self.token_mask)
        if self.constrained_rows is not None:
            protected.append(self.constrained_rows)
        restored = tuple(dict.fromkeys((*self._operands, *protected)))
        saved = [t.clone() for t in restored]
        rng = torch.cuda.get_rng_state(self.device)
        cases = []
        # Keep one private pre-replay image per protected operand. These are
        # exact storage witnesses, not numerical oracles. Admit their complete
        # payload against current device/process headroom and leave seven
        # eighths for the existing graphs, candidate preparation and workspace.
        free, total = torch.cuda.mem_get_info(self.device)
        limit = int(total * torch.cuda.get_per_process_memory_fraction(self.device))
        headroom = max(0, min(free, limit - torch.cuda.memory_reserved(self.device)))
        image_bytes = sum(t.numel() * t.element_size() for t in protected)
        image_device = self.device if image_bytes <= headroom // 8 else torch.device("cpu")
        before = [torch.empty_like(t, device=image_device).view(torch.uint8) for t in protected]
        try:
            if self.token_mask is not None:
                self.token_mask.fill_(-1)
            if self._candidate_preparation is not None:
                self._candidate_preparation.qualify()
            counts = sorted({1, max(1, self.capacity // 2), self.capacity})
            for count, case in [(n, tag) for n in counts for tag in ("finite", "ties")] + [
                (1, "invalid_seed"),
                (1, "nonfinite_owned_shard"),
                (1, "cancel"),
            ]:
                p.temperatures.zero_()
                p.top_ps.fill_(0.95)
                p.top_ks.fill_(p.max_top_k)
                p.seeds.fill_(1)
                p.ordinals.zero_()
                reference = None
                if logits is not None:
                    cpu = torch.full(logits.shape, float("nan"), dtype=logits.dtype)
                    if self.width:
                        columns = torch.arange(self.width)
                        for row in range(count):
                            cpu[row, : self.width] = ((columns + row * 7) % 19 - 9).float()
                            if case == "ties":
                                cpu[row, 0] = cpu[row, self.width - 1] = 12
                        if case == "nonfinite_owned_shard" and self.rank == self.root_rank:
                            cpu[0, 0] = float("nan")
                        scores, ids = cpu[:count, : self.width].float().max(dim=-1)
                        finite = torch.isfinite(cpu[:count, : self.width]).all(dim=-1)
                        reference = [
                            (float(s), int(i) + self._call["local_vocab_span"].start, bool(f))
                            for s, i, f in zip(scores, ids, finite, strict=True)
                        ]
                    logits.copy_(cpu)
                if case == "invalid_seed":
                    p.seeds[0] = -1
                # All ranks prove the same ordered fixture schedule before diagnostics.
                witnesses: list[Any] = [None] * dist.get_world_size()
                dist.all_gather_object(
                    witnesses,
                    (self.capacity, count, case, reference),
                    group=self._call["world_group"],
                )
                if any(w[:3] != witnesses[0][:3] for w in witnesses):
                    raise TensorParallelSamplingError("greedy audit rank schedule mismatch")
                expected = []
                should_reject = case in ("invalid_seed", "nonfinite_owned_shard")
                for row in range(count):
                    candidates = [w[3][row] for w in witnesses if w[3] is not None]
                    should_reject |= any(not item[2] for item in candidates)
                    if should_reject:
                        continue
                    maximum = max(item[0] for item in candidates)
                    expected.append(min(item[1] for item in candidates if item[0] == maximum))
                for t, image in zip(protected, before, strict=True):
                    image.copy_(t.view(torch.uint8))
                try:
                    actual = self._replay(count, case == "cancel" and self.rank == self.root_rank)
                    rejected = False
                    actual_cpu = actual.cpu()
                except TensorParallelSamplingError:
                    rejected = True
                    actual_cpu = None
                outcomes: list[Any] = [None] * dist.get_world_size()
                dist.all_gather_object(outcomes, rejected, group=self._call["world_group"])
                if any(outcome != should_reject for outcome in outcomes):
                    raise TensorParallelSamplingError(
                        "greedy audit rejection differs from reference"
                    )
                if actual_cpu is not None:
                    expected += [
                        (-2 if case == "cancel" and i == count else -1)
                        for i in range(count, self.capacity)
                    ]
                    if actual_cpu.tolist() != expected:
                        raise TensorParallelSamplingError("greedy audit token or padding mismatch")
                if any(
                    not byte_exact(t.to(b.device).view(torch.uint8), b)
                    for t, b in zip(protected, before, strict=True)
                ):
                    raise TensorParallelSamplingError("greedy audit changed protected operands")
                cases.append((count, case, "reject" if rejected else "exact"))
            if self.token_mask is not None:
                cases.extend(self._qualify_grammar(counts))
            if not torch.equal(torch.cuda.get_rng_state(self.device), rng):
                raise TensorParallelSamplingError("greedy audit advanced RNG state")
            self.evidence = dict(
                oracle="independent CPU FP32 argmax on exact BF16/FP16 operands",
                rank=self.rank,
                capacity=self.capacity,
                cases=cases,
                protected_operands=True,
                protected_operand_image_device=image_device.type,
                unchanged_rng=True,
                launch_signatures=[call.signature for call in self.calls],
                local_candidates=(None if self._candidate_preparation is None else
                                  self._candidate_preparation.evidence),
            )
            self.qualified = True
        finally:
            for operand, original in zip(restored, saved, strict=True):
                operand.copy_(original)
            torch.cuda.current_stream(self.device).synchronize()
            self._metadata_key = None

    def _qualify_grammar(self, counts: list[int]) -> list[tuple[int, str, str]]:
        """Independent singleton-language witness, including empty and excluded NaN."""
        p = self._call["parameters"]
        assert self.token_mask is not None
        receipts = []
        if self.constrained_rows is not None:
            self.constrained_rows.fill_(True)
        for count in counts:
            expected = [(row * 37 + 13) % self._call["true_vocab_size"] for row in range(count)]
            cpu_mask = torch.zeros_like(self.token_mask, device="cpu")
            for row, token in enumerate(expected):
                word = 1 << (token % 32)
                cpu_mask[row, token // 32] = word if word < 2**31 else word - 2**32
            self.token_mask.copy_(cpu_mask)
            if self.local_logits is not None:
                self.local_logits.fill_(float("nan"))
                self.local_logits[:count, : self.width].fill_(2.0)
            p.top_ks.fill_(p.max_top_k)
            p.top_ps.fill_(0.95)
            p.seeds.fill_(1)
            p.ordinals.zero_()
            for stochastic in (False, True):
                p.temperatures.fill_(0.7 if stochastic else 0.0)
                result = self._replay(count, False, stochastic=stochastic).cpu().tolist()
                if result != expected + [-1] * (self.capacity - count):
                    raise TensorParallelSamplingError(
                        "masked sampler differs from singleton language"
                    )
                if not torch.equal(self.token_mask.cpu(), cpu_mask):
                    raise TensorParallelSamplingError("sampling mutated its grammar bitmask")
                receipts.append(
                    (count, "grammar_random" if stochastic else "grammar_greedy", "exact")
                )
            for kind in ("empty", "excluded_nonfinite"):
                if kind == "empty":
                    self.token_mask.zero_()
                else:
                    self.token_mask.copy_(cpu_mask)
                    if self.rank == self.root_rank and self.width:
                        assert self.local_logits is not None
                        self.local_logits[0, 0] = float("nan")
                for stochastic in (False, True):
                    p.temperatures.fill_(0.7 if stochastic else 0.0)
                    try:
                        self._replay(count, False, stochastic=stochastic)
                    except TensorParallelSamplingError:
                        pass
                    else:
                        raise TensorParallelSamplingError(
                            "masked sampling accepted invalid operands"
                        )
                receipts.append((count, f"grammar_{kind}", "reject"))
        return receipts

    def close(self) -> None:
        if not self._closed:
            torch.cuda.synchronize(self.device)
            for graph in self.graphs:
                graph.reset()
            for extra_graph in (self.grammar_candidate_graph, self.grammar_random_graph):
                if extra_graph is not None:
                    extra_graph.reset()
            self.graphs.clear()
            self.qualified = False
            self._closed = True
