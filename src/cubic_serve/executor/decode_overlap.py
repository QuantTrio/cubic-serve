"""Capture-time interleaving of independent rows at explicit collective boundaries.

The producer owns tensor layouts, arithmetic, and row/state independence. This
module owns only stream dependencies; it neither partitions requests nor changes
collective order. Streams/events live as long as every graph captured with them.

Sources: vLLM 89af4f1 v1/worker/ubatching.py; SGLang ba433bb
batch_overlap/operations.py. Both preserve per-lane dependencies at yields.
llama.cpp 458681e ggml-cuda/allreduce.cu separates copy and compute with events,
but its two-device transport is not a general multi-body scheduler.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from statistics import median

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DecodeOverlapPlan:
    """Cold-selected launch geometry retained when final cache addresses change."""

    parts: tuple[int, ...] = ()
    estimated_linear: bool = False
    collective_threads: int | None = None

    def __post_init__(self) -> None:
        if self.parts and (
            len(self.parts) < 2 or any(type(m) is not int or m <= 0 for m in self.parts)
        ):
            raise ValueError("overlap plan requires positive independent row partitions")
        if self.collective_threads is not None and (
            type(self.collective_threads) is not int
            or self.collective_threads not in (64, 128, 256)
        ):
            raise ValueError("invalid overlap collective thread geometry")
        if not self.parts and (self.estimated_linear or self.collective_threads is not None):
            raise ValueError("empty overlap plan cannot carry launch overrides")


def common_overlap_thread_candidates(
    candidates: tuple[int | None, ...], device: torch.device,
) -> tuple[int | None, ...]:
    """All ranks visit the same candidates, including heterogeneous overrides."""
    legal = (64, 128, 256)
    admitted = torch.tensor([int(t in candidates) for t in legal], dtype=torch.int64, device=device)
    dist.all_reduce(admitted, op=dist.ReduceOp.MIN)
    return (None, *(t for t, yes in zip(legal, admitted.tolist(), strict=True) if yes))


def decode_calibration_prepare(
    *, rows: int, vocabulary_size: int, device: torch.device,
    reset: Callable[[], None], bind_tokens: Callable[[torch.Tensor], None],
) -> Callable[[], None]:
    """Reset private state and bind identical real inputs for every candidate.

    Capture scratch deliberately clears embedding ownership. Timing that scratch
    measures zero hidden rows and artificially concentrated expert routing.
    A private, deterministic token corpus exercises the loaded model's routing
    without modifying its router or the caller's random generator. Split graphs
    copy these same parent rows, including embedding ownership, into their lanes.
    """
    tokens = torch.randint(
        vocabulary_size, (rows,), dtype=torch.int64, device="cpu",
        generator=torch.Generator().manual_seed(0),
    ).to(device)

    def prepare() -> None:
        reset()
        bind_tokens(tokens)

    return prepare


def format_decode_measurements(measurements: Sequence[tuple[tuple[int, ...], float]]) -> str:
    """Format milliseconds for logging without rounding the planner's inputs."""
    return "[" + ", ".join(f"({parts}, {elapsed:.3f})" for parts, elapsed in measurements) + "]"


def measure_decode_graph(
    graph: torch.cuda.CUDAGraph,
    prepare: Callable[[], None],
    stream: torch.cuda.Stream,
    samples: int,
) -> float:
    """Startup-only full-round timing, closed against the slowest topology rank."""
    timings = []
    for _ in range(samples):
        prepare()
        stream.wait_stream(torch.cuda.current_stream(stream.device))
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            begin.record()
            graph.replay()
            end.record()
        end.synchronize()
        timings.append(begin.elapsed_time(end))
    value = torch.tensor([median(timings)], dtype=torch.float64, device=stream.device)
    dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return float(value.item())


def compare_decode_graphs(
    control: torch.cuda.CUDAGraph,
    candidate: torch.cuda.CUDAGraph,
    prepare: Callable[[], None],
    stream: torch.cuda.Stream,
    samples: int,
) -> tuple[float, float]:
    """Bracket a newly built candidate with current controls, in ABBA order.

    Compilation and warming can change clocks after the original baseline.
    Require improvement over the faster observed control using the slower
    candidate observation. This is a conservative observed ratio, not a
    statistical confidence interval or a device-specific clock correction.
    Every measurement restores the identical operand/state fixture.
    """
    before, first, second, after = (
        measure_decode_graph(graph, prepare, stream, samples)
        for graph in (control, candidate, candidate, control)
    )
    return min(before, after), max(first, second)


class DecodeInterleave:
    """One compute stream and one ordered collective stream, shared by all lanes."""

    def __init__(self, lanes: int, device: torch.device) -> None:
        if lanes < 2:
            raise ValueError("decode interleaving requires at least two independent lanes")
        self.compute = torch.cuda.Stream(device=device)
        self.communication = torch.cuda.Stream(device=device)
        self.ready = tuple(torch.cuda.Event() for _ in range(lanes))
        self.done = tuple(torch.cuda.Event() for _ in range(lanes))

    def launch(self, iterators: Sequence[Iterator[Callable[[], None]]]) -> None:
        """Run compute until each yield, then enqueue its deferred collective.

        This runs during eager qualification or startup capture, never to create
        an execution plan during performance replay. All ranks must supply the
        same ordered lane schedule, including ownerless scratch rows.
        """
        if len(iterators) != len(self.ready):
            raise ValueError("decode iterator count differs from the captured lane plan")
        main = torch.cuda.current_stream(self.compute.device)
        self.compute.wait_stream(main)
        self.communication.wait_stream(main)
        active, pending = [True] * len(iterators), [False] * len(iterators)
        while any(active):
            for index, iterator in enumerate(iterators):
                if not active[index]:
                    continue
                with torch.cuda.stream(self.compute):
                    if pending[index]:
                        self.compute.wait_event(self.done[index])
                    try:
                        operation = next(iterator)
                    except StopIteration:
                        active[index] = False
                        continue
                    self.ready[index].record()
                with torch.cuda.stream(self.communication):
                    self.communication.wait_event(self.ready[index])
                    operation()
                    self.done[index].record()
                pending[index] = True
        main.wait_stream(self.compute)
        main.wait_stream(self.communication)
