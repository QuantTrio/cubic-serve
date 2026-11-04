"""Fixed-shape BF16 linear CUDA Graph candidate; never an eager fallback.

Sources: llama.cpp c841aeeb (ggml-cuda.cu), vLLM 73fac1e1
(layers/batch_invariant.py), SGLang ba433bb4 (quantization/unquant.py).
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from threading import Lock

import torch
from torch import Tensor

from cubic_serve.contracts.execution import (
    ActiveLaunchRow,
    LaunchPlan,
    RankReplayResources,
)
from cubic_serve.guardrail.graph_padding import GraphPaddingBuffers
from cubic_serve.ops.cuda.linear.kernel import BF16_LINEAR_OPERATOR_ID, BF16LinearPrimitive

REPLAY_ABI_ID = "cubic.replay.bf16-linear.v1"
IMPLEMENTATION_ID = BF16_LINEAR_OPERATOR_ID


class LinearGraphRejected(RuntimeError):
    """The typed launch or payload does not match this captured candidate."""


class BF16LinearGraphCandidate:
    """One captured-M graph with fixed K/N and lease-bound static storage."""

    def __init__(
        self,
        *,
        weight: Tensor,
        captured_m: int,
        resource: RankReplayResources,
    ) -> None:
        if (
            not isinstance(weight, Tensor)
            or weight.device.type != "cuda"
            or weight.dtype is not torch.bfloat16
            or weight.ndim != 2
            or not weight.is_contiguous()
            or min(weight.shape) <= 0
        ):
            raise LinearGraphRejected("weight must be non-empty contiguous CUDA BF16 [N,K]")
        if type(captured_m) is not int or captured_m <= 0:
            raise LinearGraphRejected("captured_m must be a positive integer")
        if resource.rank != 0:
            raise LinearGraphRejected("the first qualification slice supports one rank")

        self.resource = resource
        self.captured_m = captured_m
        self.output_width, self.input_width = weight.shape
        self.capacity_items = (
            ("input_width", self.input_width),
            ("output_width", self.output_width),
            ("request_slots", captured_m),
        )
        self._device = weight.device
        self._weight = weight.detach().clone()
        self._inputs = torch.empty(
            (captured_m, self.input_width), dtype=torch.bfloat16, device=self._device
        )
        self._outputs = torch.empty(
            (captured_m, self.output_width), dtype=torch.bfloat16, device=self._device
        )
        self._primitive = BF16LinearPrimitive(self._inputs, self._weight, self._outputs)
        self._stream = torch.cuda.Stream(device=self._device)  # type: ignore[no-untyped-call]
        self._graph: torch.cuda.CUDAGraph | None = None
        self._poison: str | None = None
        self._lock = Lock()

    @property
    def launch_signature(self) -> str:
        return self._primitive.launch_signature

    def capture(self) -> None:
        """Compile once, then capture only the fixed-address kernel launch."""

        with self._lock:
            if self._poison is not None:
                raise LinearGraphRejected(self._poison)
            if self._graph is not None:
                raise LinearGraphRejected("linear graph capture is single-use")
            try:
                current = torch.cuda.current_stream(self._device)
                self._stream.wait_stream(current)
                with torch.cuda.device(self._device), torch.cuda.stream(self._stream):
                    self._primitive.launch(self._inputs, self._weight, self._outputs)
                    self._primitive.launch(self._inputs, self._weight, self._outputs)
                self._stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=self._stream):
                    self._primitive.launch(self._inputs, self._weight, self._outputs)
                self._stream.synchronize()
                self._graph = graph
            except Exception as error:
                self._poison = f"linear graph capture failed: {type(error).__name__}: {error}"
                raise LinearGraphRejected(self._poison) from error

    def replay(
        self,
        launch: LaunchPlan,
        inputs: Mapping[str, Tensor],
        state: MutableMapping[str, Tensor],
        buffers: GraphPaddingBuffers,
    ) -> Mapping[str, Tensor]:
        """Replay under the launch lease and return only logical rows."""

        return self._execute(launch, inputs, state, buffers, captured=True)

    def qualification_eager(
        self,
        launch: LaunchPlan,
        inputs: Mapping[str, Tensor],
        state: MutableMapping[str, Tensor],
        buffers: GraphPaddingBuffers,
    ) -> Mapping[str, Tensor]:
        """Run the same fixed-shape kernel outside the graph for qualification only."""

        return self._execute(launch, inputs, state, buffers, captured=False)

    def _execute(
        self,
        launch: LaunchPlan,
        inputs: Mapping[str, Tensor],
        state: MutableMapping[str, Tensor],
        buffers: GraphPaddingBuffers,
        *,
        captured: bool,
    ) -> Mapping[str, Tensor]:
        del state  # stateless operator; the harness proves caller state remains unchanged
        with self._lock:
            self._validate(launch, inputs, buffers)
            assert self._graph is not None
            try:
                with torch.cuda.device(self._device), torch.cuda.stream(self._stream):
                    for physical, row in enumerate(launch.rows):
                        source = (
                            inputs["x"][row.logical_index]
                            if isinstance(row, ActiveLaunchRow)
                            else buffers.padding_inputs["x"][row.scratch_index]
                        )
                        self._inputs[physical].copy_(source)
                    if captured:
                        self._graph.replay()
                    else:
                        self._primitive.launch(self._inputs, self._weight, self._outputs)

                    logical_to_physical = [0] * launch.logical_m
                    padding_to_physical = [0] * (launch.captured_m - launch.logical_m)
                    for physical, row in enumerate(launch.rows):
                        if isinstance(row, ActiveLaunchRow):
                            logical_to_physical[row.logical_index] = physical
                        else:
                            padding_to_physical[row.scratch_index] = physical
                    logical = self._outputs[logical_to_physical].clone()
                    if padding_to_physical:
                        buffers.scratch_state["y"].copy_(self._outputs[padding_to_physical])
                self._stream.synchronize()
                return {"y": logical}
            except Exception as error:
                self._poison = f"linear graph replay failed: {type(error).__name__}: {error}"
                raise LinearGraphRejected(self._poison) from error

    def _validate(
        self,
        launch: LaunchPlan,
        inputs: Mapping[str, Tensor],
        buffers: GraphPaddingBuffers,
    ) -> None:
        if self._poison is not None:
            raise LinearGraphRejected(self._poison)
        if self._graph is None:
            raise LinearGraphRejected("linear graph is not captured")
        if (
            launch.bucket.replay_abi_id != REPLAY_ABI_ID
            or launch.bucket.capacity_items != self.capacity_items
            or launch.captured_m != self.captured_m
            or launch.replay_lease.rank_resources != (self.resource,)
        ):
            raise LinearGraphRejected("launch does not bind this captured linear ABI")
        if set(inputs) != {"x"}:
            raise LinearGraphRejected("linear input ABI requires exactly x")
        x = inputs["x"]
        if (
            x.shape != (launch.logical_m, self.input_width)
            or x.dtype is not torch.bfloat16
            or x.device != self._device
            or not x.is_contiguous()
        ):
            raise LinearGraphRejected("logical x does not match the captured CUDA BF16 ABI")
        padding_m = launch.captured_m - launch.logical_m
        if set(buffers.padding_inputs) != ({"x"} if padding_m else set()) or set(
            buffers.scratch_state
        ) != ({"y"} if padding_m else set()):
            raise LinearGraphRejected("padding buffers do not match the linear scratch ABI")
        if padding_m and (
            buffers.padding_inputs["x"].shape != (padding_m, self.input_width)
            or buffers.padding_inputs["x"].dtype is not torch.bfloat16
            or buffers.padding_inputs["x"].device != self._device
            or not buffers.padding_inputs["x"].is_contiguous()
            or buffers.scratch_state["y"].shape != (padding_m, self.output_width)
            or buffers.scratch_state["y"].dtype is not torch.bfloat16
            or buffers.scratch_state["y"].device != self._device
            or not buffers.scratch_state["y"].is_contiguous()
        ):
            raise LinearGraphRejected("padding tensors do not match the captured linear ABI")
