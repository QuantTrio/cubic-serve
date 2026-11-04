"""Packed CUDA operators with explicit input dtype and FP32 accumulation.

No full dequantized matrix is allocated. Weight-only operands round to the
activation dtype inside the GEMM tile; FP8 and NVFP4 preserve their own scale
boundaries. Kernel selection is by encoding, never model, device or concurrency.
"""

from __future__ import annotations

from contextlib import contextmanager

from cubic_serve.ops.cuda.startup import weight_bound_calibration_key

from typing import TYPE_CHECKING, cast

import torch
import triton  # type: ignore[import-untyped]
from torch import Tensor

from cubic_serve.models.quantization import ActivationFormat
from cubic_serve.models.quantized_checkpoint import QuantizedMatrix
from cubic_serve.ops.cuda.compiled_artifact import (
    CompiledArtifactIdentity,
    LaunchReceipt,
    bind_tensor,
)
from cubic_serve.ops.cuda.launch_cost import (
    LinearLaunchCost,
    LinearTile,
    MatmulGeometry,
    device_resources,
    linear_cost_scope,
    linear_geometries,
    select_launch,
)
from cubic_serve.ops.strategies import (
    LinearParameters,
    OperatorConfig,
    PackedLinearParameters,
    PackedMatmulParameters,
    launch_parameters,
    linear_row_tile,
    operator_strategy_scope,
)
from cubic_serve.quantization.common.binding import PackedOperatorBinding as PackedOperatorBinding
from cubic_serve.quantization.common.binding import _PackedLaunch as _PackedLaunch
from cubic_serve.quantization.common.kernels import _merge_split_k, _plain_mm, _plain_partition_mm
from cubic_serve.quantization.common.policy import placement_candidates
from cubic_serve.quantization.extensions.fp8.kernels import _fp8_quant as _fp8_quant
from cubic_serve.quantization.extensions.nvfp4.kernels import _nvfp4_quant as _nvfp4_quant
from cubic_serve.quantization.registry import get_extension, require_compute_mode

if TYPE_CHECKING:
    from cubic_serve.ops.cuda.linear.grouped import BF16GroupedLinearPrimitive


class _GroupedPlainBinding(PackedOperatorBinding):
    """Adapt the shared native grouped primitive to sealed occurrence dispatch."""

    def __init__(
        self,
        primitive: BF16GroupedLinearPrimitive,
        x: Tensor,
        weights: tuple[Tensor, ...],
        outputs: tuple[Tensor, ...],
        active_rows: Tensor,
    ) -> None:
        super().__init__(outputs[0], (), row_tile_count=primitive.row_tile_count)
        self._primitive = primitive
        self._operands = x, weights, outputs, active_rows
        self._flat = (x, *weights, *outputs, active_rows)
        self._bindings = tuple(bind_tensor(t) for t in self._flat)

    def validate_operands(self) -> None:
        if tuple(bind_tensor(t) for t in self._flat) != self._bindings:
            raise ValueError("grouped projection operands changed after binding")

    def launch(self) -> tuple[LaunchReceipt, ...]:
        self.validate_operands()
        return (self._primitive.launch(*self._operands),)

    def project_artifacts(self) -> tuple[CompiledArtifactIdentity, ...]:
        return (self._primitive.project_artifact(),)


class PackedProjectionDispatch:
    """Explicit startup occurrence bindings; launch cannot compile or allocate.

    A body registers each concrete input/weight/output triple, then seals the
    registry before any forward. Graph qualification and leases remain external.
    """

    def __init__(self, operators: PackedLinearOperators, *, operation: str = "linear") -> None:
        if operation not in ("linear", "embedding"):
            raise ValueError("packed projection dispatch requires linear or embedding")
        self.operators, self.operation = operators, operation
        self._sealed = False
        self._groups: list[
            tuple[Tensor, tuple[Tensor | QuantizedMatrix, ...], tuple[Tensor, ...]]
        ] = []
        self._pending: dict[
            tuple[tuple[object, ...], int, tuple[object, ...]],
            tuple[Tensor, Tensor | QuantizedMatrix, Tensor],
        ] = {}
        self._scratch_pools: dict[tuple[torch.device, torch.dtype, int], Tensor] = {}
        self._entries: dict[
            tuple[tuple[object, ...], int, tuple[object, ...]],
            tuple[Tensor | QuantizedMatrix, PackedOperatorBinding],
        ] = {}

    @staticmethod
    def _view_key(tensor: Tensor) -> tuple[object, ...]:
        return (
            tensor.data_ptr(),
            tensor.storage_offset(),
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.dtype,
            tensor.device,
        )

    def prepare(self, x: Tensor, weight: Tensor | QuantizedMatrix, output: Tensor) -> None:
        if self._sealed:
            raise ValueError("packed projection occurrences are sealed")
        key = self._view_key(x), id(weight), self._view_key(output)
        self._pending.setdefault(key, (x, weight, output))

    def prepare_group(
        self, x: Tensor, weights: tuple[Tensor | QuantizedMatrix, ...], outputs: tuple[Tensor, ...]
    ) -> None:
        """Declare ordered independent projections sharing an unchanged input.

        The first member produces all grouped outputs. Later member occurrences
        are sealed no-ops; the model must not mutate the input between members.
        Unsupported formats or incompatible explicit plans remain independent.
        """
        if self.operation != "linear" or len(weights) < 2 or len(weights) != len(outputs):
            raise ValueError("projection group requires matching linear members")
        for weight, output in zip(weights, outputs, strict=True):
            self.prepare(x, weight, output)
        self._groups.append((x, weights, outputs))

    def seal(self) -> None:
        if self._sealed:
            return
        if not self._pending:
            raise ValueError("packed projection dispatch has no occurrences")
        serial_bindings = []

        def share_scratch(binding: PackedOperatorBinding) -> None:
            # Rebind at construction, before the next group is materialized.
            # Delaying this until all groups exist makes their private peaks add.
            for index in binding.reusable_scratch_indices:
                scratch = binding.scratch[index]
                # Slots within one binding can be simultaneously live, even
                # with the same dtype. Only serial occurrences share a slot.
                pool_key = scratch.device, scratch.dtype, index
                pool = self._scratch_pools.get(pool_key)
                if pool is None or pool.numel() < scratch.numel():
                    pool = torch.empty(scratch.numel(), dtype=scratch.dtype, device=scratch.device)
                    self._scratch_pools[pool_key] = pool
                    for previous in serial_bindings:
                        if index in previous.reusable_scratch_indices:
                            old = previous.scratch[index]
                            if (old.device, old.dtype, index) == pool_key:
                                previous.rebind_scratch(index, pool[: old.numel()].view(old.shape))
                binding.rebind_scratch(index, pool[: scratch.numel()].view(scratch.shape))
            serial_bindings.append(binding)

        grouped = {}
        declared = set()
        for x, weights, outputs in self._groups:
            for weight, output in zip(weights, outputs, strict=True):
                key = self._view_key(x), id(weight), self._view_key(output)
                if key in declared:
                    raise ValueError("projection occurrence belongs to multiple groups")
                declared.add(key)
            for indices in self.operators._projection_group_indices(weights):
                if len(indices) < 2:
                    continue
                members = tuple(weights[i] for i in indices)
                targets = tuple(outputs[i] for i in indices)
                binding = self.operators.bind_linear_group(x, members, targets)
                if binding is None:
                    continue
                share_scratch(binding)
                for ordinal, index in enumerate(indices):
                    weight, output = weights[index], outputs[index]
                    key = self._view_key(x), id(weight), self._view_key(output)
                    grouped[key] = binding if ordinal == 0 else PackedOperatorBinding(
                        output, (), row_tile_count=binding.row_tile_count
                    )
        for key, (x, weight, output) in self._pending.items():
            binding = grouped.get(key)
            if binding is None:
                binding = (
                    self.operators.bind_linear(x, weight, output)
                    if self.operation == "linear"
                    else self.operators.bind_embedding(x, weight, output)
                )
                share_scratch(binding)
            self._entries[key] = weight, binding
        self._groups.clear()
        self._pending.clear()
        self._sealed = True

    @property
    def row_tile_count(self) -> int | None:
        """Conservative lower bound over all sealed projection occurrences."""
        if not self._sealed or self.operation != "linear":
            return None
        counts = [binding.row_tile_count for _, binding in self._entries.values()]
        if not counts or any(count is None for count in counts):
            return None
        return min(count for count in counts if count is not None)

    @property
    def scratch(self) -> tuple[Tensor, ...]:
        return (
            *self._scratch_pools.values(),
            *(
                t
                for _, binding in self._entries.values()
                for index, t in enumerate(binding.scratch)
                if index not in binding.reusable_scratch_indices
            ),
        )

    def launch(
        self,
        x: Tensor,
        weight: Tensor | QuantizedMatrix,
        output: Tensor,
    ) -> tuple[LaunchReceipt, ...]:
        if not self._sealed:
            raise ValueError("packed projection occurrences must be sealed before forward")
        entry = self._entries.get((self._view_key(x), id(weight), self._view_key(output)))
        if entry is None:
            raise ValueError("packed projection occurrence was not bound at startup")
        return entry[1].launch()

    def project_artifacts(self) -> tuple[CompiledArtifactIdentity, ...]:
        return tuple(
            a for _, binding in self._entries.values() for a in binding.project_artifacts()
        )


class PackedLinearOperators:
    """Native eager binding. Unsupported formats fail; no reference fallback."""

    def __init__(
        self,
        config: OperatorConfig | None = None,
        *,
        device: str = "",
        model: str = "",
        prepared_weights: dict[
            tuple[object, ...], tuple[object, Tensor]
        ] | None = None,
        materialized_weights: dict[tuple[object, ...], tuple[QuantizedMatrix, QuantizedMatrix]]
        | None = None,
    ) -> None:
        self.config = config or OperatorConfig()
        self.device_identity, self.model_identity = device, model
        self._partitions: dict[int, Tensor] = {}
        self._plain_costs: dict[torch.device, LinearLaunchCost] = {}
        self._calibrate_dynamic_shapes = True
        self.dense_capacity_partitions = False
        self._tiles: dict[tuple[str, int, int], LinearParameters | PackedLinearParameters] = {}
        self._packed_choices: dict[tuple[object, ...], PackedLinearParameters] = {}
        self._routing_choices: dict[tuple[object, ...], int] = {}
        self._routed_choices: dict[tuple[object, ...], PackedMatmulParameters] = {}
        # Format-owned immutable representations share the resident model's
        # lifetime. Standalone operators own their own cache; there is no
        # process-global cache or request-time preparation.
        self.prepared_weights = {} if prepared_weights is None else prepared_weights
        self.materialized_weights = {} if materialized_weights is None else materialized_weights
        self.int8_inputs = {}

    @contextmanager
    def dense_capacity_scope(self):
        """Bind a validated block with capacity-aware dense reduction plans.

        This is an owner-local, startup-only planning hint. Formats retain their
        own arithmetic and admission; captured calls never consult the scope.
        """
        previous = self.dense_capacity_partitions
        self.dense_capacity_partitions = True
        try:
            yield
        finally:
            self.dense_capacity_partitions = previous

    def register_int8_input(self, activation) -> None:
        from cubic_serve.quantization.common.int8 import Int8Activation

        if (self.config.compute_mode != "wna8"
                or not isinstance(activation, Int8Activation)):
            raise ValueError("INT8 input registration requires WNA8 and explicit scale ownership")
        key = PackedProjectionDispatch._view_key(activation.codes)
        if key in self.int8_inputs or torch.cuda.is_current_stream_capturing():
            raise ValueError("INT8 input registration must be unique and precede capture")
        self.int8_inputs[key] = activation

    def int8_input(self, x):
        return self.int8_inputs.get(PackedProjectionDispatch._view_key(x))

    def projection_input(self, x, weight):
        encoded = self.int8_input(x) if x.dtype == torch.int8 else None
        if encoded is not None and isinstance(weight, Tensor):
            if encoded.floating is None:
                raise ValueError("plain projection requires a declared floating producer exit")
            return encoded.floating
        return x

    def materialize_weight(self, weight: QuantizedMatrix, device: torch.device) -> QuantizedMatrix:
        """Own one original-layout GPU copy only when the selected backend needs it."""
        source_device = next(iter(weight.payload.values())).device
        if source_device == device:
            return weight
        if source_device.type != "cpu":
            raise ValueError("packed weights cannot migrate across rank devices")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("packed weight materialization must precede graph capture")
        versions: list[int | None] = []
        for tensor in weight.payload.values():
            try:
                versions.append(tensor._version)
            except RuntimeError:
                # Inference tensors have no counter; checkpoint ownership
                # makes them immutable for the lifetime of this binding.
                versions.append(None)
        key = (
            device,
            weight.encoding,
            weight.shape,
            weight.partitions,
            tuple(bind_tensor(t) for t in weight.payload.values()),
            tuple(versions),
        )
        cached = self.materialized_weights.get(key)
        if cached is None:
            matrix = weight.to(device)
            self.materialized_weights[key] = weight, matrix
            return matrix
        return cached[1]

    def _validate_linear_operands(
        self, x: Tensor, weight: Tensor | QuantizedMatrix, output: Tensor,
        fp8_token_scales: Tensor | None = None,
    ) -> None:
        encoded = self.int8_input(x) if x.dtype == torch.int8 else None
        if (x.dtype not in (torch.float16, torch.bfloat16) and encoded is None) or not x.is_cuda:
            raise ValueError("native linear requires CUDA FP16/BF16 activations")
        if encoded is not None and (
            not isinstance(weight, QuantizedMatrix)
            or self.config.compute_mode != "wna8"
        ):
            raise ValueError("registered INT8 inputs require WNA8 packed weights")
        if not isinstance(weight, Tensor | QuantizedMatrix):
            raise TypeError("native linear requires a tensor or packed matrix")
        if isinstance(weight, QuantizedMatrix) and self.config.compute_mode != "wna16":
            require_compute_mode(weight.encoding, "linear", self.config.compute_mode)
        weight_device = (
            weight.device
            if isinstance(weight, Tensor)
            else next(iter(weight.payload.values())).device
        )
        if weight_device != x.device and not (
            isinstance(weight, QuantizedMatrix) and weight_device.type == "cpu"
        ):
            raise ValueError("linear weights and activations must share one device")
        if x.ndim != 2 or not x.is_contiguous() or min(x.shape) <= 0:
            raise ValueError("bound linear requires contiguous positive [M,K] activations")
        if len(weight.shape) != 2 or weight.shape[1] != x.shape[1]:
            raise ValueError("bound linear weight K differs")
        n, k = weight.shape
        if (
            output.shape != (x.shape[0], n)
            or output.dtype != (x.dtype if encoded is None else encoded.boundary_dtype)
            or output.device != x.device
            or not output.is_contiguous()
        ):
            raise ValueError("bound linear output has an invalid ABI")
        weight_tensors = (weight,) if isinstance(weight, Tensor) else tuple(weight.payload.values())
        if encoded is not None:
            weight_tensors += (encoded.scales,)
        if fp8_token_scales is not None and (
            not isinstance(weight, QuantizedMatrix)
            or weight.encoding.activation is not ActivationFormat.FP8_TOKEN
            or fp8_token_scales.shape != (x.shape[0],)
            or fp8_token_scales.dtype is not torch.float32
            or fp8_token_scales.device != x.device
            or not fp8_token_scales.is_contiguous()
            or any(
                fp8_token_scales.untyped_storage().data_ptr() == t.untyped_storage().data_ptr()
                for t in (x, output, *weight_tensors)
            )
        ):
            raise ValueError("external FP8 token scales require private contiguous FP32 [M]")
        if any(
            output.untyped_storage().data_ptr() == t.untyped_storage().data_ptr()
            for t in (x, *weight_tensors)
        ):
            raise ValueError("bound linear output must not alias its operands")

    def _projection_group_indices(
        self, weights: tuple[Tensor | QuantizedMatrix, ...]
    ) -> tuple[tuple[int, ...], ...]:
        """Independent same-input members keep their format-owned arithmetic."""
        partitions: dict[object, list[int]] = {}
        for index, weight in enumerate(weights):
            family = (
                None if isinstance(weight, Tensor)
                else (weight.encoding, self._parameters(weight))
            )
            partitions.setdefault(family, []).append(index)
        return tuple(tuple(indices) for indices in partitions.values())

    def bind_linear_groups(
        self, x: Tensor, weights: tuple[Tensor | QuantizedMatrix, ...], outputs: tuple[Tensor, ...]
    ) -> tuple[PackedOperatorBinding, ...]:
        # Validate the whole declaration, including cross-family output aliases,
        # before partitioning it through the same policy as sealed dispatch.
        whole = self.bind_linear_group(x, weights, outputs)
        if whole is not None:
            return (whole,)
        bindings = []
        for indices in self._projection_group_indices(weights):
            members = tuple(weights[i] for i in indices)
            targets = tuple(outputs[i] for i in indices)
            grouped = (
                self.bind_linear_group(x, members, targets)
                if 1 < len(indices) < len(weights) else None
            )
            if grouped is not None:
                bindings.append(grouped)
            else:
                bindings.extend(
                    self.bind_linear(x, weight, output)
                    for weight, output in zip(members, targets, strict=True)
                )
        return tuple(bindings)

    def bind_linear_group(
        self, x: Tensor, weights: tuple[Tensor | QuantizedMatrix, ...], outputs: tuple[Tensor, ...]
    ) -> PackedOperatorBinding | None:
        from cubic_serve.ops.cuda.linear.grouped import (
            bind_grouped_linear,
            bind_partitioned_linear_group,
        )

        if len(weights) < 2 or len(weights) != len(outputs):
            raise ValueError("projection group requires matching members")
        if all(isinstance(w, Tensor) for w in weights):
            x = self.projection_input(x, weights[0])
        for weight, output in zip(weights, outputs, strict=True):
            self._validate_linear_operands(x, weight, output)
        operands = tuple(t for w in weights for t in (
            (w,) if isinstance(w, Tensor) else tuple(w.payload.values())
        ))
        for index, output in enumerate(outputs):
            if any(output.untyped_storage().data_ptr() == t.untyped_storage().data_ptr()
                   for t in (*outputs[:index], *operands)):
                raise ValueError("grouped outputs must not alias each other or weights")
        if not all(isinstance(w, Tensor) for w in weights):
            if not all(isinstance(w, QuantizedMatrix) for w in weights):
                return None
            packed = cast(tuple[QuantizedMatrix, ...], weights)
            if any(w.encoding != packed[0].encoding for w in packed):
                return None
            params = tuple(self._parameters(w) for w in packed)
            if any(p != params[0] for p in params):
                return None
            linear = get_extension(packed[0].encoding.format).module("linear")
            from cubic_serve.quantization.common.small_rows_cost import choose

            candidate = choose(self, x, packed, outputs, params[0], linear)
            if candidate is not None:
                return cast(PackedOperatorBinding, candidate)
            factory = getattr(linear, "bind_grouped", None)
            if factory is None:
                return None
            binding = factory(self, x, packed, outputs, params[0])
            if binding is not None and not isinstance(binding, PackedOperatorBinding):
                raise TypeError("native group hook must return a packed binding")
            return binding
        plain_weights = cast(tuple[Tensor, ...], weights)
        # This is immutable physical capacity metadata, not recurrent scratch.
        active = torch.full((1,), x.shape[0], dtype=torch.int32, device=x.device)
        cost = self._plain_costs.get(x.device)
        if cost is None:
            cost = LinearLaunchCost(device_resources(x.device), 0.0, ())
            self._plain_costs[x.device] = cost
        with (
            operator_strategy_scope(
                self.config, device=self.device_identity, model=self.model_identity
            ),
            linear_cost_scope(cost),
        ):
            partitioned = bind_partitioned_linear_group(x, plain_weights, outputs, active)
            if partitioned is not None:
                return partitioned
            primitive = bind_grouped_linear(x, plain_weights, outputs, active)
        if primitive is None:
            return None
        return _GroupedPlainBinding(primitive, x, plain_weights, outputs, active)

    def _parameters(
        self, weight: Tensor | QuantizedMatrix
    ) -> LinearParameters | PackedLinearParameters:
        family = (
            get_extension(weight.encoding.format).family("linear")
            if isinstance(weight, QuantizedMatrix)
            else "linear"
        )
        n, k = weight.shape
        key = (family, n, k)
        if key not in self._tiles:
            params = self.config.resolve(
                family, device=self.device_identity, model=self.model_identity, n=n, k=k
            )
            if not isinstance(params, LinearParameters | PackedLinearParameters):
                raise TypeError("linear binding received a non-linear strategy")
            self._tiles[key] = params
        return self._tiles[key]

    def _arguments(self, matrix: QuantizedMatrix) -> tuple[tuple[Tensor, ...], int]:
        return get_extension(matrix.encoding.format).module("linear").arguments(self, matrix)

    def linear(self, inputs: Tensor, weight: Tensor | QuantizedMatrix) -> Tensor:
        x = inputs.contiguous().reshape(-1, inputs.shape[-1])
        encoded = self.int8_input(x) if x.dtype == torch.int8 else None
        dtype = x.dtype if encoded is None else encoded.boundary_dtype
        output = torch.empty((x.shape[0], weight.shape[0]), dtype=dtype, device=x.device)
        binding = self.bind_linear(x, weight, output)
        binding.launch()
        return output.reshape(*inputs.shape[:-1], weight.shape[0])

    def seal_dynamic_calibration(self) -> None:
        """Keep request-time eager shapes from capturing new calibration graphs."""
        self._calibrate_dynamic_shapes = False

    def bind_fp8_token_scales(self, x: Tensor, scales: Tensor) -> PackedOperatorBinding:
        from cubic_serve.quantization.extensions.fp8.linear import bind_token_scales

        return bind_token_scales(x, scales)

    def bind_routed_fp8_input(
        self,
        x: Tensor,
        *,
        row_parallel: bool = False,
    ) -> tuple[Tensor, Tensor, PackedOperatorBinding]:
        from cubic_serve.quantization.extensions.fp8.linear import bind_routed_input

        return bind_routed_input(x, row_parallel=row_parallel)

    def bind_linear(
        self,
        x: Tensor,
        weight: Tensor | QuantizedMatrix,
        output: Tensor,
        *,
        fp8_token_scales: Tensor | None = None,
    ) -> PackedOperatorBinding:
        """Allocate scratch and compile once before model bucket capture.

        Row-parallel FP8 may bind caller-owned token scales computed over the
        original logical K domain. The caller produces them before each launch;
        local quantization reads them without recomputing or overwriting them.
        """
        x = self.projection_input(x, weight)
        self._validate_linear_operands(x, weight, output, fp8_token_scales)
        n, k = weight.shape
        params = self._parameters(weight)
        if isinstance(weight, QuantizedMatrix):
            extension = get_extension(weight.encoding.format)
            linear = extension.module("linear")
            if fp8_token_scales is None:
                from cubic_serve.quantization.common.small_rows_cost import choose

                candidate = choose(self, x, (weight,), (output,), params, linear)
                if candidate is not None:
                    return cast(PackedOperatorBinding, candidate)
            factory = getattr(linear, "bind_native", None)
            if factory is not None:
                candidate = factory(self, x, weight, output, params, fp8_token_scales)
                if candidate is not None:
                    if not isinstance(candidate, PackedOperatorBinding):
                        raise TypeError("native format hook must return a packed binding")
                    return candidate
            weight = self.materialize_weight(weight, x.device)
        bm, bn, bk = params.fixed_block_m, params.fixed_block_n, params.fixed_block_k
        m = x.shape[0]
        grid: tuple[int, ...] = (triton.cdiv(m, bm), triton.cdiv(n, bn))
        compile_launch = _PackedLaunch.compile
        calls: tuple[_PackedLaunch, ...]
        scratch: tuple[Tensor, ...] = ()
        if isinstance(weight, Tensor):
            assert isinstance(params, LinearParameters)
            bm = linear_row_tile(m, bm, params.fixed_max_block_m)
            grid = (triton.cdiv(m, bm), triton.cdiv(n, bn))
            if weight.dtype != x.dtype or not weight.is_contiguous():
                raise ValueError("plain linear requires contiguous weights in the input dtype")
            with operator_strategy_scope(
                self.config,
                device=self.device_identity,
                model=self.model_identity,
            ):
                cost = self._plain_costs.get(x.device)
                if cost is None:
                    cost = LinearLaunchCost(device_resources(x.device), 0.0, ())
                    self._plain_costs[x.device] = cost
                # Reuse the compact-Marlin fixed-K supply rule. A full output
                # tile can still leave most SMs idle; admission measures device
                # supply, not the fraction of valid columns inside one tile.
                # M never selects the reduction order or partition count.
                partitions = 1
                columns = triton.cdiv(n, params.fixed_block_n)
                if (k % bk == 0 and k >= params.grouped_vector_min_k
                        and columns < cost.device.sms):
                    from cubic_serve.quantization.common.marlin.cost import fixed_k_partitions

                    partitions = fixed_k_partitions(
                        n=columns * params.fixed_block_n, k=k, group=bk,
                        block_n=params.fixed_block_n, block_k=bk, sms=cost.device.sms,
                        supply_factor=params.fixed_cta_supply_factor,
                        max_partitions=params.fixed_max_k_partitions,
                    )
                target = (torch.empty((partitions, m, n), dtype=torch.float32, device=x.device)
                          if partitions > 1 else output)
                geometries = linear_geometries(
                    cost.device,
                    rows=m,
                    columns=(n,),
                    reduction=k,
                    element_bytes=x.element_size(),
                    communication=cost.communication,
                    baseline=MatmulGeometry(bm, bn, bk, params.fixed_warps, params.fixed_stages),
                    parameters=params,
                    partitions=partitions,
                )
                if not self._calibrate_dynamic_shapes:
                    geometries = geometries[:1]
                candidates = tuple(
                    compile_launch(
                        _plain_partition_mm if partitions > 1 else _plain_mm,
                        (x, weight, target),
                        (m, n, k, point.block_m, point.block_n, bk)
                        + ((partitions,) if partitions > 1 else ()),
                        (triton.cdiv(m, point.block_m), triton.cdiv(n, point.block_n), partitions),
                        dict(num_warps=point.warps, num_stages=point.stages)
                        | ({"enable_fp_fusion": False} if partitions > 1 else {}),
                    )
                    for point in geometries
                )
                tiles = tuple(
                    LinearTile(c.kernel, c.grid, point.block_m, point.block_n, point)
                    for c, point in zip(candidates, geometries, strict=True)
                )
                selected = (
                    cost.select(
                        tiles, x, (weight,), grouped=False, masked=False,
                        partitions=partitions,
                        replays=launch_parameters().micro_calibration_replays,
                    ) if self._calibrate_dynamic_shapes else tiles[0]
                )
                calls = (candidates[next(i for i, tile in enumerate(tiles) if tile is selected)],)
                grid = selected.grid
                if partitions > 1:
                    scratch = (target,)
                    calls += (compile_launch(
                        _merge_split_k, (target, output), (m * n, partitions, 256),
                        (triton.cdiv(m * n, 256),),
                        {"num_warps": 4, "enable_fp_fusion": False},
                    ),)

        else:
            assert isinstance(params, PackedLinearParameters)
            extension = get_extension(weight.encoding.format)
            if params.generic_geometry:
                with operator_strategy_scope(
                    self.config,
                    device=self.device_identity,
                    model=self.model_identity,
                ):
                    params = self._select_packed_parameters(
                        x,
                        weight,
                        params,
                        external_scales=fp8_token_scales is not None,
                    )
            return extension.module("linear").bind_linear(
                self, x, weight, output, params, fp8_token_scales
            )
        return PackedOperatorBinding(output, calls, scratch=scratch, row_tile_count=grid[0])

    def _select_packed_parameters(
        self,
        x: Tensor,
        weight: QuantizedMatrix,
        params: PackedLinearParameters,
        *,
        external_scales: bool,
    ) -> PackedLinearParameters:
        """Cold calibration on private operands, shared across equal local ABIs.

        A row-parallel FP8 caller still owns its scale collective. Calibration
        supplies private, already-completed scales and never enters that group.
        No checkpoint decode, input scale or K-reduction rule is selected here.
        """
        extension = get_extension(weight.encoding.format)
        key = (
            x.device,
            x.dtype,
            x.shape[0],
            weight.shape,
            weight.encoding,
            weight.partitions,
            params,
            external_scales,
        )
        if key in self._packed_choices:
            return self._packed_choices[key]
        n, k = weight.shape
        candidates = placement_candidates(
            params,
            family=extension.family("linear"),
            device=x.device,
            rows=x.shape[0],
            columns=n,
            reduction=k,
            element_bytes=x.element_size(),
        )
        if not self._calibrate_dynamic_shapes:
            self._packed_choices[key] = candidates[0]
            return candidates[0]
        if len(candidates) == 1:
            self._packed_choices[key] = params
            return params
        generator = torch.Generator(device=x.device).manual_seed(0)
        sample = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator).mul_(
            0.01
        )
        result = torch.empty((x.shape[0], n), dtype=x.dtype, device=x.device)
        scales = (
            torch.ones(x.shape[0], dtype=torch.float32, device=x.device)
            if external_scales
            else None
        )
        selected = None

        def bind(point: PackedLinearParameters) -> None:
            nonlocal selected
            selected = extension.module("linear").bind_linear(
                self,
                sample,
                weight,
                result,
                point,
                scales,
            )

        def launch() -> None:
            assert selected is not None
            selected.launch()

        chosen = select_launch(
            candidates, device=x.device, bind=bind, launch=launch, snapshot=lambda: (result,),
            cache_key=weight_bound_calibration_key(("packed-linear", key)),
        )
        self._packed_choices[key] = chosen
        return chosen

    def embedding(
        self, input_ids: Tensor, weight: Tensor | QuantizedMatrix, *, dtype: torch.dtype
    ) -> Tensor:
        output = torch.empty(
            (*input_ids.shape, weight.shape[1]), dtype=dtype, device=input_ids.device
        )
        binding = self.bind_embedding(input_ids.contiguous(), weight, output)
        binding.launch()
        return output

    def bind_embedding(
        self,
        input_ids: Tensor,
        weight: Tensor | QuantizedMatrix,
        output: Tensor,
    ) -> PackedOperatorBinding:
        """Bind a rank-local lookup; the caller validates IDs before graph fill."""
        if not isinstance(weight, Tensor | QuantizedMatrix) or len(weight.shape) != 2:
            raise ValueError("embedding requires a matrix weight")
        if isinstance(weight, QuantizedMatrix):
            factory = getattr(get_extension(weight.encoding.format).module("linear"),
                              "bind_native_embedding", None)
            if factory is not None:
                binding = factory(self, input_ids, weight, output)
                if binding is not None:
                    if not isinstance(binding, PackedOperatorBinding):
                        raise TypeError("native embedding hook must return a packed binding")
                    return binding
            weight = self.materialize_weight(weight, input_ids.device)
        tensors = (
            (input_ids, output, weight)
            if isinstance(weight, Tensor)
            else (input_ids, output, *weight.payload.values())
        )
        n, k = weight.shape
        if (
            not input_ids.is_cuda
            or input_ids.dtype not in (torch.int32, torch.int64)
            or input_ids.numel() <= 0
            or output.dtype not in (torch.float16, torch.bfloat16)
            or output.shape != (*input_ids.shape, k)
            or any(t.device != input_ids.device or not t.is_contiguous() for t in tensors)
        ):
            raise ValueError(
                "bound embedding requires contiguous same-device IDs, weight and FP16/BF16 output"
            )
        if any(
            output.untyped_storage().data_ptr() == t.untyped_storage().data_ptr()
            for t in (tensors[0], *tensors[2:])
        ):
            raise ValueError("bound embedding output must not alias its operands")
        if isinstance(weight, Tensor):
            from cubic_serve.ops.cuda.embedding.token import _BLOCK_H, _gather

            if weight.dtype != output.dtype:
                raise ValueError("plain embedding dtype differs from execution dtype")
            call = _PackedLaunch.compile(
                _gather,
                (input_ids, weight, output),
                (k, _BLOCK_H),
                (input_ids.numel(), triton.cdiv(k, _BLOCK_H)),
                {"num_warps": 4},
            )
        else:
            params = self._parameters(weight)
            assert isinstance(params, PackedLinearParameters)
            return (
                get_extension(weight.encoding.format)
                .module("linear")
                .bind_embedding(self, input_ids, weight, output, params)
            )
        return PackedOperatorBinding(output, (call,))

    def bind_scaled_sharded_embedding(
        self, ids: Tensor, shards: tuple[tuple[int, Tensor | QuantizedMatrix], ...],
        scale: Tensor, row_activity: Tensor, output: Tensor,
    ) -> PackedOperatorBinding:
        """Bind device vocabulary lookup through the same weight owner as linear.

        CPU/mapped input tables keep their existing independent lookup entry;
        this entry explicitly requests device storage, without a model branch.
        """
        from cubic_serve.ops.cuda.embedding.token import bind_scaled_sharded_embedding

        if not shards:
            raise ValueError("device embedding needs original shard owners")
        first = shards[0][1]
        if isinstance(first, QuantizedMatrix):
            factory = getattr(get_extension(first.encoding.format).module("linear"),
                              "bind_prepared_scaled_sharded_embedding", None)
            if factory is not None:
                binding = factory(self, ids, shards, scale, row_activity, output)
                if binding is not None:
                    if not isinstance(binding, PackedOperatorBinding):
                        raise TypeError("prepared embedding hook must return a packed binding")
                    return binding
            shards = tuple((start, self.materialize_weight(weight, ids.device)
                            if isinstance(weight, QuantizedMatrix) else weight)
                           for start, weight in shards)
        return bind_scaled_sharded_embedding(ids, shards, scale, row_activity, output)

    def full_attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        query_ordinals: Tensor,
        key_ordinals: Tensor,
    ) -> Tensor:
        # Quantized admission changes linear/embedding execution, not the
        # model's established attention arithmetic or state boundaries.
        from cubic_serve.ops.eager.attention import full_attention

        return full_attention(
            query,
            key,
            value,
            query_ordinals=query_ordinals,
            key_ordinals=key_ordinals,
            output_dtype=query.dtype,
        )
