"""Stable extension identity. Implementations remain owned by one package.

Checkpoint, layout, reference, linear, moe and policy modules are loaded lazily.
A linear module may expose ``bind_scaled_sharded_embedding`` for packed row
lookup over immutable original owners; activity and collective ownership follow
the common sharded embedding contract.
Optional ``bind_native_embedding`` and ``bind_prepared_scaled_sharded_embedding``
hooks may reuse the operator owner's prepared device storage before an original
device copy is materialized. Returning None retains the ordinary format path.
These device lookup hooks do not change the independent mapped-host table path.
They are implementation contracts, never runtime inference fallbacks. Registration
adds capabilities; graph admission and numerical qualification remain external.

Layouts may provide ``admit_encoding(weight, encoding, records)`` to close an
exporter's schema against physical headers and ``materialize``/``slice_matrix``
for bit-packed owner spans that cannot be represented by tensor slicing alone.
These hooks run at admission/preparation, preserve codes and original scale
boundaries, and must never infer or reassign logical expert ownership.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Any, Literal, Protocol, overload

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch import Tensor

    from cubic_serve.models.quantization import EncodingRule, QuantizationScheme
    from cubic_serve.models.quantized_checkpoint import LogicalWeight, QuantizedMatrix
    from cubic_serve.ops.cuda.linear.quantized import PackedLinearOperators
    from cubic_serve.quantization.common.binding import PackedOperatorBinding
    from cubic_serve.quantization.common.parameters import (
        PackedLinearParameters,
        PackedMatmulParameters,
    )
    from cubic_serve.quantization.common.reference import ActivationQuantization


class CheckpointContract(Protocol):
    def validate_scheme(self, scheme: QuantizationScheme) -> None: ...
    def parse_config(
        self, config: Mapping[str, object]
    ) -> tuple[tuple[EncodingRule, ...], tuple[str, ...]]: ...
    def parse_group(
        self, group: Mapping[str, object], weights: Mapping[str, object]
    ) -> QuantizationScheme: ...


class LayoutContract(Protocol):
    ROW_ALIGNMENT: int
    CONCAT_AXIS: int
    SCALAR_PAYLOADS: tuple[str, ...]

    def payload_headers(
        self, weight: LogicalWeight, encoding: QuantizationScheme
    ) -> tuple[tuple[str, str, tuple[int, ...]], ...]: ...
    def slice_index(
        self,
        shape: tuple[int, ...],
        partitions: tuple[int, ...],
        encoding: QuantizationScheme,
        key: str,
        axis: int,
        start: int,
        end: int,
    ) -> tuple[tuple[slice, ...], tuple[int, ...]]: ...


class ReferenceContract(Protocol):
    """Independent format math shared by every model and execution mode.

    A format advertising WNA8 also supplies ``wna8_operand(inputs)`` and
    ``wna8_weight(matrix)``. It may supply ``wna8_linear`` for an original
    grouped integer reduction. No default INT activation rule is inherited:
    e.g. token FP32 scales and GGML Q8_1 block FP16 scales differ.
    WNA16 continues to use the existing quantize/decode boundaries.

    An optional ``prepare_wna8_linear(inputs, matrix, output_rows=...,
    byte_budget=...)`` returns ``(decode_rows, apply_part)`` or None. Its
    call-scoped producer and streamed decode workspace must respect the budget;
    apply_part preserves the caller's canonical output_rows dot shapes and
    format-owned arithmetic. The executor batches only whole canonical tiles.
    """
    def decode(
        self,
        payload: Mapping[str, Tensor],
        encoding: QuantizationScheme,
        shape: tuple[int, int],
        partitions: tuple[int, ...] | None,
    ) -> Tensor: ...
    def quantize(
        self, inputs: Tensor, encoding: QuantizationScheme, global_inverse: Tensor | None
    ) -> ActivationQuantization: ...


class LinearContract(Protocol):
    def arguments(
        self, operators: PackedLinearOperators, matrix: QuantizedMatrix
    ) -> tuple[tuple[Tensor, ...], int]: ...
    def bind_linear(
        self,
        operators: PackedLinearOperators,
        x: Tensor,
        weight: QuantizedMatrix,
        output: Tensor,
        params: PackedLinearParameters,
        fp8_token_scales: Tensor | None = None,
    ) -> PackedOperatorBinding: ...
    def bind_embedding(
        self,
        operators: PackedLinearOperators,
        input_ids: Tensor,
        weight: QuantizedMatrix,
        output: Tensor,
        params: PackedLinearParameters,
    ) -> PackedOperatorBinding: ...


@dataclass(frozen=True, slots=True)
class RoutedRows:
    """Explicit rank-local route ownership; padding has no logical row ID."""

    counts: Tensor
    starts: Tensor
    block_starts: Tensor
    total_blocks: Tensor
    row_map: Tensor
    # Startup planning prior only; device counts remain the execution authority.
    mean_rows_per_expert: float | None = None
    routing_top_k: int | None = None


@dataclass(frozen=True, slots=True)
class RoutedOperands:
    """Format-owned pointer columns and compile-time math metadata."""

    tensors: tuple[tuple[Tensor, ...], ...]
    constants: tuple[Any, ...]
    compute: Any


class MoEContract(Protocol):
    compute: Any

    def operands(
        self, operators: PackedLinearOperators, weights: tuple[QuantizedMatrix, ...]
    ) -> RoutedOperands: ...

    def prepare(
        self,
        operators: PackedLinearOperators,
        x: Tensor,
        encoding: QuantizationScheme,
        row_parallel: bool,
        cache: dict[int, tuple[Tensor, Tensor, PackedOperatorBinding]],
    ) -> tuple[Tensor, Tensor, PackedOperatorBinding | None, int]: ...


class PolicyContract(Protocol):
    """Format capabilities and parameters, independent of model identity.

    Optional ``supports_compute_mode``/``select_compute_mode`` choose an
    implementation of the requested arithmetic, never an implicit fallback
    to a different activation precision. An arithmetic capability does not
    promise a native performance kernel: format-owned native binding must fail
    explicitly when that implementation is unavailable. Reference arithmetic
    remains independently usable. Optional local TP completion is an
    arithmetic opt-in, restrictable by ``LOCAL_WEIGHTED_TP_COMPLETION_MODES``.
    """
    def validate(
        self, operation: str, values: object
    ) -> tuple[tuple[str, int | bool | float | str], ...]: ...

    PARAMETERS: dict[str, type[PackedMatmulParameters]]
    FORBIDDEN_OVERRIDES: dict[str, str]


@dataclass(frozen=True, slots=True)
class QuantizationExtension:
    name: str
    format_id: str
    package: str
    checkpoint_methods: tuple[str, ...] = ()
    group_formats: tuple[str, ...] = ()
    ignore_substrings: bool = False

    def __post_init__(self) -> None:
        if not all(isinstance(v, str) and v for v in (self.name, self.format_id, self.package)):
            raise ValueError("extension requires stable name, encoding ID and importable package")

    @overload
    def module(self, role: Literal["checkpoint"]) -> CheckpointContract: ...
    @overload
    def module(self, role: Literal["layout"]) -> LayoutContract: ...
    @overload
    def module(self, role: Literal["reference"]) -> ReferenceContract: ...
    @overload
    def module(self, role: Literal["linear"]) -> LinearContract: ...
    @overload
    def module(self, role: Literal["moe"]) -> MoEContract: ...
    @overload
    def module(self, role: Literal["policy"]) -> PolicyContract: ...
    def module(self, role: str) -> Any:
        if role not in ("checkpoint", "layout", "reference", "linear", "moe", "policy"):
            raise ValueError(f"unknown quantization extension role {role!r}")
        return import_module(f"{self.package}.{role}")

    def family(self, operation: str) -> str:
        if operation not in ("linear", "moe"):
            raise ValueError(f"unknown quantization parameter operation {operation!r}")
        return f"{self.name}_{operation}"
