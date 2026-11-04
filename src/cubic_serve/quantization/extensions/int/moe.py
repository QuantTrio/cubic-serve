from __future__ import annotations

from typing_extensions import Never

from typing import TYPE_CHECKING

from cubic_serve.quantization.api import RoutedOperands

from .marlin import bind_a8_routed as _marlin_a8_routed
from .marlin import bind_grouped as _marlin_grouped
from .marlin import bind_routed as _marlin_routed

if TYPE_CHECKING:
    from torch import Tensor

    from cubic_serve.models.quantization import QuantizationScheme
    from cubic_serve.models.quantized_checkpoint import QuantizedMatrix
    from cubic_serve.ops.cuda.linear.quantized import PackedLinearOperators
    from cubic_serve.quantization.common.binding import PackedOperatorBinding


def compute(*args: object, **kwargs: object) -> Never:
    raise RuntimeError("INT MoE arithmetic requires a selected native or packed binding")


def bind_grouped(operators, *args, **kwargs):
    if operators.config.compute_mode == "wna8":
        x, gate, up, output, rows, top_k, params = args
        if not params.grouped_projections or gate[0].encoding != up[0].encoding:
            return None
        n, k = gate[0].shape
        if any(w.shape != (n, k) for w in (*gate, *up)):
            raise ValueError("INT WNA8 grouped Marlin requires matching projection shapes")
        return _marlin_a8_routed(
            operators, x, gate, output, rows, params, top_k, projection_groups=(gate, up)
        )
    return _marlin_grouped(operators, *args, **kwargs)


def bind_routed(operators, *args, **kwargs):
    if operators.config.compute_mode == "wna8":
        x, weights, output, rows, params = args
        # Gate/up expand one input row into top-k routes; down receives those
        # routes as its input rows. The binding geometry owns that distinction.
        physical_rows = output.numel() // output.shape[-1]
        if physical_rows % x.shape[0]:
            raise ValueError("routed output rows must be a multiple of input rows")
        return _marlin_a8_routed(
            operators, x, weights, output, rows, params, physical_rows // x.shape[0]
        )
    return _marlin_routed(operators, *args, **kwargs)


def prepare(
    operators: PackedLinearOperators,
    x: Tensor,
    encoding: QuantizationScheme,
    row_parallel: bool,
    cache: dict[int, tuple[Tensor, Tensor, PackedOperatorBinding]],
) -> tuple[Tensor, Tensor, PackedOperatorBinding | None, int]:
    if operators.config.compute_mode in ("wna16", "wna8"):
        return x, x, None, 0
    raise ValueError("INT MoE requires WNA16 or WNA8")


def operands(
    operators: PackedLinearOperators, weights: tuple[QuantizedMatrix, ...]
) -> RoutedOperands:
    matrix = weights[0]
    raise ValueError(
        "INT MoE requires Marlin; generic GEMM dispatch is not admitted: "
        f"shape={matrix.shape}, encoding={matrix.encoding}"
    )
