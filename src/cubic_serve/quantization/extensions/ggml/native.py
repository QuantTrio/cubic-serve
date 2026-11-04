"""Portable original-block WNA16 primitive; not a registered serving backend.

Original FP32 decoded weights round to the activation's FP16/BF16 operand
boundary before FP32 MMA. Activations are not requantized to Q8_1. This is a
different compute policy from llama.cpp MMQ/A8, preserving the same checkpoint.
"""

from __future__ import annotations

import math
from functools import cache
from pathlib import Path

import torch
from gguf import GGML_QUANT_SIZES, GGMLQuantizationType
from torch import Tensor

from cubic_serve.ops.cuda.native import CudaScalar
from cubic_serve.ops.cuda.native_compiler import compile_native
from cubic_serve.quantization.api import RoutedRows
from cubic_serve.quantization.common.binding import PackedOperatorBinding

from .encoding import iq3_grid


@cache
def _grid(device: torch.device) -> Tensor:
    return torch.from_numpy(iq3_grid().copy()).to(device)


def bind_matmul(
    x: Tensor, packed: Tensor, type_id: int, output: Tensor, *,
    activity: Tensor | None = None, rows: RoutedRows | None = None, input_top_k: int = 1,
    vector_rows: int = 0, column_offset: int = 0,
    column_warps: int | None = None,
) -> PackedOperatorBinding:
    """Bind a fixed dense or original-expert-order contraction before capture.

    ``rows`` is the common routing contract, with 16-row block prefixes.
    ``input_top_k`` maps gate/up route IDs to token rows; down uses one.
    An omitted activity tensor explicitly declares every input row owned.
    This primitive does not admit arbitrary request-time execution or graphs.
    """
    supported = {0, 1, 8, 14, 20, 21, 23, 30}
    if type(vector_rows) is not int or vector_rows not in (0, 1, 4, 16):
        raise ValueError("GGML vector row tile must be 0, 1, 4 or 16")
    tile_m = vector_rows or 16
    if type(type_id) is not int or type_id not in supported:
        raise ValueError("GGML native primitive does not implement this weight encoding")
    if (not x.is_cuda or x.ndim != 2 or not x.is_contiguous()
            or x.dtype not in (torch.bfloat16, torch.float16) or min(x.shape) <= 0
            or type(input_top_k) is not int or input_top_k <= 0):
        raise ValueError("GGML WNA16 requires contiguous CUDA FP16/BF16 activation rows")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("GGML native binding must precede graph capture")
    if packed.ndim not in (2, 3) or packed.dtype != torch.uint8 or not packed.is_contiguous():
        raise ValueError("GGML weight must retain its contiguous encoded byte rows")
    e, n = (1, packed.shape[0]) if packed.ndim == 2 else packed.shape[:2]
    k = x.shape[1]
    if column_warps is None:
        sms = torch.cuda.get_device_properties(x.device).multi_processor_count
        # A column-poor vector grid leaves SMs idle. Share its K contraction
        # across the existing four warps, with only CTA-private FP32 partials.
        column_warps = (4 if vector_rows == 1 and rows is None and k >= 4 * 32 * 8
                        and math.ceil(n / 4) * x.shape[0] < sms else 1)
    if (type(column_warps) is not int or column_warps not in (1, 4)
            or (column_warps != 1 and vector_rows != 1)):
        raise ValueError("cooperative columns require a single-row vector primitive")
    block, size = GGML_QUANT_SIZES[GGMLQuantizationType(type_id)]
    if (type(column_offset) is not int or not 0 <= column_offset < block
            or min(e, n) <= 0 or packed.shape[-1] != math.ceil((column_offset + k) / block) * size):
        raise ValueError("GGML original block geometry differs from the logical weight")
    if rows is None and (e != 1 or input_top_k != 1):
        raise ValueError("expert weights require explicit common route ownership")
    routes = x.shape[0] * input_top_k
    if (output.shape != (routes, n) or output.dtype != x.dtype
            or output.stride(1) != 1 or output.stride(0) < n
            or output.device != x.device or packed.device != x.device):
        raise ValueError("GGML matmul operands have incompatible storage")
    if activity is not None and (activity.shape != (x.shape[0],)
                                 or activity.dtype not in (torch.bool, torch.int32)
                                 or activity.device != x.device or not activity.is_contiguous()):
        raise ValueError("GGML activity must explicitly identify input-row ownership")
    routing: tuple[Tensor | None, ...] = (None,) * 5
    if rows is not None:
        routing = (rows.counts, rows.starts, rows.block_starts, rows.total_blocks, rows.row_map)
        minimum = (e, e, e, 1, routes)
        if any(t is None or t.dtype != torch.int32 or t.device != x.device
               or not t.is_contiguous() or t.ndim != 1 or t.numel() < length
               for t, length in zip(routing, minimum, strict=True)):
            raise ValueError("GGML routing differs from the common int32 row contract")
    for operand in (x, packed, activity, *routing):
        if operand is not None and (
            output.data_ptr() < operand.data_ptr() + operand.numel() * operand.element_size()
            and operand.data_ptr() < output.data_ptr()
            + ((output.shape[0] - 1) * output.stride(0) + n) * output.element_size()
        ):
            raise ValueError("GGML output must not alias an input or routing field")
    prefix = (f"#define IS_BF16 {int(x.dtype == torch.bfloat16)}\n"
              f"#define GGML_VECTOR_ROWS {vector_rows}\n"
              f"#define GGML_WEIGHT_TYPE {type_id}\n"
              f"#define GGML_COLUMN_WARPS {column_warps}\n"
              "#define GGML_TRANSPORT_K 64\n").encode()
    source = prefix + Path(__file__).with_name("kernels.cu").read_bytes()
    module, entry = compile_native((
        ("entry.cu", source),
        ("decode.cuh", Path(__file__).with_name("decode.cuh").read_bytes()),
    ), x.device, "ggml_wna16")
    tasks = math.ceil(routes / tile_m) if rows is None else math.ceil(
        (routes + min(routes, e) * (tile_m - 1)) / tile_m
    )
    launch = module.bind(
        entry, (x, packed, _grid(x.device), output, activity, *routing,
                CudaScalar("i32", x.shape[0]), CudaScalar("i32", n), CudaScalar("i32", k),
                CudaScalar("i32", packed.shape[-1]), CudaScalar("i32", type_id),
                CudaScalar("i32", e), CudaScalar("i32", input_top_k),
                CudaScalar("i64", output.stride(0)),
                CudaScalar("i32", 0 if activity is None else activity.element_size()),
                CudaScalar("i32", column_offset)),
        operator_id="cubic.cuda.ggml.wna16.block.v1",
        grid=(math.ceil(n / (4 // column_warps if vector_rows else 64)), tasks, 1),
        block=(128, 1, 1),
    )
    return PackedOperatorBinding(
        output, (launch,), row_tile_count=math.ceil(routes / tile_m) if rows is None else None
    )
