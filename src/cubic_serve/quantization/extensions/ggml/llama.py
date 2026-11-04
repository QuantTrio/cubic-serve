"""llama.cpp packed Q8-activation dots through the native launch ABI.

This primitive is deliberately explicit about its Q8_1 activation contract. It
is not registered as WNA16 or a qualified serving mode. Binding precedes capture;
callers own the usual topology lease and all referenced storage.
"""
from __future__ import annotations

import math
from functools import cache
from pathlib import Path
from typing import Literal

import torch
from gguf import GGML_QUANT_SIZES, GGMLQuantizationType
from torch import Tensor

from cubic_serve.ops.cuda.native import CudaModule, CudaScalar
from cubic_serve.ops.cuda.native_compiler import compile_native
from cubic_serve.quantization.api import RoutedRows
from cubic_serve.quantization.common.binding import PackedOperatorBinding

from .q8_storage import Q8Storage


@cache
def _sources(
    dtype: torch.dtype, type_id: int, quantize: bool, rows_per_warp: int = 1,
    planar: bool = False,
) -> tuple[tuple[str, bytes], ...]:
    root = Path(__file__).parent
    prefix = (f"#define IS_BF16 {int(dtype == torch.bfloat16)}\n"
              f"#define WEIGHT_TYPE {0 if quantize else type_id}\n"
              f"#define QUANTIZE {int(quantize)}\n"
              f"#define ROWS_PER_WARP {rows_per_warp}\n").encode()
    if planar:
        prefix += b"#define PLANAR_Q8 1\n"
    return (("entry.cu", prefix + (root / "llama.cu").read_bytes()),
            *((("q8_storage.cuh", (root / "q8_storage.cuh").read_bytes()),) if planar else ()),
            *(("upstream/" + name, (root / "upstream" / name).read_bytes())
              for name in ("common.cuh", "ggml-common.h", "vecdotq.cuh")))


@cache
def _matrix_sources(
    dtype: torch.dtype, type_id: int, row_tile: int, register_transport: bool = False,
    planar: bool = False, query_row_owners: bool = False,
) -> tuple[tuple[str, bytes], ...]:
    root = Path(__file__).parent
    prefix = (f"#define IS_BF16 {int(dtype == torch.bfloat16)}\n"
              f"#define WEIGHT_TYPE {type_id}\n#define ROW_TILE {row_tile}\n"
              f"#define REGISTER_TRANSPORT {int(register_transport)}\n").encode()
    if planar:
        prefix += b"#define PLANAR_Q8 1\n"
    if query_row_owners:
        prefix += b"#define QUERY_ROW_OWNERS 1\n"
    return (("entry.cu", prefix + (root / "mmq.cu").read_bytes()),
            *((("q8_storage.cuh", (root / "q8_storage.cuh").read_bytes()),) if planar else ()),
            *(("upstream/" + name, (root / "upstream" / name).read_bytes())
              for name in ("common.cuh", "ggml-common.h", "vecdotq.cuh", "mmq-tiles.cuh")))


@cache
def _resident_blocks(module: CudaModule, entry: str, shared_bytes: int,
                     threads: int = 128) -> int:
    """Read the compiled function's actual resources once; never time candidates.

    Cache keys retain code modules, not launch bindings or their weight tensors.
    Grid geometry does not alter residency for a fixed block size.
    """
    from cuda.bindings import driver

    with torch.cuda.device(module.device):
        result, function = driver.cuModuleGetFunction(module.handle, entry.encode())
        if int(result):
            raise RuntimeError("GGML matrix function resource query failed")
        result, blocks = driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(
            function, threads, shared_bytes,
        )
    if int(result) or blocks <= 0:
        raise RuntimeError("GGML matrix function has no admissible resident block")
    return int(blocks)


@cache
def _local_bytes(module: CudaModule, entry: str) -> int:
    from cuda.bindings import driver

    with torch.cuda.device(module.device):
        result, function = driver.cuModuleGetFunction(module.handle, entry.encode())
        if int(result):
            raise RuntimeError("GGML matrix function resource query failed")
        result, value = driver.cuFuncGetAttribute(
            driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, function,
        )
    if int(result):
        raise RuntimeError("GGML matrix local storage query failed")
    return int(value)


def bind_matmul(
    x: Tensor, packed: Tensor | Q8Storage, type_id: int, output: Tensor, *,
    activity: Tensor | None = None, rows: RoutedRows | None = None, input_top_k: int = 1,
    algorithm: Literal["auto", "vector", "matrix"] = "auto",
    matrix_rows: int | None = None,
) -> PackedOperatorBinding:
    """Original-block integer dot, with explicit source-row ownership.

    GGML's Q8_1 scale is rounded to FP16 after codes are chosen using the FP32
    scale. We preserve that arithmetic rather than reusing another A8 encoding.
    """
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("llama.cpp binding must precede capture")
    if algorithm not in ("auto", "vector", "matrix"):
        raise ValueError("unknown GGML contraction algorithm")
    baseline_rows = 16 if rows is None else 8
    if matrix_rows is not None and (
        type(matrix_rows) is not int or matrix_rows not in (baseline_rows, 2 * baseline_rows)
        or algorithm != "matrix"
    ):
        raise ValueError("GGML forced matrix rows differ from the declared kernel portfolio")
    if (not x.is_cuda or x.ndim != 2 or not x.is_contiguous()
            or x.dtype not in (torch.bfloat16, torch.float16) or min(x.shape) <= 0
            or x.shape[1] % 32 or type(input_top_k) is not int or input_top_k <= 0):
        raise ValueError("llama.cpp Q8_1 requires contiguous FP16/BF16 CUDA rows aligned to 32")
    if type(type_id) is not int or type_id not in (8, 14, 20, 21, 23):
        raise ValueError("llama.cpp primitive does not support this GGML encoding")
    prepared = packed if isinstance(packed, Q8Storage) else None
    if prepared is not None:
        if type_id != 8 or prepared.k != x.shape[1]:
            raise ValueError("prepared Q8 storage differs from its projection")
        packed = prepared.storage
    if (packed.dtype != torch.uint8 or packed.ndim not in ((1,) if prepared else (2, 3))
            or not packed.is_contiguous() or packed.device != x.device):
        raise ValueError("llama.cpp requires original contiguous CUDA block bytes")
    m, k = x.shape
    e, n = ((prepared.experts, prepared.n) if prepared is not None else
            (1, packed.shape[0]) if packed.ndim == 2 else packed.shape[:2])
    block, size = GGML_QUANT_SIZES[GGMLQuantizationType(type_id)]
    routes = m * input_top_k
    row_bytes = k // block * size
    if (k % block or min(e, n) <= 0
            or (prepared is None and packed.shape[-1] != row_bytes)):
        raise ValueError("llama.cpp primitive requires complete original block ownership")
    if (output.shape != (routes, n) or output.dtype != x.dtype or output.device != x.device
            or output.stride(1) != 1 or output.stride(0) < n):
        raise ValueError("llama.cpp output has incompatible row storage")
    if activity is not None and (activity.shape != (m,) or activity.dtype != torch.int32
                                or activity.device != x.device or not activity.is_contiguous()):
        raise ValueError("llama.cpp activity requires explicit contiguous int32 row ownership")
    routing: tuple[Tensor | None, Tensor | None, Tensor | None] = (None, None, None)
    if rows is None:
        if e != 1 or input_top_k != 1:
            raise ValueError("llama.cpp expert weights require explicit routing")
    else:
        routing = (rows.counts, rows.starts, rows.row_map)
        if any(t is None or t.dtype != torch.int32 or t.device != x.device
               or t.ndim != 1 or not t.is_contiguous() or t.numel() < length
               for t, length in zip(routing, (e, e, routes), strict=True)):
            raise ValueError("llama.cpp routing differs from the shared row contract")
    end = output.data_ptr() + ((routes - 1) * output.stride(0) + n) * output.element_size()
    if any(t is not None and output.data_ptr() < t.data_ptr() + t.numel()*t.element_size()
           and t.data_ptr() < end for t in (x, packed, activity, *routing)):
        raise ValueError("llama.cpp output must not overlap its inputs")
    scratch = torch.empty((m, k // 32, 36), dtype=torch.uint8, device=x.device)
    quant, qentry = compile_native(_sources(x.dtype, type_id, True), x.device, "ggml_llama_q8")
    qlaunch = quant.bind(qentry, (x, scratch, activity, CudaScalar("i32", m), CudaScalar("i32", k)),
                         operator_id="cubic.cuda.ggml.llama.q8_1.v1",
                         grid=(math.ceil(k / 256), m, 1), block=(256, 1, 1))
    from .policy import routed_output_rows_per_warp, select_matrix, select_routed_matrix

    hardware = torch.cuda.get_device_properties(x.device)
    resources = {"n": n, "k": k, "row_bytes": row_bytes,
                 "sms": hardware.multi_processor_count, "l2_bytes": hardware.L2_cache_size}
    matrix = algorithm == "matrix" or (algorithm == "auto" and (
        select_matrix(m=m, **resources) if rows is None else
        select_routed_matrix(mean_rows=rows.mean_rows_per_expert, experts=e, **resources)))
    if matrix:
        if rows is not None and (
            rows.block_starts is None or rows.total_blocks is None
            or any(t.dtype != torch.int32 or t.device != x.device or not t.is_contiguous()
                   for t in (rows.block_starts, rows.total_blocks))
            or rows.block_starts.shape != (e,) or rows.total_blocks.shape != (1,)
        ):
            raise ValueError("GGML MMQ requires the common route block prefix")
        row_tile = baseline_rows if matrix_rows is None else matrix_rows
        query_row_owners = row_tile > baseline_rows
        register_transport = False
        module, entry = compile_native(_matrix_sources(x.dtype, type_id, row_tile,
                                                       planar=prepared is not None,
                                                       query_row_owners=query_row_owners), x.device,
                                       "ggml_llama_mmq")
        route_args = ((None,) * 5 if rows is None else (
            rows.counts, rows.starts, rows.block_starts, rows.total_blocks, rows.row_map))
        tiles = (math.ceil(m / row_tile) if rows is None else
                 min(routes, math.ceil(routes / row_tile) + e - 1))
        # IQ4_NL feeds original codes/scales directly to MMA register
        # fragments. Other formats retain their decoded shared-memory tile.
        shared = 0 if type_id == int(GGMLQuantizationType.IQ4_NL) else (
            4 * (row_tile * 36 + 64 * 76)
        )
        if k % 256 and type_id != int(GGMLQuantizationType.IQ4_NL) and prepared is None:
            shared += 64 * (256 // block) * size
        launch = module.bind(entry, (packed, scratch, output, activity, *route_args,
                              *(CudaScalar("i32", v) for v in
                                (m, n, k, row_bytes, e, input_top_k)),
                              CudaScalar("i64", output.stride(0))),
                      operator_id="cubic.cuda.ggml.llama.mmq.v1",
                      grid=(math.ceil(n / 64), tiles, 1), block=(32, 4, 1), shared_bytes=shared)
        if type_id == int(GGMLQuantizationType.Q8_0) and rows is None:
            from .policy import select_register_matrix_transport

            register, register_entry = compile_native(
                _matrix_sources(x.dtype, type_id, row_tile, True,
                                planar=prepared is not None,
                                query_row_owners=query_row_owners), x.device, "ggml_llama_mmq",
            )
            if select_register_matrix_transport(
                m=m, n=n, k=k, row_tile=row_tile, sms=hardware.multi_processor_count,
                shared_resident=_resident_blocks(module, entry, shared),
                register_resident=_resident_blocks(register, register_entry, 0),
            ):
                launch = register.bind(
                    register_entry, launch.arguments, operator_id=launch.operator_id,
                    grid=launch.grid, block=launch.block,
                )
                register_transport = True
        if (algorithm == "auto" and rows is None
                and type_id == int(GGMLQuantizationType.Q8_0)):
            from .q8_mma import bind as bind_mma, prefer_over_matrix

            if prefer_over_matrix(x, output, launch, planar=prepared is not None):
                launch = bind_mma(x, (prepared if prepared is not None else packed,),
                                  (output,), scratch, activity=activity)
        if (matrix_rows is None and launch.operator_id == "cubic.cuda.ggml.llama.mmq.v1"
                and e * n * row_bytes > hardware.L2_cache_size):
            from .policy import prefer_wider_matrix_rows

            wide_rows = 2 * row_tile
            wide_module, wide_entry = compile_native(
                _matrix_sources(x.dtype, type_id, wide_rows, register_transport,
                                planar=prepared is not None, query_row_owners=True),
                x.device, "ggml_llama_mmq",
            )
            wide_shared = (0 if launch.shared_bytes == 0 else
                           4 * (wide_rows * 36 + 64 * 76))
            if wide_shared and k % 256 and prepared is None:
                wide_shared += 64 * (256 // block) * size
            if prefer_wider_matrix_rows(
                m=m, n=n, k=k, row_bytes=row_bytes, experts=e,
                mean_rows=None if rows is None else rows.mean_rows_per_expert,
                sms=hardware.multi_processor_count, l2_bytes=hardware.L2_cache_size,
                baseline_rows=row_tile, wider_rows=wide_rows,
                baseline_resident=_resident_blocks(
                    launch.module, launch.entry, launch.shared_bytes,
                ),
                wider_resident=_resident_blocks(wide_module, wide_entry, wide_shared),
                baseline_local=_local_bytes(launch.module, launch.entry),
                wider_local=_local_bytes(wide_module, wide_entry),
            ):
                tiles = (math.ceil(m / wide_rows) if rows is None else
                         min(routes, math.ceil(routes / wide_rows) + e - 1))
                launch = wide_module.bind(
                    wide_entry, launch.arguments, operator_id=launch.operator_id,
                    grid=(math.ceil(n / 64), tiles, 1), block=(32, 4, 1), shared_bytes=wide_shared,
                )
                row_tile = wide_rows
        declared_rows = row_tile if launch.operator_id == "cubic.cuda.ggml.llama.mmq.v1" else 8
        return PackedOperatorBinding(output, (qlaunch, launch), scratch=(scratch,),
                                     reusable_scratch=True, row_tile_size=declared_rows)
    # Preserve each original row's dot and XOR reduction, with sufficient CTA
    # work before expanding register reuse. Dense contraction stays unchanged.
    rows_per_warp = (1 if rows is None else routed_output_rows_per_warp(
        n=n, routes=routes, sms=hardware.multi_processor_count,
        max_threads_per_sm=hardware.max_threads_per_multi_processor,
    ))
    dot, entry = compile_native(_sources(x.dtype, type_id, False, rows_per_warp,
                                         planar=prepared is not None),
                                x.device, "ggml_llama_dot")
    launch = dot.bind(entry, (packed, scratch, output, activity, *routing,
                              *(CudaScalar("i32", v) for v in
                                (m, n, k, row_bytes, e, input_top_k)),
                              CudaScalar("i64", output.stride(0))),
                      operator_id="cubic.cuda.ggml.llama.mmvq.v1",
                      grid=(math.ceil(n / (4 * rows_per_warp)), routes, 1), block=(32, 4, 1))
    return PackedOperatorBinding(output, (qlaunch, launch), scratch=(scratch,),
                                 reusable_scratch=True, row_tile_size=1)
