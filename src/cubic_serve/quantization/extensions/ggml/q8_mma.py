"""Q8_0/Q8_1 integer MMA with an explicitly declared full-group reduction."""
from __future__ import annotations

import math
from functools import cache
from pathlib import Path

import torch
from torch import Tensor

from cubic_serve.ops.cuda.native import CudaScalar, NativeLaunch
from cubic_serve.ops.cuda.native_compiler import compile_native

from .q8_group import _sources as original_sources
from .q8_storage import Q8Storage


def selected(*, m: int, widths: tuple[int, ...], sms: int) -> bool:
    """Use a complete eight-row tile only with at least two CTA waves.

    The caller retains the original MMQ crossover. M1 and four-row reviews
    keep their existing arithmetic; no model or device-generation key exists.
    """
    if len(widths) not in (1, 2) or min(m, *widths, sms) <= 0:
        raise ValueError("Q8 MMA requires positive geometry and SM supply")
    return m >= 8 and sum(math.ceil(n / 16) for n in widths) * math.ceil(m / 8) >= 2 * sms


@cache
def _sources(dtype: torch.dtype, pipelined: bool = False,
             planar: bool = False) -> tuple[tuple[str, bytes], ...]:
    header = (f"#define IS_BF16 {int(dtype == torch.bfloat16)}\n"
              f"#define PIPELINED {int(pipelined)}\n").encode()
    if planar:
        header += b"#define PLANAR_Q8 1\n"
    return (("entry.cu", header + Path(__file__).with_suffix(".cu").read_bytes()),
            *original_sources(dtype, planar=planar)[1:])


def bind(
    x: Tensor, weights: tuple[Tensor | Q8Storage, ...], outputs: tuple[Tensor, ...],
    quantized_input: Tensor, *, activity: Tensor | None = None,
) -> NativeLaunch:
    """Consume already validated operands and the existing Q8_1 producer.

    One member aliases the unused second ports with zero tiles. This changes
    no packed codes or scales and allocates no global partial workspace.
    """
    if len(weights) not in (1, 2) or len(outputs) != len(weights):
        raise ValueError("Q8 MMA requires one or two matched projection members")
    m, k = x.shape
    n0, n1 = outputs[0].shape[1], outputs[1].shape[1] if len(outputs) == 2 else 0
    ws = weights if len(weights) == 2 else (weights[0], weights[0])
    planar = isinstance(ws[0], Q8Storage)
    if any(isinstance(w, Q8Storage) != planar for w in ws):
        raise ValueError("Q8 MMA group requires a shared carrier layout")
    if planar and any(w.k != k or w.n != y.shape[1] or w.experts != 1
                      for w, y in zip(ws, outputs if len(outputs) == 2 else
                                      (outputs[0], outputs[0]), strict=True)):
        raise ValueError("Q8 MMA prepared geometry differs from its consumers")
    ws = tuple(w.storage if isinstance(w, Q8Storage) else w for w in ws)
    ys = outputs if len(outputs) == 2 else (outputs[0], outputs[0])
    # A single K tile cannot overlap transport with another tile's compute.
    pipelined = k // 32 > 16
    module, entry = compile_native(_sources(x.dtype, pipelined, planar), x.device, "ggml_q8_mma")
    return module.bind(
        entry,
        (*ws, quantized_input, *ys, activity,
         *(CudaScalar("i32", v) for v in (m, n0, n1, k)),
         *(CudaScalar("i64", y.stride(0)) for y in ys)),
        operator_id="cubic.cuda.ggml.q8_mma.v1",
        grid=(math.ceil(n0 / 16) + math.ceil(n1 / 16), math.ceil(m / 8), 1),
        block=(32, 8, 1),
        shared_bytes=27392 if pipelined else 0,
    )


def prefer_over_matrix(x: Tensor, output: Tensor, matrix: NativeLaunch, *,
                       planar: bool = False) -> bool:
    """Price the actual compiled implementations, without timing at binding."""
    if x.shape[1] <= 512:
        return False
    from .llama import _resident_blocks
    from .policy import select_pipelined_q8_matrix

    module, entry = compile_native(_sources(x.dtype, True, planar), x.device, "ggml_q8_mma")
    return select_pipelined_q8_matrix(
        m=x.shape[0], n=output.shape[1], k=x.shape[1],
        sms=torch.cuda.get_device_properties(x.device).multi_processor_count,
        matrix_shared=bool(matrix.shared_bytes),
        matrix_resident=_resident_blocks(matrix.module, "llama_mmq", matrix.shared_bytes),
        pipeline_resident=_resident_blocks(module, entry, 27392, threads=256),
    )
