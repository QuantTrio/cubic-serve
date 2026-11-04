"""One or two same-input Q8 projections, retaining the MMVQ arithmetic.

MMA evaluates the original eight-byte integer dots independently. Eight K
residues and four byte quarters preserve the original FP32 FMA/XOR order;
only the CTA grid and integer implementation change. No weight repack, global
partial buffer, state allocation or runtime selection is needed.
"""
from __future__ import annotations

import math
from functools import cache
from pathlib import Path

import torch
from torch import Tensor

from cubic_serve.ops.cuda.native import CudaScalar, NativeLaunch
from cubic_serve.ops.cuda.native_compiler import compile_native

from .llama import _sources as vector_sources
from .q8_storage import Q8Storage


def selected(*, m: int, widths: tuple[int, ...], sms: int) -> bool:
    """Require two CTA waves before using the four-token, sixteen-column tiles.

    Narrow projections can underfill this grid despite having enough vector
    warps. Pooling members supplies work without changing their reductions.
    The conservative wave floor protects narrow projections; the caller also
    retains the existing vector/matrix crossover and permanent M1 binding.
    """
    if len(widths) not in (1, 2) or min(m, *widths, sms) <= 0:
        raise ValueError("Q8 grouping requires positive geometry and SM supply")
    tiles = sum(math.ceil(n / 16) for n in widths) * math.ceil(m / 4)
    return m > 1 and tiles >= 2 * sms


@cache
def _sources(dtype: torch.dtype, planar: bool = False) -> tuple[tuple[str, bytes], ...]:
    header = f"#define IS_BF16 {int(dtype == torch.bfloat16)}\n".encode()
    if planar:
        header += b"#define PLANAR_Q8 1\n"
    return (("entry.cu", header + Path(__file__).with_suffix(".cu").read_bytes()),
            *vector_sources(dtype, 8, False, planar=planar)[1:])


def bind_pair(
    x: Tensor, weights: tuple[Tensor, Tensor], outputs: tuple[Tensor, Tensor],
    quantized_input: Tensor,
    *, activity: Tensor | None = None,
) -> NativeLaunch:
    """Consume operands validated by both original Q8_1 projection bindings.

    The caller retains their common activation producer and exclusive scratch.
    Both weights and both outputs remain explicit guarded launch operands.
    """
    return _bind(
        x, weights, outputs, quantized_input, activity=activity,
        second_width=outputs[1].shape[1], operator_id="cubic.cuda.ggml.q8_group.v1",
    )


def bind_single(
    x: Tensor, weight: Tensor, output: Tensor, quantized_input: Tensor,
    *, activity: Tensor | None = None,
) -> NativeLaunch:
    """Reuse the same native grid with no second member's tiles or writes.

    Explicit aliases retain valid guarded tensor identities; a zero second
    width prevents the unused ports from being read or written by the kernel.
    """
    return _bind(
        x, (weight, weight), (output, output), quantized_input, activity=activity,
        second_width=0, operator_id="cubic.cuda.ggml.q8_single.v1",
    )


def _bind(
    x: Tensor, weights: tuple[Tensor, Tensor], outputs: tuple[Tensor, Tensor],
    quantized_input: Tensor, *, activity: Tensor | None, second_width: int,
    operator_id: str,
) -> NativeLaunch:
    m, k = x.shape
    n0, n1 = outputs[0].shape[1], second_width
    planar = isinstance(weights[0], Q8Storage)
    if any(isinstance(w, Q8Storage) != planar for w in weights):
        raise ValueError("Q8 group requires a shared carrier layout")
    weights = tuple(w.storage if isinstance(w, Q8Storage) else w for w in weights)
    module, entry = compile_native(_sources(x.dtype, planar), x.device, "ggml_q8_group")
    return module.bind(
        entry,
        (*weights, quantized_input, *outputs, activity,
         *(CudaScalar("i32", v) for v in (m, n0, n1, k)),
         *(CudaScalar("i64", output.stride(0)) for output in outputs)),
        operator_id=operator_id,
        grid=(math.ceil(n0 / 16) + math.ceil(n1 / 16), math.ceil(m / 4), 1),
        block=(32, 8, 1),
    )
