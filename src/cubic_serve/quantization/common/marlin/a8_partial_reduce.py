"""Ascending FP32 reduction of dense Marlin A8 accumulator fragments."""
from functools import cache
import hashlib
from pathlib import Path

import torch

from cubic_serve.ops.cuda.native import CudaScalar, NativeLaunch
from cubic_serve.ops.cuda.native_compiler import compile_native

_SOURCE = Path(__file__).with_name("native") / "a8_partial_reduce.cu"


@cache
def source_identity() -> tuple[str, str]:
    return "cubic.marlin.a8-partial-reduce.v1", hashlib.sha256(_SOURCE.read_bytes()).hexdigest()


@cache
def _module(device: torch.device):
    return compile_native((("entry.cu", _SOURCE.read_bytes()),), device,
                          "marlin_a8_partial_reduce", minimum_isa=80)


def bind(partials: torch.Tensor, output: torch.Tensor, *, m: int, n: int,
         block_m: int, block_n: int, partitions: int) -> NativeLaunch:
    """Bind one rank-private output; grouped destinations keep their existing reducer."""
    dimensions = (m, n, block_m, block_n, partitions)
    if (any(type(v) is not int or not 0 < v < 2**31 - 4 for v in dimensions)
            or block_m < 16 or block_m & (block_m - 1)
            or block_n < 32 or block_n & (block_n - 1)
            or n % block_n or n % 64
            or not partials.is_cuda or partials.dtype != torch.float32
            or output.device != partials.device
            or output.dtype not in (torch.bfloat16, torch.float16, torch.float32)
            or output.shape != (m, n)
            or not partials.is_contiguous() or not output.is_contiguous()
            or partials.numel() != ((m + block_m - 1) // block_m) * (n // block_n)
               * partitions * block_m * block_n):
        raise ValueError("dense A8 partial reduction operands disagree with their layout")
    left, right = partials.data_ptr(), output.data_ptr()
    if left < right + output.numel() * output.element_size() and right < left + partials.numel() * 4:
        raise ValueError("dense A8 partial reduction cannot alias its source")
    module, entry = _module(partials.device)
    dtype = {torch.bfloat16: 0, torch.float16: 1, torch.float32: 2}[output.dtype]
    geometry = (m, n, block_m.bit_length() - 1, block_n.bit_length() - 1,
                n // block_n, block_m * block_n, partitions, dtype)
    return module.bind(
        entry, (partials, output, *(CudaScalar("i32", value) for value in geometry)),
        operator_id="cubic.cuda.marlin.a8_partial_reduce.v1",
        grid=(n // 64, (m + 3) // 4, 1), block=(256, 1, 1),
    )
