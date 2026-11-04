"""Shared Marlin source compilation; no neighboring checkout dependency."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch
from cuda.bindings import driver

from cubic_serve.ops.cuda.native import CudaModule, CudaScalar, NativeLaunch, _check
from cubic_serve.ops.cuda.native_compiler import compile_native
from cubic_serve.quantization.common.marlin.parameters import MarlinParameters


@dataclass(frozen=True)
class KernelResources:
    resident: int
    registers: int
    local_bytes: int


def resident_blocks(function, params: MarlinParameters, shared_bytes: int) -> int:
    """Query actual compiled residency in the caller's CUDA context."""
    resident = int(_check(driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(
        function, params.marlin_warps * 32, shared_bytes,
    ))[0])
    return min(resident, params.marlin_blocks_per_sm) if params.marlin_blocks_per_sm else resident


def kernel_resources(module: CudaModule, entry: str, params: MarlinParameters,
                     shared_bytes: int) -> KernelResources:
    """Read native allocation facts without binding operands or launching work."""
    with torch.cuda.device(module.device):
        function = _check(driver.cuModuleGetFunction(module.handle, entry.encode()))[0]
        attr = driver.CUfunction_attribute
        dynamic = attr.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
        if shared_bytes > int(_check(driver.cuFuncGetAttribute(dynamic, function))[0]):
            _check(driver.cuFuncSetAttribute(function, dynamic, shared_bytes))
        return KernelResources(
            resident_blocks(function, params, shared_bytes),
            int(_check(driver.cuFuncGetAttribute(attr.CU_FUNC_ATTRIBUTE_NUM_REGS, function))[0]),
            int(_check(driver.cuFuncGetAttribute(
                attr.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, function,
            ))[0]),
        )


def bind_launch(
    module: CudaModule, entry: str,
    arguments: tuple[torch.Tensor | CudaScalar | None, ...], *,
    operator_id: str, params: MarlinParameters, output_tiles: int, shared_bytes: int,
) -> NativeLaunch:
    """Bound whole-K CTA supply by output work and actual kernel residency.

    Register allocation and shared-memory limits come from the loaded kernel,
    including driver allocation granularity. Routed output_tiles is a metadata
    capacity bound; the kernel still consumes the device's actual route count.
    This is a resource rule, not a claim of optimal tile geometry or utilization.
    """
    if type(output_tiles) is not int or output_tiles < 1:
        raise ValueError("Marlin output tile capacity must be positive")
    sms = torch.cuda.get_device_properties(module.device).multi_processor_count
    threads = params.marlin_warps * 32
    launch = module.bind(
        entry, arguments, operator_id=operator_id, grid=(sms, 1, 1),
        block=(threads, 1, 1), shared_bytes=shared_bytes,
    )
    with torch.cuda.device(module.device):
        resident = resident_blocks(launch.function, params, shared_bytes)
    if resident < 1:
        raise RuntimeError("Marlin kernel has no resident CTA for its declared resources")
    grid = (min(output_tiles, sms * resident), 1, 1)
    if grid == launch.grid:
        return launch
    return module.bind(
        entry, arguments, operator_id=operator_id, grid=grid,
        block=(threads, 1, 1), shared_bytes=shared_bytes,
    )


@lru_cache(maxsize=32)
def compile_module(source: str, device: torch.device) -> tuple[CudaModule, str]:
    """Compile immutable source content once before graph binding, targeting SM80 PTX."""
    root = Path(__file__).with_name("native")
    sources = {
        str(p.relative_to(root)): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.suffix in (".h", ".hpp", ".cuh")
    }
    sources["entry.cu"] = source.encode()
    return compile_native(tuple(sorted(sources.items())), device, "marlin")
