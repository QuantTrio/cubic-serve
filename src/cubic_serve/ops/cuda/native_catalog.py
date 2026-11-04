"""Source-owned AOT recipes, independent of models, caches and GPU discovery.

Prebuild the shared startup tile portfolio and format arithmetic families. Matrix
dimensions and request counts are runtime operands, never catalog lookup keys.
Additional validated source variants retain explicit startup compilation.
"""
from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path

from .native_bundle import source_key


def _groups(base):
    """Power-of-two groups within the native pipeline's four-stage contract."""
    group = 32
    while group <= base.marlin_block_k*4:
        yield group
        group *= 2


def recipes(package: Path):
    # Import only when building: importing the catalog itself initializes no
    # framework, device, native module, model, selector or qualification state.
    import torch
    from cubic_serve.quantization.common.marlin.parameters import (
        MarlinParameters, pipeline_compatible,
    )
    from cubic_serve.quantization.extensions.int.marlin_source import _a8_source, _source
    from cubic_serve.quantization.extensions.cubic.marlin import _kernel_source as cubic
    from cubic_serve.quantization.extensions.cubic.policy import CubicLinearParameters
    from cubic_serve.quantization.extensions.awq.marlin import _kernel_source as awq
    from cubic_serve.quantization.extensions.fp8.marlin import _kernel_source as fp8
    from cubic_serve.quantization.extensions.nvfp4.marlin import _kernel_source as nvfp4
    from cubic_serve.quantization.extensions.nvfp4.policy import NVFP4LinearParameters
    from cubic_serve.quantization.extensions.ggml.llama import _sources, _matrix_sources
    from cubic_serve.quantization.extensions.ggml.q8_group import _sources as q8_group
    from cubic_serve.quantization.extensions.ggml.q8_mma import _sources as q8_mma
    from cubic_serve.quantization.extensions.ggml.q8_storage import sources as q8_storage

    headers = package/'quantization/common/marlin/native'
    shared = {str(p.relative_to(headers)): p.read_bytes() for p in sorted(headers.rglob('*'))
              if p.suffix in ('.h', '.hpp', '.cuh')}
    seen = set()

    def unit(namespace, sources, family, minimum_isa=80):
        sources = dict(sources)
        key = source_key(namespace, sources, minimum_isa=minimum_isa)
        if key in seen:
            return None
        seen.add(key)
        return dict(namespace=namespace, sources=sources, minimum_isa=minimum_isa,
                    family=family)

    def marlin(source, family):
        return unit('marlin', {**shared, 'entry.cu': source.encode()}, family)

    base = MarlinParameters()
    # Precompile the complete baseline arithmetic families. Adaptive or explicitly
    # overridden tiles use the same source owner through declared extensions;
    # enumeration must not multiply every independent tuning knob into the wheel.
    for p in (base,):
        for group in (8, 16, 24, 32, 48, 64, 96, 128, 192, 256):
            quantum = max(16, math.gcd(group, p.marlin_block_k))
            if not pipeline_compatible(p, group=quantum, single_subtile=True):
                continue
            for scale in ('F16', 'BF16', 'F32'):
                for dtype in ('kFloat16', 'kBFloat16'):
                    for bits in range(2, 9):
                        for dense, partitions in ((False, 0), (True, 0), (True, 1)):
                            item = marlin(_source(p, dtype, bits, group, quantum, scale,
                                                  dense, partitions, n=0, k=0), 'int_wna16')
                            if item:
                                yield item

    for p in (base,):
        for group in _groups(p):
            if not pipeline_compatible(p, group=group, single_subtile=True):
                continue
            for dtype in ('kFloat16', 'kBFloat16'):
                for bits in range(2, 9):
                    for dense, partitions in ((False, 0), (True, 0), (True, 1)):
                        # These are the actual symmetric, inline-zero and
                        # external-zero branches of the format-owned binders.
                        policies = ((False, False), (True, False))
                        if bits < 8:
                            # Inline zero is consumed inside MMA; it does not
                            # use the external FP32 zero-correction epilogue.
                            policies += ((False, True),)
                        for fp32, inline_zero in policies:
                            item = marlin(_a8_source(
                                p, dtype, bits, 0, 0, group, fp32_output=fp32,
                                dense=dense, fixed_partitions=partitions,
                                inline_zero=inline_zero), 'int_wna8')
                            if item:
                                yield item

    for original in (CubicLinearParameters(),):
        for group in _groups(original):
            reuse = max(1, group//original.marlin_block_k)
            p = replace(original, marlin_stages=math.lcm(original.marlin_stages, reuse))
            if not pipeline_compatible(p, group=group):
                continue
            for dtype in ('kFloat16', 'kBFloat16'):
                for bits in range(2, 9):
                    for dense, partitions in ((False, 0), (True, 0), (True, 1)):
                        for fragment in ((4, 8) if dense and bits <= 4 else (8,)):
                            item = marlin(cubic(p, dtype, bits, group,
                                                fragment_bits=fragment, dense_linear=dense,
                                                fixed_partitions=partitions), 'cubic_wna16')
                            if item:
                                yield item

    for p in (base,):
        for group in _groups(p):
            if pipeline_compatible(p, group=group):
                for dtype in ('kFloat16', 'kBFloat16'):
                    item = marlin(awq(p, dtype, group), 'awq')
                    if item:
                        yield item
        item = marlin(fp8(p), 'fp8')
        if item:
            yield item
    for p in (NVFP4LinearParameters(),):
        for dense in (False, True):
            item = marlin(nvfp4(p, dense_linear=dense), 'nvfp4')
            if item:
                yield item

    for dtype in (torch.float16, torch.bfloat16):
        for planar in (False, True):
            item = unit('ggml_llama_q8', _sources(dtype, 8, True, planar=planar), 'ggml_q8')
            if item:
                yield item
            for type_id in (8, 14, 20, 21, 23):
                for rows in (1, 2, 4, 8):
                    item = unit('ggml_llama_dot', _sources(dtype, type_id, False, rows,
                                                         planar), 'ggml_mmvq')
                    if item:
                        yield item
                for rows, owners in ((8, False), (16, False), (16, True), (32, True)):
                    transports = (False, True) if type_id == 8 and rows >= 16 else (False,)
                    for register in transports:
                        item = unit('ggml_llama_mmq', _matrix_sources(
                            dtype, type_id, rows, register_transport=register,
                            planar=planar, query_row_owners=owners), 'ggml_mmq')
                        if item:
                            yield item
            item = unit('ggml_q8_group', q8_group(dtype, planar), 'ggml_q8_group')
            if item:
                yield item
            for pipelined in (False, True):
                item = unit('ggml_q8_mma', q8_mma(dtype, pipelined, planar), 'ggml_q8_mma')
                if item:
                    yield item
    item = unit('ggml_q8_storage', q8_storage(), 'ggml_q8_storage')
    if item:
        yield item
    for bf16 in (0, 1):
        item = unit('cache_rotation_native', {'entry.cu':
                    f'#define NATIVE_BF16 {bf16}\n'.encode()
                    + (package/'ops/cuda/cache_rotation_native.cu').read_bytes()}, 'cache_rotation')
        if item:
            yield item
    item = unit('marlin_a8_partial_reduce', {'entry.cu':
                (package/'quantization/common/marlin/native/a8_partial_reduce.cu').read_bytes()},
                'int_wna8_reduce')
    if item:
        yield item
