"""FP8-token Marlin binding with original codes and FP32 scale boundaries."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import triton  # type: ignore[import-untyped]
import triton.language as tl  # type: ignore[import-untyped]

from cubic_serve.models.quantization import ActivationFormat
from cubic_serve.ops.cuda.compiled_artifact import bind_tensor
from cubic_serve.ops.cuda.native import CudaScalar, NativeLaunch
from cubic_serve.quantization.common.binding import PackedOperatorBinding, _PackedLaunch
from cubic_serve.quantization.common.marlin.compiler import bind_launch, compile_module
from cubic_serve.quantization.common.marlin.cost import ProbeFormat
from cubic_serve.quantization.common.marlin.cost import select as select_native_geometry
from cubic_serve.quantization.common.marlin.parameters import (
    MarlinParameters,
    shared_bytes,
)
from cubic_serve.quantization.common.marlin.repack import repack_words
from cubic_serve.quantization.extensions.fp8.kernels import (
    _fp8_scale,
    fp8_e4m3fn_decode,
    fp8_e4m3fn_encode,
)

if TYPE_CHECKING:
    from torch import Tensor

    from cubic_serve.models.quantized_checkpoint import QuantizedMatrix
    from cubic_serve.ops.cuda.linear.quantized import PackedLinearOperators
    from cubic_serve.quantization.common.parameters import PackedLinearParameters


@triton.jit  # type: ignore[untyped-decorator]
def quant(  # type: ignore[no-untyped-def]
    X, Q, S, K: tl.constexpr, BLOCK: tl.constexpr, EXTERNAL: tl.constexpr
):
    row = tl.program_id(0)
    k = tl.arange(0, BLOCK)
    x = tl.load(X + row * K + k, k < K, 0).to(tl.float32)
    scale = tl.load(S + row) if EXTERNAL else _fp8_scale(tl.max(tl.abs(x), 0))
    v = tl.minimum(tl.maximum(tl.div_rn(x, scale), -448.0), 448.0)
    tl.store(Q + row * K + k, fp8_e4m3fn_decode(fp8_e4m3fn_encode(v)), k < K)
    if not EXTERNAL:
        tl.store(S + row, scale)


@triton.jit  # type: ignore[untyped-decorator]
def finish(  # type: ignore[no-untyped-def]
    P, XS, WS, Y, M: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr
):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    val = (
        (tl.load(P + idx, idx < M * N, 0) * 256.0)
        * tl.load(XS + idx // N, idx < M * N, 0)
        * tl.load(WS + idx % N, idx < M * N, 0)
    )
    tl.store(Y + idx, val, idx < M * N)


@triton.jit  # type: ignore[untyped-decorator]
def finish_group(  # type: ignore[no-untyped-def]
    P, XS, WS, OUTPUTS, M: tl.constexpr, WIDTHS: tl.constexpr,
    N: tl.constexpr, DTYPE: tl.constexpr, BLOCK: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = idx // N, idx % N
    value = ((tl.load(P + idx, row < M, 0) * 256.0)
             * tl.load(XS + row, row < M, 0)
             * tl.load(WS + col, row < M, 0))
    begin = 0
    for part in tl.static_range(len(WIDTHS)):
        pointer = tl.load(OUTPUTS + part).to(tl.pointer_type(DTYPE))
        width = WIDTHS[part]
        tl.store(pointer + row * width + col - begin, value,
                 (row < M) & (col >= begin) & (col < begin + width))
        begin += width


class GroupedFP8Binding(PackedOperatorBinding):
    """Retain original member outputs behind the private destination table."""

    def __init__(
        self, outputs: tuple[Tensor, ...], weights: tuple[QuantizedMatrix, ...], x: Tensor,
        launches: tuple[_PackedLaunch | NativeLaunch, ...], *, scratch: tuple[Tensor, ...],
    ) -> None:
        super().__init__(outputs[0], launches, scratch=scratch, reusable_scratch=True)
        self.outputs, self.weights, self.input = outputs, weights, x
        self._output_bindings = tuple(bind_tensor(y) for y in outputs)

    def rebind_private_scratch(self, scratch: Tensor) -> None:
        if any(scratch.untyped_storage().data_ptr() == y.untyped_storage().data_ptr()
               for y in self.outputs):
            raise ValueError("private scratch must not alias grouped output")
        super().rebind_private_scratch(scratch)

    def validate_operands(self) -> None:
        if tuple(bind_tensor(y) for y in self.outputs) != self._output_bindings:
            raise ValueError("grouped FP8 output ABI changed")
        super().validate_operands()


def bind_linear(
    operators: PackedLinearOperators, x: Tensor, weight: QuantizedMatrix,
    output: Tensor, params: PackedLinearParameters,
    fp8_token_scales: Tensor | None = None,
) -> PackedOperatorBinding | None:
    return _bind(operators, x, (weight,), (output,), params, fp8_token_scales)


def bind_grouped(
    operators: PackedLinearOperators, x: Tensor, weights: tuple[QuantizedMatrix, ...],
    outputs: tuple[Tensor, ...], params: PackedLinearParameters,
) -> PackedOperatorBinding | None:
    if not getattr(params, "grouped_projections", False):
        return None
    return _bind(operators, x, weights, outputs, params)


def _bind(
    operators: PackedLinearOperators,
    x: Tensor,
    weights: tuple[QuantizedMatrix, ...],
    outputs: tuple[Tensor, ...],
    params: PackedLinearParameters,
    fp8_token_scales: Tensor | None = None,
) -> PackedOperatorBinding | None:
    widths = tuple(w.shape[0] for w in weights)
    n, k = sum(widths), weights[0].shape[1]
    m = x.shape[0]
    if (not isinstance(params, MarlinParameters)
            or any(w.shape[1] != k or not eligible(w, params, x.device) for w in weights)):
        return None
    params = select_native_geometry(
        params, operators=operators, family="fp8_linear", scope=dict(n=widths[0], k=k),
        source=_kernel_source, spec=ProbeFormat(8, torch.float16, torch.float32,
                                              torch.float16, 0x38383838, 1., 1 / 256,
                                              column_scale=True),
        device=x.device, n=n, k=k, experts=1, group=k, rows=m, routed=False,
    )
    bm, bn = params.marlin_block_m, params.marlin_block_n
    shared = shared_bytes(params, bits=8, group=k)
    resource = torch.cuda.get_device_properties(x.device)
    if shared > int(
        getattr(resource, "shared_memory_per_block_optin", resource.shared_memory_per_block)
    ):
        return None
    # Integer-only repack cached with this checkpoint owner; no process-global weights.
    key = ("marlin.fp8.v1", x.device, tuple(
        (bind_tensor(w.payload["weight"]), bind_tensor(w.payload["weight_scale"]))
        for w in weights
    ))
    cache = operators.prepared_weights
    if all((*key, name) in cache for name in ("codes", "ones", "scales")):
        packed = cache[(*key, "codes")][1]
        ones = cache[(*key, "ones")][1]
        weight_scales = cache[(*key, "scales")][1]
    else:
        raw = torch.cat(tuple(w.payload["weight"].view(torch.uint8).to(x.device)
                              for w in weights), dim=0)
        source = raw.contiguous().view(torch.int32).T.contiguous()
        packed = repack_words(source, bits=8, axis="k")
        ones = torch.ones(1, n, dtype=torch.float16, device=x.device)
        cache[(*key, "codes")] = (weights, packed)
        cache[(*key, "ones")] = (weights, ones)
        weight_scales = torch.cat(tuple(w.payload["weight_scale"].to(x.device)
                                        for w in weights), dim=0)
        cache[(*key, "scales")] = (weights, weight_scales)
    code = _kernel_source(params)
    module, entry = compile_module(code, x.device)
    q = torch.empty_like(x, dtype=torch.float16)
    xs = torch.empty(m, device=x.device) if fp8_token_scales is None else fp8_token_scales
    partial = torch.empty(m, n, device=x.device)
    padded = triton.cdiv(m, bm) * bm
    ids = torch.arange(padded, dtype=torch.int32, device=x.device)
    owners = torch.zeros(padded // bm, dtype=torch.int32, device=x.device)
    total = torch.tensor([padded], dtype=torch.int32, device=x.device)
    rw = torch.ones(m, 1, device=x.device)
    args = (
        q,
        packed,
        partial,
        None,
        None,
        None,
        ones,
        None,
        None,
        None,
        ids,
        owners,
        total,
        rw,
        CudaScalar("i32", 1),
        CudaScalar("bool", False),
        CudaScalar("i32", 1),
        CudaScalar("i32", m),
        CudaScalar("i32", n),
        CudaScalar("i32", k),
        None,
        CudaScalar("bool", False),
        CudaScalar("bool", False),
        CudaScalar("bool", True),
        CudaScalar("i32", 0),
    )
    native = bind_launch(
        module,
        entry,
        args,
        operator_id="cubic.cuda.fp8.marlin.v1",
        params=params,
        output_tiles=owners.numel() * (n // bn),
        shared_bytes=shared,
    )
    prep = _PackedLaunch.compile(
        quant,
        (x, q, xs),
        (k, triton.next_power_of_2(k), fp8_token_scales is not None),
        (m,),
        {"enable_fp_fusion": False},
    )
    scratch = (partial, q, ids, owners, total, rw) + ((xs,) if fp8_token_scales is None else ())
    if len(outputs) == 1:
        post = _PackedLaunch.compile(
            finish, (partial, xs, weight_scales, outputs[0]), (m, n, 256),
            (triton.cdiv(m * n, 256),), {"enable_fp_fusion": False},
        )
        return PackedOperatorBinding(
            outputs[0], (prep, native, post), scratch=scratch, reusable_scratch=True
        )
    destinations = torch.tensor([y.data_ptr() for y in outputs],
                                dtype=torch.int64, device=x.device)
    post = _PackedLaunch.compile(
        finish_group, (partial, xs, weight_scales, destinations),
        (m, widths, n, tl.bfloat16 if x.dtype == torch.bfloat16 else tl.float16, 256),
        (triton.cdiv(m * n, 256),), {"enable_fp_fusion": False},
    )
    return GroupedFP8Binding(outputs, weights, x, (prep, native, post),
                             scratch=(*scratch, destinations))


def eligible(weight: QuantizedMatrix, params: PackedLinearParameters, device: torch.device) -> bool:
    if not isinstance(params, MarlinParameters):
        return False
    resource = torch.cuda.get_device_properties(device)
    if shared_bytes(params, bits=8, group=weight.shape[1]) > int(
        getattr(resource, "shared_memory_per_block_optin", resource.shared_memory_per_block)
    ):
        return False
    return (
        getattr(params, "marlin_enabled", False)
        and weight.encoding.activation is ActivationFormat.FP8_TOKEN
        and weight.shape[0] % params.marlin_block_n == 0
        and weight.shape[1] % params.marlin_block_k == 0
    )


def _kernel_source(params: MarlinParameters) -> str:
    bm, bn, bk = params.marlin_block_m, params.marlin_block_n, params.marlin_block_k
    threads, stages = params.marlin_warps * 32, params.marlin_stages
    return f"""#define MARLIN_NAMESPACE_NAME cubic_fp8
#define CUBIC_MARLIN_FP32_OUTPUT 1
#include "libtorch_stable/moe/marlin_moe_wna16/kernel.h"
#include "marlin_template.h"
namespace cubic_fp8 {{template __global__ void Marlin<vllm::kFloat16.id(),vllm::kFE4M3fn.id(),
vllm::kFloat16.id(),vllm::kFloat16.id(),{threads},{bm // 16},{bn // 16},{bk // 16},
false,{stages},-1,false>
(MARLIN_KERNEL_PARAMS); }}"""
