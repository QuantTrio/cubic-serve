"""Pre-readiness row normalization using NInfer's register/reduction organization."""
import math
from pathlib import Path

import torch

from cubic_serve.ops.cuda.activation import activation_type
from cubic_serve.ops.cuda.native import CudaScalar
from cubic_serve.ops.cuda.native_compiler import compile_native
from cubic_serve.quantization.common.binding import PackedOperatorBinding


def bind(x: torch.Tensor, weight: torch.Tensor, output: torch.Tensor, *,
         epsilon: float, gate: torch.Tensor | None = None,
         zero_centered: bool = True, activation: str = "silu") -> PackedOperatorBinding:
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("row normalization must be bound before capture")
    if (not x.is_cuda or x.dtype not in (torch.float16, torch.bfloat16)
            or x.ndim < 2 or not x.numel() or output.shape != x.shape or output.dtype != x.dtype
            or weight.shape != (x.shape[-1],) or weight.dtype not in (x.dtype, torch.float32)
            or not math.isfinite(epsilon) or epsilon <= 0
            or any(not t.is_contiguous() or t.device != x.device for t in (x, weight, output))
            or gate is not None and (gate.shape != x.shape or gate.dtype != x.dtype
                                    or gate.device != x.device or not gate.is_contiguous())
            or gate is not None and zero_centered or activation not in ("silu", "sigmoid")):
        raise ValueError("row normalization operands differ from the shared ABI")
    for source in (x, weight, *((gate,) if gate is not None else ())):
        if (output.data_ptr()<source.data_ptr()+source.numel()*source.element_size()
                and source.data_ptr()<output.data_ptr()+output.numel()*output.element_size()):
            raise ValueError("row normalization output aliases an input")
    width=x.shape[-1];rows=x.numel()//width
    kind={"fp32":0,"fp16":1,"bf16":2}[str(activation_type(x.dtype,x.device))]
    warp_rows=width<=256
    definitions=dict(WIDTH=width, ROWS=rows,WARP_ROWS=int(warp_rows),
                     IS_BF16=int(x.dtype==torch.bfloat16), WEIGHT_FP32=int(weight.dtype==torch.float32),
                     ACT_KIND=kind,EPILOGUE=2 if gate is not None else 0 if zero_centered else 1,
                     SIGMOID_GATE=int(activation=="sigmoid"))
    prefix="".join(f"#define {key} {value}\n" for key,value in definitions.items()).encode()
    header=Path(__file__).parents[1]/"activation_native.cuh"
    module,entry=compile_native((("entry.cu",prefix+Path(__file__).with_suffix(".cu").read_bytes()),
                                 ("activation_native.cuh",header.read_bytes())),x.device,
                                "row_norm_native",minimum_isa=90 if gate is not None and kind==2 else 80)
    launch=module.bind(entry,(x,weight,gate,output,CudaScalar("f32",float(epsilon))),
                       operator_id="cubic.cuda.row_norm.native.v1",
                       grid=((rows+7)//8 if warp_rows else rows,1,1),block=(256,1,1))
    return PackedOperatorBinding(output,(launch,))
