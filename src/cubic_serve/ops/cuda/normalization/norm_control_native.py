"""NInfer fused norm/control ownership with the original normalized storage rule."""
import math
from pathlib import Path

import torch

from cubic_serve.ops.cuda.native import CudaScalar
from cubic_serve.ops.cuda.activation import activation_type
from cubic_serve.ops.cuda.native_compiler import compile_native
from cubic_serve.quantization.common.binding import PackedOperatorBinding


def bind(x: torch.Tensor, norm_weight: torch.Tensor, a_weight: torch.Tensor,
         b_weight: torch.Tensor, normalized: torch.Tensor, a: torch.Tensor,
         b: torch.Tensor, *, epsilon: float,
         gating: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None) -> PackedOperatorBinding:
    operands=(x,norm_weight,a_weight,b_weight,normalized,a,b)
    if (torch.cuda.is_current_stream_capturing() or not x.is_cuda or x.ndim!=2
            or x.dtype not in (torch.float16,torch.bfloat16) or not x.numel()
            or a_weight.ndim!=2 or b_weight.shape!=a_weight.shape
            or a_weight.shape[1]!=x.shape[1] or a_weight.shape[0]<1
            or norm_weight.shape!=(x.shape[1],) or normalized.shape!=x.shape
            or a.shape!=(x.shape[0],a_weight.shape[0]) or b.shape!=a.shape
            or not math.isfinite(epsilon) or epsilon<=0
            or any(t.device!=x.device or t.dtype!=x.dtype or not t.is_contiguous() for t in operands)):
        raise ValueError("norm/control operands differ from the original dense weight ABI")
    for i,target in enumerate((normalized,a,b)):
        for source in (*operands[:4],*(normalized,a,b)[i+1:]):
            if (target.data_ptr()<source.data_ptr()+source.numel()*source.element_size()
                    and source.data_ptr()<target.data_ptr()+target.numel()*target.element_size()):
                raise ValueError("norm/control output aliases another operand")
    rows,width=x.shape;heads=a_weight.shape[0];tile=min(rows,2)
    if gating is not None:
        if len(gating)!=3:
            raise ValueError("control gating requires alog, bias and coefficients")
        alog,bias,coefficients=gating
        if (alog.shape!=(heads,) or alog.dtype not in (x.dtype,torch.float32)
                or bias.shape!=(heads,) or bias.dtype!=x.dtype
                or coefficients.shape!=(rows,heads,2) or coefficients.dtype!=torch.float32
                or any(t.device!=x.device or not t.is_contiguous() for t in gating)):
            raise ValueError("control gating operands differ from the recurrence ABI")
        writable=(normalized,a,b,coefficients)
        for i,target in enumerate(writable):
            for source in (*operands[:4],alog,bias,*writable[i+1:]):
                if (target.data_ptr()<source.data_ptr()+source.numel()*source.element_size()
                        and source.data_ptr()<target.data_ptr()+target.numel()*target.element_size()):
                    raise ValueError("control gating aliases another operand")
    kind={"fp32":0,"fp16":1,"bf16":2}[str(activation_type(x.dtype,x.device))] if gating else 0
    definitions=dict(ROWS=rows,WIDTH=width,HEADS=heads,TILE=tile,IS_BF16=int(x.dtype==torch.bfloat16),
                     WITH_GATES=int(gating is not None),ACT_KIND=kind,
                     ALOG_TYPE="float" if gating and gating[0].dtype==torch.float32 else "scalar")
    definitions['norm_control_native']=f'norm_control_native_m{rows}_h{heads}_d{width}'+('_gating' if gating else '')
    prefix="".join(f"#define {key} {value}\n" for key,value in definitions.items()).encode()
    sources=(("entry.cu",prefix+Path(__file__).with_suffix('.cu').read_bytes()),)
    if gating:sources+=(("activation_native.cuh",(Path(__file__).parents[1]/'activation_native.cuh').read_bytes()),)
    module,entry=compile_native(sources,x.device,'norm_control_native',minimum_isa=90 if kind==2 else 80)
    launch=module.bind(entry,(*operands,CudaScalar('f32',float(epsilon)),*(gating or ())),
                       operator_id='cubic.cuda.norm_control.native.v1',grid=(heads,(rows+tile-1)//tile,1),
                       block=(512,1,1))
    return PackedOperatorBinding(normalized,(launch,))
