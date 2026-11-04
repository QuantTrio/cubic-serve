"""Startup-bound projected convolution with optional immutable replay journal."""
from pathlib import Path

import torch

from cubic_serve.ops.cuda.activation import activation_type
from cubic_serve.ops.cuda.native_compiler import compile_native
from cubic_serve.quantization.common.binding import PackedOperatorBinding


def bind(projected: torch.Tensor, weight: torch.Tensor, state: torch.Tensor,
         sources: torch.Tensor, destinations: torch.Tensor, segments: torch.Tensor,
         activity: torch.Tensor, output: torch.Tensor, *,
         values: torch.Tensor | None = None, record: torch.Tensor | None = None) -> PackedOperatorBinding:
    if projected.ndim!=2 or weight.ndim!=2 or state.ndim!=3:
        raise ValueError("projected convolution requires explicit channel/history geometry")
    rows,first=projected.shape;channels,width=weight.shape
    operands=(projected,weight,state,sources,destinations,segments,activity,output,
              *((values,) if values is not None else ()),*((record,) if record is not None else ()))
    if (torch.cuda.is_current_stream_capturing() or not projected.is_cuda
            or projected.dtype not in (torch.float16,torch.bfloat16)
            or min(rows,first,channels,state.shape[0])<1 or not 2<=width<=32
            or state.shape[1:]!=(channels,width-1) or output.shape!=(rows,channels)
            or (values is None and first!=channels)
            or (values is not None and (first>=channels or values.shape!=(rows,channels-first)))
            or (record is not None and record.shape!=(rows,channels))
            or any(t.shape!=(rows,) or t.dtype!=torch.int32 for t in (sources,destinations,segments,activity))
            or any(t.dtype!=projected.dtype for t in (weight,state,output,*((values,) if values is not None else ()),*((record,) if record is not None else ())))
            or any(t.device!=projected.device or not t.is_contiguous() for t in operands)):
        raise ValueError("projected convolution differs from the segmented storage ABI")
    writable=(output,*((record,) if record is not None else (state,)))
    readable=(projected,weight,sources,destinations,segments,activity,*((values,) if values is not None else ()),*((state,) if record is not None else ()))
    for i,target in enumerate(writable):
        for source in (*readable,*writable[i+1:]):
            if (target.data_ptr()<source.data_ptr()+source.numel()*source.element_size()
                    and source.data_ptr()<target.data_ptr()+target.numel()*target.element_size()):
                raise ValueError("projected convolution aliases another operand")
    kind={"fp32":0,"fp16":1,"bf16":2}[str(activation_type(projected.dtype,projected.device))]
    definitions=dict(ROWS=rows,CHANNELS=channels,PROJECTED_CHANNELS=first,WIDTH=width,
                     IS_BF16=int(projected.dtype==torch.bfloat16),ACT_KIND=kind,RECORD_ONLY=int(record is not None))
    definitions['causal_conv_native']=f'causal_conv_native_m{rows}_c{channels}_w{width}'+('_record' if record is not None else '')
    prefix="".join(f"#define {k} {v}\n" for k,v in definitions.items()).encode()
    module,entry=compile_native((("entry.cu",prefix+Path(__file__).with_suffix('.cu').read_bytes()),
                                 ("activation_native.cuh",(Path(__file__).parents[1]/'activation_native.cuh').read_bytes())),
                                projected.device,'causal_conv_native',minimum_isa=90 if kind==2 else 80)
    launch=module.bind(entry,(projected,values,weight,state,sources,destinations,segments,activity,output,record),
                       operator_id='cubic.cuda.conv.record.v1' if record is not None else 'cubic.cuda.conv.native.v1',
                       grid=((channels+127)//128,rows,1),block=(128,1,1))
    return PackedOperatorBinding(output,(launch,),scratch=(record,) if record is not None else ())
