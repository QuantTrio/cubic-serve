"""Startup-bound GDN controls from actual rounded A/B projection operands."""
from pathlib import Path

import torch

from cubic_serve.ops.cuda.activation import activation_type
from cubic_serve.ops.cuda.native_compiler import compile_native
from cubic_serve.quantization.common.binding import PackedOperatorBinding


def bind(a: torch.Tensor,b: torch.Tensor,alog: torch.Tensor,bias: torch.Tensor,
         activity: torch.Tensor,output: torch.Tensor) -> PackedOperatorBinding:
    if a.ndim!=2:
        raise ValueError("GDN controls require a row/head matrix")
    rows,heads=a.shape;operands=(a,b,alog,bias,activity,output)
    if (torch.cuda.is_current_stream_capturing() or not a.is_cuda or min(rows,heads)<1
            or a.dtype not in (torch.bfloat16,torch.float16) or b.shape!=a.shape or b.dtype!=a.dtype
            or alog.shape!=(heads,) or alog.dtype not in (a.dtype,torch.float32)
            or bias.shape!=(heads,) or bias.dtype!=a.dtype
            or activity.shape!=(rows,) or activity.dtype!=torch.int32
            or output.shape!=(rows,heads,2) or output.dtype!=torch.float32
            or any(t.device!=a.device or not t.is_contiguous()for t in operands)):
        raise ValueError("GDN control operands differ from their producing ABI")
    for source in operands[:-1]:
        if (output.data_ptr()<source.data_ptr()+source.numel()*source.element_size()
                and source.data_ptr()<output.data_ptr()+output.numel()*output.element_size()):
            raise ValueError("GDN controls alias a producing operand")
    kind={"fp32":0,"fp16":1,"bf16":2}[str(activation_type(a.dtype,a.device))]
    definitions=dict(ROWS=rows,HEADS=heads,IS_BF16=int(a.dtype==torch.bfloat16),
                     ALOG_FP32=int(alog.dtype==torch.float32),ACT_KIND=kind)
    prefix=''.join(f'#define {key} {value}\n'for key,value in definitions.items()).encode()
    module,entry=compile_native((('entry.cu',prefix+Path(__file__).with_suffix('.cu').read_bytes()),
                                 ('activation_native.cuh',(Path(__file__).parents[1]/'activation_native.cuh').read_bytes())),
                                a.device,'gated_delta_coefficients_native',minimum_isa=90 if kind==2 else 80)
    launch=module.bind(entry,operands,operator_id='cubic.cuda.gdn.coefficients.v1',
                       grid=((rows*heads+127)//128,1,1),block=(128,1,1))
    return PackedOperatorBinding(output,(launch,))
