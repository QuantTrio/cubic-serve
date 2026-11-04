"""Startup-bound accepted-prefix fold of immutable actual recurrence operands."""
from pathlib import Path

import torch

from cubic_serve.ops.cuda.native_compiler import compile_native
from cubic_serve.ops.strategies import attention_precision
from cubic_serve.quantization.common.binding import PackedOperatorBinding


def bind(keys: torch.Tensor, values: torch.Tensor, coefficients: torch.Tensor,
         state: torch.Tensor, sources: torch.Tensor, destinations: torch.Tensor,
         segments: torch.Tensor, activity: torch.Tensor, request_rows: torch.Tensor,
         decisions: torch.Tensor, *,
         convolution: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
         state_compute: str = "simt", mma_warps: int = 1) -> PackedOperatorBinding:
    operands=(keys,values,coefficients,state,sources,destinations,segments,activity,request_rows,decisions)
    if keys.ndim!=4 or values.ndim!=4 or state.ndim!=4 or decisions.ndim!=2:
        raise ValueError("GDN fold requires explicit layer/row/owner geometry")
    layers,rows,qh,kd=keys.shape;_,_,vh,vd=values.shape
    mma=state_compute=="mma"
    if (state_compute not in ("simt","mma") or mma_warps not in (1,2,4)
            or (mma and attention_precision()!="auto")):
        raise ValueError("GDN fold differs from its producing compute contract")
    if (torch.cuda.is_current_stream_capturing() or not keys.is_cuda
            or keys.dtype not in (torch.float16,torch.bfloat16)
            or min(layers,rows,qh,kd,vh,vd,state.shape[0],decisions.shape[0])<1 or vh%qh
            or values.shape[:2]!=(layers,rows) or values.dtype!=keys.dtype
            or coefficients.shape!=(layers,rows,vh,2) or coefficients.dtype!=torch.float32
            or state.shape[1:]!=(vh,vd,kd) or state.dtype not in (torch.bfloat16,torch.float32)
            or any(t.shape!=(layers,rows) or t.dtype!=torch.int32 for t in (sources,destinations))
            or any(t.shape!=(rows,) or t.dtype!=torch.int32 for t in (segments,activity,request_rows))
            or decisions.shape[1]!=2 or decisions.dtype!=torch.int64
            or any(t.device!=keys.device or not t.is_contiguous() for t in operands)):
        raise ValueError("GDN fold operands differ from the journal ABI")
    for source in (*operands[:3],*operands[4:]):
        if (state.data_ptr()<source.data_ptr()+source.numel()*source.element_size()
                and source.data_ptr()<state.data_ptr()+state.numel()*state.element_size()):
            raise ValueError("GDN fold state aliases a producing operand")
    if convolution is not None:
        if len(convolution)!=4:
            raise ValueError("GDN convolution fold requires record, state and owner maps")
        record,conv_state,conv_sources,conv_destinations=convolution
        if (record.ndim!=3 or record.shape[:2]!=(layers,rows) or record.shape[2]<1
                or conv_state.ndim!=3 or conv_state.shape[0]<1 or conv_state.shape[1]!=record.shape[2]
                or not 1<=conv_state.shape[2]<=31
                or any(t.dtype!=keys.dtype for t in (record,conv_state))
                or any(t.shape!=(layers,rows) or t.dtype!=torch.int32 for t in (conv_sources,conv_destinations))
                or any(t.device!=keys.device or not t.is_contiguous() for t in convolution)):
            raise ValueError("GDN convolution fold differs from the journal ABI")
        for target in (state,conv_state):
            for source in (*operands[:3],*operands[4:],record,conv_sources,conv_destinations,
                           *((conv_state,) if target is state else (state,))):
                if (target.data_ptr()<source.data_ptr()+source.numel()*source.element_size()
                        and source.data_ptr()<target.data_ptr()+target.numel()*target.element_size()):
                    raise ValueError("GDN convolution fold state aliases another operand")
    definitions=dict(ROWS=rows,REQUESTS=decisions.shape[0],Q_HEADS=qh,V_HEADS=vh,V_DIM=vd,
                     K_DIM=kd,IS_BF16=int(keys.dtype==torch.bfloat16),
                     STATE_BF16=int(state.dtype==torch.bfloat16),NATIVE=int(attention_precision()=="auto"),
                     FOLD_CONV=int(convolution is not None),STATE_MMA=int(mma),STATE_WARPS=mma_warps)
    if convolution is not None:definitions.update(CONV_CHANNELS=record.shape[2],CONV_WIDTH=conv_state.shape[2]+1)
    definitions['gated_delta_fold']=f'gated_delta_fold_l{layers}_m{rows}_q{qh}_v{vh}_k{kd}_d{vd}'+('_conv' if convolution is not None else '')+(f'_mma_w{mma_warps}' if mma else '')
    prefix="".join(f"#define {key} {value}\n" for key,value in definitions.items()).encode()
    module,entry=compile_native((("entry.cu",prefix+Path(__file__).with_suffix('.cu').read_bytes()),
                                 ("gated_delta_transition.cuh",Path(__file__).with_name('gated_delta_transition.cuh').read_bytes()),
                                 *((name,Path(__file__).with_name(name).read_bytes()) for name in
                                   ('gated_delta_mma.cuh','gated_delta_mma_fold.cuh'))),
                                keys.device,'gated_delta_fold')
    launch=module.bind(entry,(*operands,*(convolution or ())),operator_id='cubic.cuda.gdn.fold.v1',
                       grid=(vh,rows,layers*((vd+16*mma_warps-1)//(16*mma_warps) if mma else (vd+15)//16)),
                       block=(mma_warps*32 if mma else 128,1,1))
    return PackedOperatorBinding(state,(launch,))
