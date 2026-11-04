"""Pre-readiness NInfer recurrence organization with Cubic numerical boundaries."""
from pathlib import Path

import torch

from cubic_serve.ops.cuda.activation import activation_type
from cubic_serve.ops.cuda.native import CudaScalar
from cubic_serve.ops.cuda.native_compiler import compile_native
from cubic_serve.ops.strategies import attention_precision
from cubic_serve.quantization.common.binding import PackedOperatorBinding


def bind(operands: tuple[torch.Tensor, ...], activity: torch.Tensor, *,
         q_heads: int, scale: float, epsilon: float = 1e-6,
         coefficients: torch.Tensor | None = None,
         journal: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
         state_compute: str = "simt", normalized_inputs: bool = False,
         mma_warps: int = 1) -> PackedOperatorBinding:
    """Startup-only, segmented per-row checkpoint publication; no selector.

    The owner validates index generations and topology-wide write ownership.
    These operands follow GatedDeltaDecodePrimitive's eleven-tensor ABI.
    """
    if len(operands) != 11 or torch.cuda.is_current_stream_capturing():
        raise ValueError("native recurrence requires startup operands")
    packed, normalized, a, b, alog, bias, state, sources, destinations, segments, output = operands
    rows = packed.shape[0]
    mma = state_compute == "mma"
    if (state_compute not in ("simt", "mma") or type(normalized_inputs) is not bool
            or mma_warps not in (1, 2, 4)
            or (mma and (not normalized_inputs or attention_precision() != "auto"))
            or (not mma and normalized_inputs)):
        raise ValueError("native recurrence compute differs from its producing precision contract")
    if state.ndim != 4:
        raise ValueError("native recurrence state must be slots/heads/value/key")
    _, heads, value_dim, key_dim = state.shape
    width = 2*q_heads*key_dim+heads*value_dim
    if (not packed.is_cuda or packed.dtype not in (torch.float16, torch.bfloat16)
            or min(rows, q_heads, heads, value_dim, key_dim, state.shape[0]) < 1
            or heads % q_heads or packed.shape != (rows, width)
            or normalized.shape != packed.shape or output.shape != (rows, heads, value_dim)
            or a.shape != (rows, heads) or b.shape != a.shape
            or alog.shape != (heads,) or bias.shape != alog.shape
            or any(t.dtype != packed.dtype for t in (normalized, a, b, bias, output))
            or alog.dtype not in (packed.dtype, torch.float32)
            or state.dtype not in (torch.bfloat16, torch.float32)
            or any(t.shape != (rows,) or t.dtype != torch.int32
                   for t in (sources, destinations, segments, activity))
            or not state[0].is_contiguous()
            or state.stride(0) < heads * value_dim * key_dim
            or any(t.device != packed.device or (t is not state and not t.is_contiguous())
                   for t in (*operands, activity))):
        raise ValueError("native recurrence differs from the segmented state ABI")
    if coefficients is not None and (coefficients.shape != (rows, heads, 2)
            or coefficients.dtype != torch.float32 or coefficients.device != packed.device
            or not coefficients.is_contiguous()):
        raise ValueError("native recurrence coefficient ABI differs")
    if journal is not None:
        if (len(journal)!=3 or journal[0].shape!=(rows,q_heads,key_dim)
                or journal[1].shape!=(rows,heads,value_dim) or journal[2].shape!=(rows,heads,2)
                or journal[0].dtype!=packed.dtype or journal[1].dtype!=packed.dtype
                or journal[2].dtype!=torch.float32
                or any(t.device!=packed.device or not t.is_contiguous() for t in journal)):
            raise ValueError("native recurrence journal differs from the actual-operand ABI")
    writable = (*(() if mma else (normalized,)), state, output, *(journal or ()))
    readable = (packed, a, b, alog, bias, sources, destinations, segments, activity,
                *((normalized,) if mma else ()),
                *((coefficients,) if coefficients is not None else ()))
    for index, target in enumerate(writable):
        for other in (*readable, *writable[index+1:]):
            def span(t):
                return (1 + sum((n - 1) * s for n, s in zip(t.shape, t.stride(), strict=True))) * t.element_size()
            if (target.data_ptr() < other.data_ptr()+span(other)
                    and other.data_ptr() < target.data_ptr()+span(target)):
                raise ValueError("native recurrence has overlapping storage")
    kind = {"fp32": 0, "fp16": 1, "bf16": 2}[str(activation_type(packed.dtype, packed.device))]
    definitions = dict(ROWS=rows, Q_HEADS=q_heads, V_HEADS=heads, V_DIM=value_dim,
                       K_DIM=key_dim, PACKED_DIM=width, IS_BF16=int(packed.dtype==torch.bfloat16),
                       STATE_SLOT_STRIDE=state.stride(0),
                       STATE_BF16=int(state.dtype==torch.bfloat16), ACT_KIND=kind,
                       ALOG_FP32=int(alog.dtype==torch.float32),
                       NATIVE=int(attention_precision()=="auto"), PRECOMPUTED=int(coefficients is not None),
                       RECORD_ONLY=int(journal is not None), STATE_MMA=int(mma), STATE_WARPS=mma_warps)
    definitions["gated_delta_native"] = (f"gated_delta_native_m{rows}_q{q_heads}_v{heads}_k{key_dim}_d{value_dim}"
                                          + ("_record" if journal is not None else "")
                                          + (f"_mma_w{mma_warps}" if mma else ""))
    prefix="".join(f"#define {name} {value}\n" for name,value in definitions.items()).encode()
    header=Path(__file__).parents[1]/"activation_native.cuh"
    module, entry=compile_native((("entry.cu",prefix+Path(__file__).with_suffix(".cu").read_bytes()),
                                 ("activation_native.cuh",header.read_bytes()),
                                 ("gated_delta_transition.cuh",Path(__file__).with_name("gated_delta_transition.cuh").read_bytes()),
                                 *((name,Path(__file__).with_name(name).read_bytes()) for name in
                                   ("gated_delta_mma.cuh", "gated_delta_mma_forward.cuh"))),
                                packed.device,"gated_delta_native",minimum_isa=90 if kind==2 else 80)
    launch=module.bind(entry, (*operands[:-1], activity, output, coefficients,
                              CudaScalar("f32",float(scale)), CudaScalar("f32",float(epsilon)),*(journal or ())),
                       operator_id=("cubic.cuda.gdn.record.v1" if journal is not None else "cubic.cuda.gdn.native.v1"),
                       grid=(heads,rows,(value_dim+16*mma_warps-1)//(16*mma_warps) if mma else (value_dim+15)//16),
                       block=(mma_warps*32 if mma else 128,1,1))
    return PackedOperatorBinding(output,(launch,),scratch=journal or ())
