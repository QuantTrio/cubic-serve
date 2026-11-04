// Execution organization adapted from NInfer (Apache-2.0), b9a6b5c,
// src/ops/linear_attention/gated_delta_net/recurrent.cuh. Four value rows
// per warp, four warps per block, register state across a candidate segment.
// Cubic owns normalization storage, operand precision and state step rounding.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "activation_native.cuh"
#include "gated_delta_transition.cuh"
#if IS_BF16
using scalar=__nv_bfloat16;
#else
using scalar=__half;
#endif
#if ALOG_FP32
using control_scalar=float;
#else
using control_scalar=scalar;
#endif
#if STATE_BF16
using state_scalar=__nv_bfloat16;
#else
using state_scalar=float;
#endif
constexpr int Keys=(K_DIM+31)/32;
__device__ __forceinline__ float stored(float value) { return float(scalar(value)); }
__device__ __forceinline__ float warp_sum(float value) {
    #pragma unroll
    for(int mask=16;mask>0;mask>>=1)
        value=__fadd_rn(value,__shfl_xor_sync(0xffffffff,value,mask));
    return value;
}
#if STATE_MMA
#include "gated_delta_mma_forward.cuh"
#else
extern "C" __launch_bounds__(128) __global__ void gated_delta_native(
    const scalar* __restrict__ packed, scalar* __restrict__ normalized,
    const scalar* __restrict__ a,const scalar* __restrict__ b,
    const control_scalar* __restrict__ alog,const scalar* __restrict__ bias,
    state_scalar* __restrict__ state,
    const int* __restrict__ sources,const int* __restrict__ destinations,
    const int* __restrict__ segments,const int* __restrict__ activity,
    scalar* __restrict__ output,const float* __restrict__ coefficients,
    float scale,float epsilon
#if RECORD_ONLY
    , scalar* __restrict__ key_record,scalar* __restrict__ value_record,
    float* __restrict__ coefficient_record
#endif
    ) {
    const int lane=threadIdx.x&31,warp=threadIdx.x>>5;
    const int head=blockIdx.x,start=blockIdx.y;
    const int length=min(max(segments[start],0),ROWS-start);
    if(!length||!activity[start])return;
    const int keyhead=head/(V_HEADS/Q_HEADS);
    const int vbase=blockIdx.z*16+warp*4;
    const long long stride=STATE_SLOT_STRIDE;
    const long long base=static_cast<long long>(head)*V_DIM*K_DIM;
    float history[4][Keys];
    #pragma unroll
    for(int r=0;r<4;++r) {
        #pragma unroll
        for(int c=0;c<Keys;++c) {
            const int key=lane*Keys+c;
            history[r][c]=(vbase+r<V_DIM&&key<K_DIM)
                ?float(state[static_cast<long long>(sources[start])*stride+base+(vbase+r)*K_DIM+key]):0.f;
        }
    }
    for(int token=0;token<length;++token) {
        const int row=start+token;
        if(!activity[row])continue;
        const long long qbase=static_cast<long long>(row)*PACKED_DIM+keyhead*K_DIM;
        float query[Keys],key[Keys],qs=0.f,ks=0.f;
        #pragma unroll
        for(int c=0;c<Keys;++c) {
            const int i=lane*Keys+c;
            query[c]=i<K_DIM?float(packed[qbase+i]):0.f;
            key[c]=i<K_DIM?float(packed[qbase+Q_HEADS*K_DIM+i]):0.f;
            qs=__fadd_rn(qs,__fmul_rn(query[c],query[c]));
            ks=__fadd_rn(ks,__fmul_rn(key[c],key[c]));
        }
        const float qi=rsqrtf(__fadd_rn(warp_sum(qs),epsilon));
        const float ki=rsqrtf(__fadd_rn(warp_sum(ks),epsilon));
        #pragma unroll
        for(int c=0;c<Keys;++c) {
            query[c]=stored(__fmul_rn(query[c],qi));
            key[c]=stored(__fmul_rn(key[c],ki));
            if(head%(V_HEADS/Q_HEADS)==0&&blockIdx.z==0&&warp==0&&lane*Keys+c<K_DIM) {
                normalized[qbase+lane*Keys+c]=scalar(query[c]);
                normalized[qbase+Q_HEADS*K_DIM+lane*Keys+c]=scalar(key[c]);
#if RECORD_ONLY
                key_record[(static_cast<long long>(row)*Q_HEADS+keyhead)*K_DIM+lane*Keys+c]=scalar(key[c]);
#endif
            }
        }
        const long long control=static_cast<long long>(row)*V_HEADS+head;
        float beta,decay;
#if PRECOMPUTED
        beta=coefficients[control*2];decay=coefficients[control*2+1];
#else
        beta=stored(activation_sigmoid_native(float(b[control])));
        const float alpha=__fadd_rn(float(a[control]),float(bias[head]));
        const float softplus=alpha<=20.f?log1pf(expf(alpha)):alpha;
        decay=expf(__fmul_rn(-expf(float(alog[head])),softplus));
#endif
        float values[4];
        #pragma unroll
        for(int r=0;r<4;++r) {
            values[r]=vbase+r<V_DIM?float(packed[static_cast<long long>(row)*PACKED_DIM+
                2*Q_HEADS*K_DIM+head*V_DIM+vbase+r]):0.f;
#if RECORD_ONLY
            if(lane==r&&vbase+r<V_DIM)
                value_record[(static_cast<long long>(row)*V_HEADS+head)*V_DIM+vbase+r]=scalar(values[r]);
#endif
        }
#if RECORD_ONLY
        if(blockIdx.z==0&&warp==0&&lane==0) {
            coefficient_record[control*2]=beta;coefficient_record[control*2+1]=decay;
        }
#endif
        apply_gdn_transition<scalar,Keys,NATIVE,STATE_BF16>(history,key,values,beta,decay);
        #pragma unroll
        for(int r=0;r<4;++r) {
            float readout=0.f;
            #pragma unroll
            for(int c=0;c<Keys;++c) {
#if NATIVE
                const float q=stored(__fmul_rn(query[c],stored(scale)));
                readout=__fadd_rn(readout,__fmul_rn(stored(history[r][c]),q));
#else
                readout=__fadd_rn(readout,__fmul_rn(history[r][c],__fmul_rn(query[c],scale)));
#endif
#if !RECORD_ONLY
                const int i=lane*Keys+c;
                if(vbase+r<V_DIM&&i<K_DIM)
                    state[static_cast<long long>(destinations[row])*stride+base+(vbase+r)*K_DIM+i]
                        =state_scalar(history[r][c]);
#endif
            }
            readout=warp_sum(readout);
            if(lane==r&&vbase+r<V_DIM)
                output[(static_cast<long long>(row)*V_HEADS+head)*V_DIM+vbase+r]=scalar(readout);
        }
    }
}
#endif
