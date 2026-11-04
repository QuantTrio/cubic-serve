// NInfer's apply_gdn_transition organization, retaining Cubic's original
// low-operand/native and FP32 policies plus the state storage step boundary.
#pragma once
#include <cuda_bf16.h>

__device__ __forceinline__ float transition_warp_sum(float value) {
    #pragma unroll
    for(int mask=16;mask>0;mask>>=1)
        value=__fadd_rn(value,__shfl_xor_sync(0xffffffff,value,mask));
    return value;
}

template<class Low,int Keys,bool Native,bool StateBF16>
__device__ __forceinline__ void apply_gdn_transition(
    float (&history)[4][Keys], const float (&key)[Keys],
    const float (&values)[4], float beta,float decay) {
    #pragma unroll
    for(int r=0;r<4;++r) {
        float memory=0.f;
        #pragma unroll
        for(int c=0;c<Keys;++c) {
            if constexpr(Native)
                history[r][c]=float(Low(__fmul_rn(float(Low(history[r][c])),float(Low(decay)))));
            else history[r][c]=__fmul_rn(history[r][c],decay);
            memory=__fadd_rn(memory,__fmul_rn(history[r][c],key[c]));
        }
        memory=transition_warp_sum(memory);
        float residual;
        if constexpr(Native)
            residual=float(Low(__fmul_rn(float(Low(__fsub_rn(values[r],float(Low(memory))))),float(Low(beta)))));
        else residual=__fmul_rn(__fsub_rn(values[r],memory),beta);
        #pragma unroll
        for(int c=0;c<Keys;++c) {
            history[r][c]=__fadd_rn(history[r][c],__fmul_rn(residual,key[c]));
            if constexpr(StateBF16)history[r][c]=float(__float2bfloat16_rn(history[r][c]));
        }
    }
}
