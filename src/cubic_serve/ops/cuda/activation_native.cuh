// Native translation of activation.py's immutable arithmetic policy.
// ACT_KIND: 0=FP32, 1=FP16, 2=BF16 (native exponential requires SM90).
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cmath>

__device__ __forceinline__ float activation_round(float x) {
#if ACT_KIND == 2
    return __bfloat162float(__float2bfloat16_rn(x));
#elif ACT_KIND == 1
    return __half2float(__float2half_rn(x));
#else
    return x;
#endif
}

__device__ __forceinline__ float activation_exp_nonpositive(float x) {
#if ACT_KIND == 2
    const auto input = __bfloat16_as_ushort(__float2bfloat16_rn(x));
    unsigned short output;
    asm("ex2.approx.ftz.bf16 %0, %1;" : "=h"(output) : "h"(input));
    return __bfloat162float(__ushort_as_bfloat16(output));
#elif ACT_KIND == 1
    const auto input = __half_as_ushort(__float2half_rn(x));
    unsigned short output;
    asm("ex2.approx.f16 %0, %1;" : "=h"(output) : "h"(input));
    return __half2float(__ushort_as_half(output));
#else
    return expf(x);
#endif
}

__device__ __forceinline__ float activation_sigmoid_native(float x) {
#if ACT_KIND == 0
    return __fdiv_rn(1.f, __fadd_rn(1.f, expf(-x)));
#else
    const float magnitude = activation_round(fabsf(x));
    const float argument = activation_round(__fmul_rn(-magnitude,
                                      activation_round(1.4426950408889634f)));
    const float exponential = activation_exp_nonpositive(argument);
    const float denominator = activation_round(__fadd_rn(1.f, exponential));
    return activation_round(__fdiv_rn(x >= 0 ? 1.f : exponential, denominator));
#endif
}

__device__ __forceinline__ float activation_silu_native(float x) {
#if ACT_KIND == 0
    return __fdiv_rn(x, __fadd_rn(1.f, expf(-x)));
#else
    const float gate = activation_sigmoid_native(x);
#if ACT_KIND == 1
    if (fabsf(x) > 65504.f) return x > 0 ? x : 0.f;
#endif
    return activation_round(__fmul_rn(activation_round(x), gate));
#endif
}
