// Same producing A/B storage operands and gate rule as _gdn_coefficients.
// Generate once per live row/head, rather than once per recurrent value tile.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "activation_native.cuh"
#if IS_BF16
using scalar=__nv_bfloat16;
#else
using scalar=__half;
#endif
#if ALOG_FP32
using log_scalar=float;
#else
using log_scalar=scalar;
#endif
extern "C" __global__ void gated_delta_coefficients_native(
    const scalar* __restrict__ a,const scalar* __restrict__ b,
    const log_scalar* __restrict__ alog,const scalar* __restrict__ bias,
    const int* __restrict__ activity,float* __restrict__ output) {
    const int item=blockIdx.x*128+threadIdx.x;if(item>=ROWS*HEADS)return;
    if(!activity[item/HEADS])return;
    const int head=item%HEADS;
    const float beta=float(scalar(activation_sigmoid_native(float(b[item]))));
    const float alpha=__fadd_rn(float(a[item]),float(bias[head]));
    const float softplus=alpha<=20.f?log1pf(expf(alpha)):alpha;
    output[item*2]=beta;
    output[item*2+1]=expf(__fmul_rn(-expf(float(alog[head])),softplus));
}
