// Adapted from NInfer (Apache-2.0), b9a6b5c,
// src/ops/gdn_gating_proj/bf16/bf16_gdn_norm_gating_proj_27.cu.
// Each control head owns both dots and a disjoint normalized-input stripe.
// Cubic's rounded normalized operands require a norm reduction before the
// dots, rather than NInfer's post-dot scaling of unrounded operands.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#if WITH_GATES
#include "activation_native.cuh"
#endif
#if IS_BF16
using scalar=__nv_bfloat16;
#else
using scalar=__half;
#endif
constexpr int Warps=16;
__device__ __forceinline__ float reduce(float x) {
    #pragma unroll
    for(int shift=16;shift>0;shift>>=1)x=__fadd_rn(x,__shfl_down_sync(0xffffffff,x,shift));
    return x;
}
extern "C" __launch_bounds__(512) __global__ void norm_control_native(
    const scalar* __restrict__ x,const scalar* __restrict__ nw,
    const scalar* __restrict__ aw,const scalar* __restrict__ bw,
    scalar* __restrict__ h,scalar* __restrict__ a,scalar* __restrict__ b,float epsilon
#if WITH_GATES
    , const ALOG_TYPE* __restrict__ alog,const scalar* __restrict__ bias,
    float* __restrict__ coefficients
#endif
    ) {
    const int tid=threadIdx.x,lane=tid&31,warp=tid>>5,head=blockIdx.x;
    const int first=blockIdx.y*TILE;
    __shared__ float partial[3][TILE][Warps],inverse[TILE];
    float sums[TILE]={},aa[TILE]={},bb[TILE]={};
    for(int base=tid*8;base<WIDTH;base+=512*8) {
        #pragma unroll
        for(int t=0;t<TILE;++t) {
            if(first+t>=ROWS)continue;
            #pragma unroll
            for(int p=0;p<8;++p) {
                if(base+p>=WIDTH)continue;
                const float v=float(x[static_cast<long long>(first+t)*WIDTH+base+p]);
                sums[t]=__fadd_rn(sums[t],__fmul_rn(v,v));
            }
        }
    }
    #pragma unroll
    for(int t=0;t<TILE;++t) {
        sums[t]=reduce(sums[t]);if(lane==0)partial[0][t][warp]=sums[t];
    }
    __syncthreads();
    if(tid<TILE) {
        float total=0.f;
        #pragma unroll
        for(int w=0;w<Warps;++w)total=__fadd_rn(total,partial[0][tid][w]);
        inverse[tid]=__frsqrt_rn(__fadd_rn(__fdiv_rn(total,float(WIDTH)),epsilon));
    }
    __syncthreads();
    for(int base=tid*8;base<WIDTH;base+=512*8) {
        #pragma unroll
        for(int p=0;p<8;++p) {
            if(base+p>=WIDTH)continue;
            const float n=__fadd_rn(1.f,float(nw[base+p]));
            const float wa=float(aw[static_cast<long long>(head)*WIDTH+base+p]);
            const float wb=float(bw[static_cast<long long>(head)*WIDTH+base+p]);
            #pragma unroll
            for(int t=0;t<TILE;++t) {
                if(first+t>=ROWS)continue;
                const float v=float(x[static_cast<long long>(first+t)*WIDTH+base+p]);
                const float normalized=float(scalar(__fmul_rn(__fmul_rn(v,inverse[t]),n)));
                aa[t]=__fmaf_rn(wa,normalized,aa[t]);bb[t]=__fmaf_rn(wb,normalized,bb[t]);
            }
        }
    }
    #pragma unroll
    for(int t=0;t<TILE;++t) {
        aa[t]=reduce(aa[t]);bb[t]=reduce(bb[t]);
        if(lane==0){partial[1][t][warp]=aa[t];partial[2][t][warp]=bb[t];}
    }
    __syncthreads();
    if(tid<TILE&&first+tid<ROWS) {
        float va=0.f,vb=0.f;
        #pragma unroll
        for(int w=0;w<Warps;++w){va=__fadd_rn(va,partial[1][tid][w]);vb=__fadd_rn(vb,partial[2][tid][w]);}
        a[(first+tid)*HEADS+head]=scalar(va);b[(first+tid)*HEADS+head]=scalar(vb);
#if WITH_GATES
        // Keep the original A/B storage boundary even though no reload is
        // needed. NInfer produces these controls once, before recurrent tiles.
        const long long index=static_cast<long long>(first+tid)*HEADS+head;
        const float alpha=__fadd_rn(float(scalar(va)),float(bias[head]));
        const float softplus=alpha<=20.f?log1pf(expf(alpha)):alpha;
        coefficients[index*2]=float(scalar(activation_sigmoid_native(float(scalar(vb)))));
        coefficients[index*2+1]=expf(__fmul_rn(-expf(float(alog[head])),softplus));
#endif
    }
    constexpr int ColumnsPerHead=(WIDTH+HEADS-1)/HEADS;
    for(int i=tid;i<ColumnsPerHead;i+=512) {
        const int column=head*ColumnsPerHead+i;if(column>=WIDTH)continue;
        const float n=__fadd_rn(1.f,float(nw[column]));
        #pragma unroll
        for(int t=0;t<TILE;++t)if(first+t<ROWS) {
            const long long offset=static_cast<long long>(first+t)*WIDTH+column;
            h[offset]=scalar(__fmul_rn(__fmul_rn(float(x[offset]),inverse[t]),n));
        }
    }
}
