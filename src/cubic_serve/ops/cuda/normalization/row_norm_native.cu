// NInfer (Apache-2.0), b9a6b5c, src/ops/kernel/rmsnorm.cuh:
// pair-vector loads, register-resident row, warp/CTA reduction and epilogue.
// Parameterized row width and FP16/BF16 storage; Cubic's rounding remains.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "activation_native.cuh"
#if IS_BF16
using scalar=__nv_bfloat16;
#else
using scalar=__half;
#endif
#if WEIGHT_FP32
using weight_scalar=float;
#else
using weight_scalar=scalar;
#endif
constexpr int Lanes=WARP_ROWS?32:256;
constexpr int Chunks=(WIDTH+Lanes*2-1)/(Lanes*2);
__device__ __forceinline__ float stored(float x){return float(scalar(x));}
__device__ __forceinline__ float2 load_pair(const scalar* pointer,int column) {
    if constexpr (WIDTH%2==0) {
        const unsigned bits=*reinterpret_cast<const unsigned*>(pointer+column);
#if IS_BF16
        union {unsigned u;__nv_bfloat162 h;} value;value.u=bits;
        return __bfloat1622float2(value.h);
#else
        union {unsigned u;__half2 h;} value;value.u=bits;
        return __half22float2(value.h);
#endif
    } else return make_float2(float(pointer[column]),column+1<WIDTH?float(pointer[column+1]):0.f);
}
__device__ __forceinline__ float2 load_weight(const weight_scalar* pointer,int column) {
#if WEIGHT_FP32
    return make_float2(pointer[column],column+1<WIDTH?pointer[column+1]:0.f);
#else
    return load_pair(pointer,column);
#endif
}
extern "C" __launch_bounds__(256) __global__ void row_norm_native(
    const scalar* __restrict__ x,const weight_scalar* __restrict__ weight,
    const scalar* __restrict__ gate,scalar* __restrict__ output,float epsilon) {
    const int lane=threadIdx.x&31,warp=threadIdx.x>>5;
    const int row=WARP_ROWS?blockIdx.x*8+warp:blockIdx.x;
    if(row>=ROWS)return;
    const int owner=WARP_ROWS?lane:threadIdx.x;
    float2 values[Chunks],weights[Chunks],gates[Chunks];float sum=0.f;
    #pragma unroll
    for(int c=0;c<Chunks;++c) {
        const int column=(owner+c*Lanes)*2;
        values[c]=column<WIDTH?load_pair(x+static_cast<long long>(row)*WIDTH,column):make_float2(0.f,0.f);
        weights[c]=column<WIDTH?load_weight(weight,column):make_float2(0.f,0.f);
#if EPILOGUE == 2
        gates[c]=column<WIDTH?load_pair(gate+static_cast<long long>(row)*WIDTH,column):make_float2(0.f,0.f);
#endif
        sum=__fadd_rn(sum,__fadd_rn(__fmul_rn(values[c].x,values[c].x),__fmul_rn(values[c].y,values[c].y)));
    }
    #pragma unroll
    for(int shift=16;shift>0;shift>>=1)sum=__fadd_rn(sum,__shfl_down_sync(0xffffffff,sum,shift));
    float inverse;
#if WARP_ROWS
    inverse=lane==0?rsqrtf(__fadd_rn(__fdiv_rn(sum,float(WIDTH)),epsilon)):0.f;
    inverse=__shfl_sync(0xffffffff,inverse,0);
#else
    __shared__ float partial[8],shared_inverse;
    if(lane==0)partial[warp]=sum;
    __syncthreads();
    if(threadIdx.x==0) {
        float total=0.f;
        #pragma unroll
        for(int i=0;i<8;++i)total=__fadd_rn(total,partial[i]);
        shared_inverse=__frsqrt_rn(__fadd_rn(__fdiv_rn(total,float(WIDTH)),epsilon));
    }
    __syncthreads();inverse=shared_inverse;
#endif
    #pragma unroll
    for(int c=0;c<Chunks;++c) {
        const int column=(owner+c*Lanes)*2;
        if(column>=WIDTH)continue;
        #pragma unroll
        for(int element=0;element<2;++element) {
        if(column+element>=WIDTH)continue;
        float value=__fmul_rn(element?values[c].y:values[c].x,inverse),w=element?weights[c].y:weights[c].x;
#if EPILOGUE == 0
        w=__fadd_rn(w,1.f);
#elif EPILOGUE == 2
        value=stored(value);
#endif
        value=__fmul_rn(value,w);
#if EPILOGUE == 2
#if !WEIGHT_FP32
        value=stored(value);
#endif
        const float g=element?gates[c].y:gates[c].x;
        const float activated=SIGMOID_GATE?activation_sigmoid_native(g):activation_silu_native(g);
        value=__fmul_rn(value,activated);
#endif
        output[static_cast<long long>(row)*WIDTH+column+element]=scalar(value);
        }
    }
}
