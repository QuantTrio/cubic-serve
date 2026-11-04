// Direct split-four execution adapted from T-Crypt/ninfer-4090 (Apache-2.0),
// b9a6b5c, src/ops/linear/q5/q5_rowsplit_gemm_simt.cuh:284-416.
// Preserve its cyclic slab ownership, direct register operands, vector input
// loads, warp reduction and ascending split publication. The decoder owns
// checkpoint arithmetic, including decoded-weight storage rounding.
#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <numeric>

template<int RawBytes>
__device__ __forceinline__ void packed_group_copy(std::uint8_t* target,
    const std::uint8_t* source, int column, int tile, int lane) {
    if (lane == 0)
        for (int i = 0; i < RawBytes; ++i)
            target[i] = source[(std::int64_t(column) * (INPUT_K/GI) + tile)*RawBytes+i];
}
#include "decoder.cuh"
#if FUSE_CONV
#include "activation_native.cuh"
#endif
#ifndef RAW_OFFSET
#define RAW_OFFSET 0
#endif
#if IS_BF16
using scalar = __nv_bfloat16;
#else
using scalar = __half;
#endif

__device__ __forceinline__ float2 input_pair(unsigned word) {
#if IS_BF16
    union { unsigned u; __nv_bfloat162 x; } v; v.u=word;
    return __bfloat1622float2(v.x);
#else
    union { unsigned u; __half2 x; } v; v.u=word;
    return __half22float2(v.x);
#endif
}

__device__ __forceinline__ unsigned long long fragment(const std::uint8_t* row, int offset) {
    const int bit=offset*BITS, half=bit/16, shift=bit%16;
    const auto* words=reinterpret_cast<const unsigned short*>(row);
    unsigned long long value=words[half];
    #pragma unroll
    for (int i=1;i<(ELEMENTS*BITS+15)/16+1;++i)
        if (16*i<ELEMENTS*BITS+shift)
            value|=static_cast<unsigned long long>(words[half+i])<<(16*i);
    return value>>shift;
}

// NInfer's ten-block residency hint assumes its inexpensive Q5 decoder.
// Original format arithmetic may need more registers. Keep identical work
// ownership without forcing checkpoint decode operands into local memory.
extern "C" __launch_bounds__(128) __global__ void packed_direct_rows(
    const scalar* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ metadata, scalar* __restrict__ output
#if JOINED
    , const std::uint8_t* __restrict__ other_codes,
    const std::uint8_t* __restrict__ other_metadata,scalar* __restrict__ other_output
#endif
#if FUSE_CONV
    , const scalar* __restrict__ conv_weight, scalar* __restrict__ state,
    const int* __restrict__ sources, const int* __restrict__ destinations,
    const int* __restrict__ segments, const int* __restrict__ activity,
    scalar* __restrict__ convolved
#endif
    ) {
    __shared__ float partial[4][ROWS];
    const int lane=threadIdx.x&31, split=threadIdx.x>>5;
    bool other=false;int column=blockIdx.x,output_stride=OUTPUT_N,raw_offset=RAW_OFFSET;
#if JOINED
    other=column>=OUTPUT_N;
    if(other) {
        column-=OUTPUT_N;codes=other_codes;metadata=other_metadata;
        output=other_output;output_stride=OTHER_N;raw_offset=OTHER_RAW_OFFSET;
    }
#endif
    auto context=format_context(metadata,column);
#if JOINED
    if(other)context=format_rebind_context(context,OTHER_ZERO_OFFSET);
#endif
    const auto* row=codes+std::int64_t(column)*(INPUT_K*BITS/8);
    float acc[ROWS]={};
    // Four warps own consecutive256-value chunks of each1024-value slab.
    // Tail ownership has no model/device dependency and does not add graph rows.
    // Cubic's multi-level curve decode has a longer live range than Q5's
    // multiply-only decoder. Bound its lifetime to one upstream slab.
    #pragma unroll 1
    for (int slab=0;slab<(INPUT_K+1023)/1024;++slab) {
        const int offset=slab*1024+split*256+lane*ELEMENTS;
        const bool live=offset<INPUT_K;
        const int group=live?offset/GI:0;
        const auto* raw=metadata+raw_offset+
            (std::int64_t(column)*(INPUT_K/GI)+group)*FORMAT_RAW_BYTES;
        float2 data;
        // TILE_K=GI: the format sees the actual absolute checkpoint group.
        format_expand(&data,raw,metadata,context,column,group,0);
        const auto levels=format_levels(data,lane);
        const auto packed=live?fragment(row,offset):0;
        float weights[ELEMENTS];
        #pragma unroll
        for (int e=0;e<ELEMENTS;++e)
            weights[e]=format_decode((packed>>(e*BITS))&((1u<<BITS)-1),levels,data,lane);
        #pragma unroll
        for (int token=0;token<ROWS;++token) {
            const uint4 xv=live?*reinterpret_cast<const uint4*>(x+token*INPUT_K+offset)
                :make_uint4(0,0,0,0);
            const float2 f0=input_pair(xv.x),f1=input_pair(xv.y);
            const float2 f2=input_pair(xv.z),f3=input_pair(xv.w);
            acc[token]=__fmaf_rn(weights[0],f0.x,acc[token]);
            acc[token]=__fmaf_rn(weights[1],f0.y,acc[token]);
            acc[token]=__fmaf_rn(weights[2],f1.x,acc[token]);
            acc[token]=__fmaf_rn(weights[3],f1.y,acc[token]);
            acc[token]=__fmaf_rn(weights[4],f2.x,acc[token]);
            acc[token]=__fmaf_rn(weights[5],f2.y,acc[token]);
            acc[token]=__fmaf_rn(weights[6],f3.x,acc[token]);
            acc[token]=__fmaf_rn(weights[7],f3.y,acc[token]);
        }
    }
    #pragma unroll
    for (int token=0;token<ROWS;++token) {
        float v=acc[token];
        #pragma unroll
        for(int delta=16;delta>0;delta>>=1)v=__fadd_rn(v,__shfl_down_sync(0xffffffff,v,delta));
        if(lane==0)partial[split][token]=v;
    }
    __syncthreads();
    if(split==0&&lane<ROWS) {
        float value=0;
        #pragma unroll
        for(int p=0;p<4;++p)value=__fadd_rn(value,partial[p][lane]);
        const scalar rounded = scalar(value);
        output[lane*output_stride+column]=rounded;
#if FUSE_CONV
        // Preserve Cubic's projection storage boundary before convolution and
        // before current/next-step history consumption. NInfer consumes the
        // FP32 accumulator here; its organization does not authorize that math.
        partial[0][lane]=float(rounded);
        if (!other&&!activity[lane]) convolved[lane*OUTPUT_N+column]=scalar(0.f);
#endif
    }
#if FUSE_CONV
    __syncthreads();
    if (!other&&split==0&&lane==0) {
        // NInfer GdnConvEpilogue: one output-channel owner keeps segment
        // history in registers and publishes every candidate checkpoint.
        // Cubic uses channel-major history and explicit logical activity.
        for(int start=0;start<ROWS;++start) {
            const int length=min(max(segments[start],0),ROWS-start);
            if(!length||!activity[start]) continue;
            float history[CONV_WIDTH-1];
            const long long source=(static_cast<long long>(sources[start])*OUTPUT_N+column)*(CONV_WIDTH-1);
            #pragma unroll
            for(int t=0;t<CONV_WIDTH-1;++t) history[t]=float(state[source+t]);
            for(int token=0;token<length;++token) {
                const int row=start+token;
                if(!activity[row]) continue;
                const float current=partial[0][row];
                float result=0.f;
                #pragma unroll
                for(int t=0;t<CONV_WIDTH-1;++t)
                    result=__fadd_rn(result,__fmul_rn(history[t],float(conv_weight[column*CONV_WIDTH+t])));
                result=__fadd_rn(result,__fmul_rn(current,float(conv_weight[column*CONV_WIDTH+CONV_WIDTH-1])));
                convolved[row*OUTPUT_N+column]=scalar(activation_silu_native(result));
                #pragma unroll
                for(int t=0;t<CONV_WIDTH-2;++t) history[t]=history[t+1];
                history[CONV_WIDTH-2]=current;
                const long long destination=(static_cast<long long>(destinations[row])*OUTPUT_N+column)*(CONV_WIDTH-1);
                #pragma unroll
                for(int t=0;t<CONV_WIDTH-1;++t) state[destination+t]=scalar(history[t]);
            }
        }
    }
#endif
}
