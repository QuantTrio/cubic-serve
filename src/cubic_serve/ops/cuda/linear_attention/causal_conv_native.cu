// NInfer gdn_projected_conv_kernel / NoHistoryPublish organization,
// Apache-2.0, b9a6b5c. Original projection, activation and history rounding
// remain mathematical boundaries; row activity is separate from segmentation.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "activation_native.cuh"
#if IS_BF16
using scalar=__nv_bfloat16;
#else
using scalar=__half;
#endif
extern "C" __global__ void causal_conv_native(
    const scalar* __restrict__ projected,const scalar* __restrict__ values,
    const scalar* __restrict__ weight,scalar* __restrict__ state,
    const int* __restrict__ sources,const int* __restrict__ destinations,
    const int* __restrict__ segments,const int* __restrict__ activity,
    scalar* __restrict__ output,scalar* __restrict__ record) {
    const int channel=blockIdx.x*128+threadIdx.x,start=blockIdx.y;
    if(channel>=CHANNELS)return;
    if(!activity[start]){output[static_cast<long long>(start)*CHANNELS+channel]=scalar(0.f);return;}
    const int length=min(max(segments[start],0),ROWS-start);
    if(!length||!activity[start])return;
    float history[WIDTH-1],weights[WIDTH];
    #pragma unroll
    for(int i=0;i<WIDTH-1;++i)
        history[i]=float(state[(static_cast<long long>(sources[start])*CHANNELS+channel)*(WIDTH-1)+i]);
    #pragma unroll
    for(int i=0;i<WIDTH;++i)weights[i]=float(weight[channel*WIDTH+i]);
    for(int token=0;token<length;++token) {
        const int row=start+token;if(!activity[row])continue;
        const scalar raw=channel<PROJECTED_CHANNELS
            ?projected[static_cast<long long>(row)*PROJECTED_CHANNELS+channel]
            :values[static_cast<long long>(row)*(CHANNELS-PROJECTED_CHANNELS)+channel-PROJECTED_CHANNELS];
        const float current=float(raw);float sum=0.f;
        #pragma unroll
        for(int i=0;i<WIDTH-1;++i)sum=__fadd_rn(sum,__fmul_rn(history[i],weights[i]));
        sum=__fadd_rn(sum,__fmul_rn(current,weights[WIDTH-1]));
        output[static_cast<long long>(row)*CHANNELS+channel]=scalar(activation_silu_native(sum));
        #pragma unroll
        for(int i=0;i<WIDTH-2;++i)history[i]=history[i+1];
        history[WIDTH-2]=current;
#if RECORD_ONLY
        record[static_cast<long long>(row)*CHANNELS+channel]=raw;
#else
        #pragma unroll
        for(int i=0;i<WIDTH-1;++i)
            state[(static_cast<long long>(destinations[row])*CHANNELS+channel)*(WIDTH-1)+i]=scalar(history[i]);
#endif
    }
}
