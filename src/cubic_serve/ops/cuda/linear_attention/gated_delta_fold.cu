// NInfer ReplaySSM accepted-prefix fold (Apache-2.0), b9a6b5c,
// src/ops/linear_attention/gated_delta_net/recurrent.cuh:FoldAccess.
// Geometry, layer count and source/destination maps remain explicit operands.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "gated_delta_transition.cuh"
#if IS_BF16
using scalar=__nv_bfloat16;
#else
using scalar=__half;
#endif
#if STATE_BF16
using state_scalar=__nv_bfloat16;
#else
using state_scalar=float;
#endif
constexpr int Keys=(K_DIM+31)/32,Tiles=(V_DIM+15)/16;
#if STATE_MMA
#include "gated_delta_mma_fold.cuh"
#else
extern "C" __launch_bounds__(128) __global__ void gated_delta_fold(
    const scalar* __restrict__ keys,const scalar* __restrict__ values,
    const float* __restrict__ coefficients,state_scalar* __restrict__ state,
    const int* __restrict__ sources,const int* __restrict__ destinations,
    const int* __restrict__ segments,const int* __restrict__ activity,
    const int* __restrict__ request_rows,const long long* __restrict__ decisions
#if FOLD_CONV
    , const scalar* __restrict__ conv_record,scalar* __restrict__ conv_state,
    const int* __restrict__ conv_sources,const int* __restrict__ conv_destinations
#endif
    ) {
    const int start=blockIdx.y,head=blockIdx.x,layer=blockIdx.z/Tiles;
    const int vbase=(blockIdx.z%Tiles)*16+(threadIdx.x>>5)*4,lane=threadIdx.x&31;
    const int length=min(max(segments[start],0),ROWS-start);
    if(!length||!activity[start])return;
    const int request=request_rows[start];
    if(request<0||request>=REQUESTS)return;
    const long long accepted=decisions[request*2];
    if(accepted<0||accepted>=length)return;
    const int keyhead=head/(V_HEADS/Q_HEADS);
    const long long stride=static_cast<long long>(V_HEADS)*V_DIM*K_DIM;
    const long long base=static_cast<long long>(head)*V_DIM*K_DIM;
    float history[4][Keys];
    const int source=sources[layer*ROWS+start];
    // The original target map's segment endpoint owns durable state. A
    // rejected prefix never publishes into its temporary checkpoint slots.
    const int destination=destinations[layer*ROWS+start+length-1];
    #pragma unroll
    for(int r=0;r<4;++r) {
        #pragma unroll
        for(int c=0;c<Keys;++c) {
            const int i=lane*Keys+c;
            history[r][c]=vbase+r<V_DIM&&i<K_DIM
                ?float(state[static_cast<long long>(source)*stride+base+(vbase+r)*K_DIM+i]):0.f;
        }
    }
    for(int token=0;token<=accepted;++token) {
        const int row=start+token;
        if(!activity[row])return;
        float key[Keys],value[4];
        #pragma unroll
        for(int c=0;c<Keys;++c) {
            const int i=lane*Keys+c;
            key[c]=i<K_DIM?float(keys[((static_cast<long long>(layer)*ROWS+row)*Q_HEADS+keyhead)*K_DIM+i]):0.f;
        }
        #pragma unroll
        for(int r=0;r<4;++r)value[r]=vbase+r<V_DIM
            ?float(values[((static_cast<long long>(layer)*ROWS+row)*V_HEADS+head)*V_DIM+vbase+r]):0.f;
        const long long control=((static_cast<long long>(layer)*ROWS+row)*V_HEADS+head)*2;
        apply_gdn_transition<scalar,Keys,NATIVE,STATE_BF16>(history,key,value,
            coefficients[control],coefficients[control+1]);
    }
    #pragma unroll
    for(int r=0;r<4;++r) {
        #pragma unroll
        for(int c=0;c<Keys;++c) {
            const int i=lane*Keys+c;
            if(vbase+r<V_DIM&&i<K_DIM)
                state[static_cast<long long>(destination)*stride+base+(vbase+r)*K_DIM+i]
                    =state_scalar(history[r][c]);
        }
    }
#if FOLD_CONV
    // NInfer publishes convolution history in the accepted-prefix fold, not
    // for every speculative candidate. Each channel has one explicit owner.
    const int tile=blockIdx.z%Tiles;
    for(int channel=(head*Tiles+tile)*128+threadIdx.x;channel<CONV_CHANNELS;
            channel+=V_HEADS*Tiles*128) {
        scalar final_history[CONV_WIDTH-1];
        #pragma unroll
        for(int i=0;i<CONV_WIDTH-1;++i) {
            const int position=int(accepted)+1-(CONV_WIDTH-1)+i;
            final_history[i]=position>=0
                ?conv_record[(static_cast<long long>(layer)*ROWS+start+position)*CONV_CHANNELS+channel]
                :conv_state[(static_cast<long long>(conv_sources[layer*ROWS+start])*CONV_CHANNELS+channel)*(CONV_WIDTH-1)+(CONV_WIDTH-1)+position];
        }
        const int dest=conv_destinations[layer*ROWS+start+length-1];
        #pragma unroll
        for(int i=0;i<CONV_WIDTH-1;++i)
            conv_state[(static_cast<long long>(dest)*CONV_CHANNELS+channel)*(CONV_WIDTH-1)+i]=final_history[i];
    }
#endif
}
#endif
