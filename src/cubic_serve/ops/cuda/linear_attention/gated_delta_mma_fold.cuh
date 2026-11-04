#include "gated_delta_mma.cuh"
constexpr int FoldTiles=(V_DIM+STATE_WARPS*16-1)/(STATE_WARPS*16);
extern "C" __launch_bounds__(STATE_WARPS*32) __global__ void gated_delta_fold(
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
    const int head=blockIdx.x,start=blockIdx.y,layer=blockIdx.z/FoldTiles;
    const int tile=blockIdx.z%FoldTiles,lane=threadIdx.x&31,warp=threadIdx.x>>5;
    const int length=min(max(segments[start],0),ROWS-start);
    if(!length||!activity[start])return;
    const int request=request_rows[start];if(request<0||request>=REQUESTS)return;
    const long long accepted=decisions[request*2];if(accepted<0||accepted>=length)return;
    const int keyhead=head/(V_HEADS/Q_HEADS),vbase=(tile*STATE_WARPS+warp)*16;
    const long long stride=static_cast<long long>(V_HEADS)*V_DIM*K_DIM;
    const long long base=static_cast<long long>(head)*V_DIM*K_DIM;
    StatePair history[StateKTiles][4];
    load_state_pairs(history,state,static_cast<long long>(sources[layer*ROWS+start])*stride+base,vbase,lane);
    for(int token=0;token<=accepted;++token) {
        const int row=start+token;if(!activity[row])return;
        const long long outer=static_cast<long long>(layer)*ROWS+row;
        unsigned key[StateKTiles][2];
        load_vector_pairs(key,keys+(outer*Q_HEADS+keyhead)*K_DIM,lane,0,false);
        const int group=lane/4;
        const long long valuebase=(outer*V_HEADS+head)*V_DIM+vbase+group;
        const float v0=vbase+group<V_DIM?float(values[valuebase]):0.f;
        const float v1=vbase+group+8<V_DIM?float(values[valuebase+8]):0.f;
        const long long control=(outer*V_HEADS+head)*2;
        apply_gdn_mma_transition(history,key,act_pack(v0,v1),coefficients[control],coefficients[control+1],lane);
    }
    const int destination=destinations[layer*ROWS+start+length-1];
    store_state_pairs(history,state,static_cast<long long>(destination)*stride+base,vbase,lane);
#if FOLD_CONV
    for(int channel=(head*FoldTiles+tile)*STATE_WARPS*32+threadIdx.x;channel<CONV_CHANNELS;
            channel+=V_HEADS*FoldTiles*STATE_WARPS*32) {
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
