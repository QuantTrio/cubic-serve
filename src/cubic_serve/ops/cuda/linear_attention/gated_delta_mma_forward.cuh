#include "gated_delta_mma.cuh"
extern "C" __launch_bounds__(STATE_WARPS*32) __global__ void gated_delta_native(
    const scalar* __restrict__ packed,const scalar* __restrict__ normalized,
    const scalar* __restrict__ a,const scalar* __restrict__ b,
    const control_scalar* __restrict__ alog,const scalar* __restrict__ bias,
    state_scalar* __restrict__ state,const int* __restrict__ sources,
    const int* __restrict__ destinations,const int* __restrict__ segments,
    const int* __restrict__ activity,scalar* __restrict__ output,
    const float* __restrict__ coefficients,float scale,float epsilon
#if RECORD_ONLY
    ,scalar* __restrict__ key_record,scalar* __restrict__ value_record,
    float* __restrict__ coefficient_record
#endif
    ) {
    const int head=blockIdx.x,start=blockIdx.y,lane=threadIdx.x&31,warp=threadIdx.x>>5;
    const int length=min(max(segments[start],0),ROWS-start);
    if(!length||!activity[start])return;
    const int keyhead=head/(V_HEADS/Q_HEADS),vbase=(blockIdx.z*STATE_WARPS+warp)*16;
    const int group=lane/4,part=lane%4;
    const long long stride=STATE_SLOT_STRIDE;
    const long long base=static_cast<long long>(head)*V_DIM*K_DIM;
    StatePair history[StateKTiles][4];
    load_state_pairs(history,state,static_cast<long long>(sources[start])*stride+base,vbase,lane);
    for(int token=0;token<length;++token) {
        const int row=start+token;if(!activity[row])continue;
        const long long qbase=static_cast<long long>(row)*PACKED_DIM+keyhead*K_DIM;
        unsigned key[StateKTiles][2],query[StateKTiles][2];
        load_vector_pairs(key,normalized+qbase+Q_HEADS*K_DIM,lane,0,false);
        load_vector_pairs(query,normalized+qbase,lane,act_pack(scale,scale),true);
        const long long control=static_cast<long long>(row)*V_HEADS+head;
        float beta,decay;
#if PRECOMPUTED
        beta=coefficients[control*2];decay=coefficients[control*2+1];
#else
        beta=float(scalar(activation_sigmoid_native(float(b[control]))));
        const float alpha=__fadd_rn(float(a[control]),float(bias[head]));
        const float softplus=alpha<=20.f?log1pf(expf(alpha)):alpha;
        decay=expf(__fmul_rn(-expf(float(alog[head])),softplus));
#endif
        const long long valuebase=static_cast<long long>(row)*PACKED_DIM+2*Q_HEADS*K_DIM+head*V_DIM;
        const float v0=vbase+group<V_DIM?float(packed[valuebase+vbase+group]):0.f;
        const float v1=vbase+group+8<V_DIM?float(packed[valuebase+vbase+group+8]):0.f;
#if RECORD_ONLY
        if(blockIdx.z==0&&warp==0&&head%(V_HEADS/Q_HEADS)==0)
            for(int i=lane;i<K_DIM;i+=32)
                key_record[(static_cast<long long>(row)*Q_HEADS+keyhead)*K_DIM+i]
                    =normalized[qbase+Q_HEADS*K_DIM+i];
        if(part==0) {
            if(vbase+group<V_DIM)value_record[(static_cast<long long>(row)*V_HEADS+head)*V_DIM+vbase+group]=scalar(v0);
            if(vbase+group+8<V_DIM)value_record[(static_cast<long long>(row)*V_HEADS+head)*V_DIM+vbase+group+8]=scalar(v1);
        }
        if(blockIdx.z==0&&warp==0&&lane==0) {
            coefficient_record[control*2]=beta;coefficient_record[control*2+1]=decay;
        }
#endif
        apply_gdn_mma_transition(history,key,act_pack(v0,v1),beta,decay,lane);
        float result[4]={};gdn_mma_readout(result,history,query);
        if(part==0) {
            const long long offset=(static_cast<long long>(row)*V_HEADS+head)*V_DIM+vbase+group;
            if(vbase+group<V_DIM)output[offset]=scalar(result[0]);
            if(vbase+group+8<V_DIM)output[offset+8]=scalar(result[2]);
        }
#if !RECORD_ONLY
        store_state_pairs(history,state,static_cast<long long>(destinations[row])*stride+base,vbase,lane);
#endif
    }
}
