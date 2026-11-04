// Shared native GDN transition. MMA operand layout follows the existing
// SM80 m16n8k16 row/column contract used by Cubic's shared small-M operator.
// Original low operand arithmetic, explicit FP32 RN addition and state
// storage rounding are preserved in both forward and accepted-prefix fold.
#pragma once
#include <type_traits>
constexpr int StateKTiles=(K_DIM+15)/16;
using StatePair=std::conditional_t<STATE_BF16,unsigned,float2>;

__device__ __forceinline__ unsigned act_pack(float a,float b) {
#if IS_BF16
    return unsigned(__bfloat16_as_ushort(__float2bfloat16_rn(a))) |
        (unsigned(__bfloat16_as_ushort(__float2bfloat16_rn(b)))<<16);
#else
    return unsigned(__half_as_ushort(__float2half_rn(a))) |
        (unsigned(__half_as_ushort(__float2half_rn(b)))<<16);
#endif
}
__device__ __forceinline__ float2 act_unpack(unsigned pair) {
#if IS_BF16
    union {unsigned bits;__nv_bfloat162 pair;} v;v.bits=pair;
    return __bfloat1622float2(v.pair);
#else
    union {unsigned bits;__half2 pair;} v;v.bits=pair;
    return __half22float2(v.pair);
#endif
}
__device__ __forceinline__ float2 state_unpack(StatePair pair) {
#if STATE_BF16
    union {unsigned bits;__nv_bfloat162 pair;} v;v.bits=pair;
    return __bfloat1622float2(v.pair);
#else
    return pair;
#endif
}
__device__ __forceinline__ StatePair state_pack(float a,float b) {
#if STATE_BF16
    return unsigned(__bfloat16_as_ushort(__float2bfloat16_rn(a))) |
        (unsigned(__bfloat16_as_ushort(__float2bfloat16_rn(b)))<<16);
#else
    return make_float2(a,b);
#endif
}
__device__ __forceinline__ unsigned state_as_act(StatePair pair) {
#if STATE_BF16 && IS_BF16
    return pair;
#else
    const float2 v=state_unpack(pair);return act_pack(v.x,v.y);
#endif
}
__device__ __forceinline__ unsigned low_fma_pair(unsigned a,unsigned b,unsigned c) {
    unsigned result;
#if IS_BF16
    asm("fma.rn.bf16x2 %0,%1,%2,%3;":"=r"(result):"r"(a),"r"(b),"r"(c));
#else
    asm("fma.rn.f16x2 %0,%1,%2,%3;":"=r"(result):"r"(a),"r"(b),"r"(c));
#endif
    return result;
}
__device__ __forceinline__ unsigned low_mul_pair(unsigned a,unsigned b) {
    return low_fma_pair(a,b,0x80008000u);
}
__device__ __forceinline__ void gdn_mma(float (&c)[4],unsigned a0,unsigned a1,
        unsigned a2,unsigned a3,unsigned b0,unsigned b1) {
#if IS_BF16
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
#else
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
#endif
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
        :"=f"(c[0]),"=f"(c[1]),"=f"(c[2]),"=f"(c[3])
        :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
         "f"(c[0]),"f"(c[1]),"f"(c[2]),"f"(c[3]));
}

__device__ __forceinline__ void load_state_pairs(StatePair (&history)[StateKTiles][4],
        const state_scalar* state,long long base,int vbase,int lane) {
    const int group=lane/4,part=lane%4;
    #pragma unroll
    for(int tile=0;tile<StateKTiles;++tile) {
        #pragma unroll
        for(int i=0;i<4;++i) {
            const int row=vbase+group+(i%2)*8,key=tile*16+part*2+(i/2)*8;
            const long long offset=base+static_cast<long long>(row)*K_DIM+key;
            const float a=row<V_DIM&&key<K_DIM?float(state[offset]):0.f;
            const float b=row<V_DIM&&key+1<K_DIM?float(state[offset+1]):0.f;
            history[tile][i]=state_pack(a,b);
        }
    }
}
__device__ __forceinline__ void store_state_pairs(const StatePair (&history)[StateKTiles][4],
        state_scalar* state,long long base,int vbase,int lane) {
    const int group=lane/4,part=lane%4;
    #pragma unroll
    for(int tile=0;tile<StateKTiles;++tile) {
        #pragma unroll
        for(int i=0;i<4;++i) {
            const int row=vbase+group+(i%2)*8,key=tile*16+part*2+(i/2)*8;
            const long long offset=base+static_cast<long long>(row)*K_DIM+key;
            const float2 v=state_unpack(history[tile][i]);
            if(row<V_DIM&&key<K_DIM)state[offset]=state_scalar(v.x);
            if(row<V_DIM&&key+1<K_DIM)state[offset+1]=state_scalar(v.y);
        }
    }
}
__device__ __forceinline__ void load_vector_pairs(unsigned (&vector)[StateKTiles][2],
        const scalar* input,int lane,unsigned scale,bool multiply) {
    const int group=lane/4,part=lane%4;
    #pragma unroll
    for(int tile=0;tile<StateKTiles;++tile) {
        #pragma unroll
        for(int i=0;i<2;++i) {
            const int key=tile*16+part*2+i*8;
            const float a=group==0&&key<K_DIM?float(input[key]):0.f;
            const float b=group==0&&key+1<K_DIM?float(input[key+1]):0.f;
            const unsigned pair=act_pack(a,b);
            vector[tile][i]=multiply?low_mul_pair(pair,scale):pair;
        }
    }
}
__device__ __forceinline__ void apply_gdn_mma_transition(
        StatePair (&history)[StateKTiles][4],const unsigned (&key)[StateKTiles][2],
        unsigned values,float beta,float decay,int lane) {
    unsigned decayed[StateKTiles][4];float memory[4]={};
    const unsigned d=act_pack(decay,decay);
    #pragma unroll
    for(int tile=0;tile<StateKTiles;++tile) {
        #pragma unroll
        for(int i=0;i<4;++i)decayed[tile][i]=low_mul_pair(state_as_act(history[tile][i]),d);
        gdn_mma(memory,decayed[tile][0],decayed[tile][1],decayed[tile][2],decayed[tile][3],key[tile][0],key[tile][1]);
    }
    const int group=lane/4,part=lane%4;
    const unsigned gathered=act_pack(__shfl_sync(0xffffffff,memory[0],group*4),
                                    __shfl_sync(0xffffffff,memory[2],group*4));
    unsigned residual=low_fma_pair(gathered,act_pack(-1.f,-1.f),values);
    residual=low_mul_pair(residual,act_pack(beta,beta));
    const unsigned a0=part==0?residual&65535u:0,a1=part==0?residual>>16:0;
    #pragma unroll
    for(int tile=0;tile<StateKTiles;++tile) {
        const unsigned k0=__shfl_sync(0xffffffff,key[tile][0],group/2);
        const unsigned k1=__shfl_sync(0xffffffff,key[tile][1],group/2);
        float upper[4]={},lower[4]={};
        const unsigned b0=part==0?(k0>>(16*(group&1)))&65535u:0;
        const unsigned b1=part==0?(k1>>(16*(group&1)))&65535u:0;
        gdn_mma(upper,a0,a1,0,0,b0,0);gdn_mma(lower,a0,a1,0,0,b1,0);
        const float2 d0=act_unpack(decayed[tile][0]),d1=act_unpack(decayed[tile][1]);
        const float2 d2=act_unpack(decayed[tile][2]),d3=act_unpack(decayed[tile][3]);
        history[tile][0]=state_pack(__fadd_rn(d0.x,upper[0]),__fadd_rn(d0.y,upper[1]));
        history[tile][1]=state_pack(__fadd_rn(d1.x,upper[2]),__fadd_rn(d1.y,upper[3]));
        history[tile][2]=state_pack(__fadd_rn(d2.x,lower[0]),__fadd_rn(d2.y,lower[1]));
        history[tile][3]=state_pack(__fadd_rn(d3.x,lower[2]),__fadd_rn(d3.y,lower[3]));
    }
}
__device__ __forceinline__ void gdn_mma_readout(float (&result)[4],
        const StatePair (&history)[StateKTiles][4],const unsigned (&query)[StateKTiles][2]) {
    #pragma unroll
    for(int tile=0;tile<StateKTiles;++tile)
        gdn_mma(result,state_as_act(history[tile][0]),state_as_act(history[tile][1]),
                state_as_act(history[tile][2]),state_as_act(history[tile][3]),query[tile][0],query[tile][1]);
}
