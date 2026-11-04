// Runtime geometry: one baseline compilation unit, independent of model/TP/M.
// E4M3FN is storage. The bounded FP32 encoder avoids the vendor SM80 helper's
// FP64 conversion while preserving its saturating round-to-nearest-even rule.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

__device__ float read_native(uint16_t x, int bf16) {
    if (bf16) { __nv_bfloat16_raw r; r.x=x; return float(__nv_bfloat16(r)); }
    __half_raw r; r.x=x; return float(__half(r));
}
__device__ float scale_for(float maximum) {
    return maximum==0.f ? 1.f : fmaxf(__fdiv_rn(maximum,CACHE_INTEGER?float((1<<(CACHE_BITS-1))-1):448.f),0x1p-126f);
}
__device__ uint8_t encode(float x,float scale) {
#if CACHE_INTEGER
    const float limit=float((1<<(CACHE_BITS-1))-1);
    const float normalized=fminf(fmaxf(__fdiv_rn(x,scale),-limit),limit);
    return uint8_t(__float2int_rn(normalized))&((1<<CACHE_BITS)-1);
#else
    const float normalized=fminf(fmaxf(__fdiv_rn(x,scale),-448.f),448.f);
    const uint32_t bits=__float_as_uint(normalized),magnitude=bits&0x7fffffff;
    const uint32_t rounded=magnitude+0x7ffff+((magnitude>>20)&1);
    const uint32_t code=magnitude<0x3c800000?
        __float2uint_rn(__fmul_rn(__uint_as_float(magnitude),512.f)):
        (rounded>>20)-960;
    return uint8_t(code|((bits>>24)&0x80));
#endif
}
__device__ __forceinline__ uint8_t packed_byte(const uint8_t* codes,int byte,int group) {
    const int first=byte*8/CACHE_BITS,shift=byte*8%CACHE_BITS;
    const unsigned a=codes[first];
    const unsigned b=shift+8>CACHE_BITS && first+1<group?codes[first+1]:0;
    return uint8_t((a>>shift)|(b<<(CACHE_BITS-shift)));
}
__device__ float warp_max(float x) {
    for(int delta=16;delta;delta>>=1)x=fmaxf(x,__shfl_xor_sync(0xffffffff,x,delta));
    return x;
}
__device__ uint64_t divide_index(uint64_t value,uint64_t multiplier,int shift) {
    return (__umul64hi(value,multiplier)+value)>>shift;
}
__device__ int remainder(uint64_t value,int divisor,uint64_t multiplier,int shift) {
    return int(value-divide_index(value,multiplier,shift)*divisor);
}

// action: 0 sealed rows, 1 sealed columns, 2 append rows, 3 append columns.
extern "C" __global__ void cache_quantization_native(
    const uint16_t* incoming, void* payload, float* scales, uint16_t* tail,
    const int64_t* sources,const void* destinations,const int* segments,
    const int* activity,const int64_t* positions,const int* pages,bool* written,
    int action,int bf16,int rows,int layers,int layer,int page,int heads,int dim,
    int group,int quantized,int store_groups,int store_tail,int dest_i64,
    uint64_t gm,int gs,uint64_t pm,int ps,uint64_t bm,int bs,
    int64_t payload_pitch,int64_t scales_pitch,int64_t tail_pitch,int64_t written_pitch) {
    action=CACHE_ACTION;
    bf16=CACHE_BF16;
    quantized=CACHE_QUANTIZED;
    store_groups=CACHE_STORE_GROUPS;
    store_tail=CACHE_STORE_TAIL;
    const int lane=threadIdx.x&31,warp=threadIdx.x>>5;
    const int64_t hd=int64_t(heads)*dim;
    auto* codes=static_cast<uint8_t*>(payload);
    __shared__ float partial[128];
    extern __shared__ uint16_t history[];
    auto* packed=reinterpret_cast<uint8_t*>(history)+(action==3?group*32*2:0);
    // Odd word stride permutes channel lanes over all shared-memory banks.
    const int packed_pitch=(((group+3)/4)|1)*4;
    const int group_bytes=(group*CACHE_BITS+7)/8;
    if(action==0 || action==2) {
        const int groups=(dim+group-1)/group;
        const int64_t index=int64_t(blockIdx.x)*4+warp;
        const int64_t count=int64_t(rows)*heads*groups;
        if(index>=count)return;
        const int row=index/(heads*groups),hg=index%(heads*groups);
        if(action==2 && !activity[row])return;
        const int head=hg/groups,start=(hg%groups)*group;
        float maximum=0.f;
        if(quantized)for(int member=lane;member<group;member+=32) {
            const int channel=start+member;
            const float x=channel<dim?read_native(incoming[(int64_t(row)*heads+head)*dim+channel],bf16):0.f;
            maximum=fmaxf(maximum,fabsf(x));
        }
        const float scale=scale_for(warp_max(maximum));
        const int64_t target_page=action==0?0:pages[row];
        const int64_t target=action==0?index:
            (int64_t(layer)*page+remainder(positions[row],page,pm,ps))*heads*groups+hg;
        if(quantized && lane==0)scales[target_page*scales_pitch+target]=scale;
        if(quantized)for(int byte=lane;byte<group_bytes;byte+=32) {
            const int first=byte*8/CACHE_BITS,shift=byte*8%CACHE_BITS;
            const int64_t origin=(int64_t(row)*heads+head)*dim+start;
            const unsigned a=start+first<dim?encode(read_native(incoming[origin+first],bf16),scale):0;
            const unsigned b=shift+8>CACHE_BITS && first+1<group && start+first+1<dim?
                encode(read_native(incoming[origin+first+1],bf16),scale):0;
            codes[target_page*payload_pitch+target*group_bytes+byte]=uint8_t((a>>shift)|(b<<(CACHE_BITS-shift)));
        }
        else for(int member=lane;member<group;member+=32) {
            const int channel=start+member;
            if(channel<dim)static_cast<uint16_t*>(payload)[
                target_page*(payload_pitch/sizeof(uint16_t))+
                ((int64_t(layer)*page+remainder(positions[row],page,pm,ps))*heads+head)*dim+channel]=
                    incoming[(int64_t(row)*heads+head)*dim+channel];
        }
        if(action==2 && hg==0 && lane==0)
            written[int64_t(pages[row])*written_pitch+int64_t(layer)*page+remainder(positions[row],page,pm,ps)]=true;
        return;
    }
    const int64_t flat=int64_t(blockIdx.y)*32+lane;
    if(action==1) {
        const int count=(rows+group-1)/group;
        float maximum=0.f;
        for(int member=warp;member<group;member+=4) {
            const int token=blockIdx.x*group+member;
            const float x=token<rows && flat<hd?read_native(incoming[int64_t(token)*hd+flat],bf16):0.f;
            maximum=fmaxf(maximum,fabsf(x));
        }
        partial[threadIdx.x]=maximum;__syncthreads();
        maximum=fmaxf(fmaxf(partial[lane],partial[32+lane]),fmaxf(partial[64+lane],partial[96+lane]));
        const float scale=scale_for(maximum);
        const int64_t index=flat*count+blockIdx.x;
        if(flat<hd) {
            if(warp==0)scales[index]=scale;
            for(int member=warp;member<group;member+=4) {
                const int token=blockIdx.x*group+member;
                const float x=token<rows?read_native(incoming[int64_t(token)*hd+flat],bf16):0.f;
                packed[lane*packed_pitch+member]=encode(x,scale);
            }
        }
        __syncthreads();
        for(int item=threadIdx.x;item<group_bytes*32;item+=128) {
            const int channel_in_tile=divide_index(item,bm,bs);
            const int64_t channel=int64_t(blockIdx.y)*32+channel_in_tile;
            const int member=item-channel_in_tile*group_bytes;
            if(channel<hd)codes[(channel*count+blockIdx.x)*group_bytes+member]=
                packed_byte(packed+channel_in_tile*packed_pitch,member,group);
        }
        return;
    }
    const int root=blockIdx.x;
    const int length=segments[root];
    if(length<=0 || !activity[root])return;
    const int64_t source=sources[root],first=positions[root];
    for(int member=warp;member<group;member+=4)
        history[member*32+lane]=source>=0 && flat<hd?
            tail[source*tail_pitch+(int64_t(layer)*group+member)*hd+flat]:0;
    __syncthreads();
    for(int step=0;step<length && root+step<rows;++step) {
        const int row=root+step;
        if(!activity[row])continue;
        const int64_t position=first+step;
        if(warp==0 && flat<hd)history[remainder(position,group,gm,gs)*32+lane]=incoming[int64_t(row)*hd+flat];
        __syncthreads();
        if(remainder(position+1,group,gm,gs)==0) {
            if(store_groups) {
                float maximum=0.f;
                for(int member=warp;member<group;member+=4)
                    maximum=fmaxf(maximum,fabsf(read_native(history[member*32+lane],bf16)));
                partial[threadIdx.x]=maximum;__syncthreads();
                maximum=fmaxf(fmaxf(partial[lane],partial[32+lane]),fmaxf(partial[64+lane],partial[96+lane]));
                const float scale=scale_for(maximum);
                const int groups_per_page=divide_index(page,gm,gs);
                const int group_in_page=divide_index(remainder(position,page,pm,ps),gm,gs);
                const int64_t index=int64_t(layer)*hd*groups_per_page
                    +flat*groups_per_page+group_in_page;
                if(flat<hd) {
                    if(warp==0)scales[int64_t(pages[row])*scales_pitch+index]=scale;
                    for(int member=warp;member<group;member+=4)
                        packed[lane*packed_pitch+member]=encode(read_native(history[member*32+lane],bf16),scale);
                }
                __syncthreads();
                for(int item=threadIdx.x;item<group_bytes*32;item+=128) {
                    const int channel_in_tile=divide_index(item,bm,bs);
                    const int64_t channel=int64_t(blockIdx.y)*32+channel_in_tile;
                    const int64_t target=int64_t(layer)*hd*groups_per_page
                        +channel*groups_per_page+group_in_page;
                    const int member=item-channel_in_tile*group_bytes;
                    if(channel<hd)codes[int64_t(pages[row])*payload_pitch+target*group_bytes+member]=
                        packed_byte(packed+channel_in_tile*packed_pitch,member,group);
                }
            }
            __syncthreads();
            for(int member=warp;member<group;member+=4)history[member*32+lane]=0;
        }
        __syncthreads();
        if(store_tail) {
            const int64_t destination=dest_i64?static_cast<const int64_t*>(destinations)[row]:
                static_cast<const int*>(destinations)[row];
            if(destination>=0 && flat<hd)for(int member=warp;member<group;member+=4)
                tail[destination*tail_pitch+(int64_t(layer)*group+member)*hd+flat]=history[member*32+lane];
        }
        if(store_groups && blockIdx.y==0 && threadIdx.x==0)
            written[int64_t(pages[row])*written_pitch+int64_t(layer)*page+remainder(position,page,pm,ps)]=true;
        __syncthreads();
    }
}
