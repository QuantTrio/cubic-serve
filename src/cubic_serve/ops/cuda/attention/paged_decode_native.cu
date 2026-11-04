// Baseline tiled MMA attention: runtime geometry, one live K/V tile and no expanded
// historical cache. The format reader preserves the native-query boundary.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include <cuda/atomic>
#include <stdint.h>
#include "activation_native.cuh"
#define CUBIC_MMA_BF16 INPUT_BF16
#include "mma.cuh"
#if INPUT_BF16
using scalar=__nv_bfloat16;
#else
using scalar=__half;
#endif

struct CacheStrides {
    int64_t page_bytes,scale_elements,tail_elements;
};

struct IndexDivisor {
    unsigned divisor,multiplier,shift;
    __device__ __forceinline__ unsigned divide(unsigned value) const {
        return (__umulhi(value,multiplier)+value)>>shift;
    }
    __device__ __forceinline__ unsigned remainder(unsigned value) const {
        return value-divide(value)*divisor;
    }
};
__device__ __forceinline__ int native_tile_offset(int row,int channel) {
    return (channel/16)*384+row*24+channel%16;
}
__device__ __forceinline__ int cache_tile_offset(int token,int channel,int column) {
    return column?(channel/16)*384+(channel%16)*24+token:native_tile_offset(token,channel);
}
__device__ __forceinline__ unsigned native_pair(float a,float b) {
#if INPUT_BF16
    return unsigned(__bfloat16_as_ushort(__float2bfloat16_rn(a)))|
        (unsigned(__bfloat16_as_ushort(__float2bfloat16_rn(b)))<<16);
#else
    return unsigned(__half_as_ushort(__float2half_rn(a)))|
        (unsigned(__half_as_ushort(__float2half_rn(b)))<<16);
#endif
}
__device__ __forceinline__ unsigned query_pair(const scalar* query,int row,int first,
    int head,int channel,int query_heads,int dim,int live_heads) {
    const auto* bits=reinterpret_cast<const unsigned short*>(query);
    const int64_t index=(int64_t(row)*query_heads+first+head)*dim+channel;
    const unsigned a=head<live_heads && channel<dim?bits[index]:0;
    const unsigned b=head<live_heads && channel+1<dim?bits[index+1]:0;
    return a|(b<<16);
}

__device__ float decode_code(uint8_t code,int integer,int bits) {
    if(integer)return float((int(code)^(1<<(bits-1)))-(1<<(bits-1)));
    const unsigned magnitude=code&127;
    float value=__uint_as_float((magnitude<<20)+0x3c000000);
    if(magnitude<8)value=float(magnitude)*0.001953125f;
    if(magnitude==127)value=__uint_as_float(0x7fc00000);
    return __uint_as_float(__float_as_uint(value)|((unsigned(code)&128)<<24));
}
__device__ __forceinline__ scalar read_cache(
    const void* data,const float* scales,const scalar* incoming,const scalar* tail,
    int64_t start,int64_t cutoff,int64_t source,int root,
    int row,int position,int page_id,int head,int channel,
    int layers,int layer,int page,int heads,int dim,int group,int integer,int bits,int column,int window,
    IndexDivisor page_div,IndexDivisor group_div,CacheStrides strides) {
    if(window) {
        if(position>=start && position>=cutoff)
            return incoming[((int64_t(root)+position-start)*heads+head)*dim+channel];
        if(group && column && position>=cutoff && position<start)
            return tail[source*strides.tail_elements+
                ((int64_t(layer)*group+group_div.remainder(position))*heads+head)*dim+channel];
    }
    const int token=page_div.remainder(position);
    const auto* page_data=static_cast<const uint8_t*>(data)+int64_t(page_id)*strides.page_bytes;
    if(!group)return reinterpret_cast<const scalar*>(page_data)[
        ((int64_t(layer)*page+token)*heads+head)*dim+channel];
    const int groups=group_div.divide((column?page:dim)+group-1);
    const int64_t count=column?int64_t(heads)*dim*groups:int64_t(page)*heads*groups;
    const int64_t index=int64_t(layer)*count+(column?
        (int64_t(head)*dim+channel)*groups+group_div.divide(token):
        (int64_t(token)*heads+head)*groups+group_div.divide(channel));
    const int member=group_div.remainder(column?token:channel);
    const int group_bytes=(group*bits+7)/8,offset=member*bits,byte=offset/8,shift=offset%8;
    const auto* codes=page_data+index*group_bytes;
    unsigned code=codes[byte]>>shift;
    if(shift+bits>8)code|=unsigned(codes[byte+1])<<(8-shift);
    code&=(1<<bits)-1;
    return scalar(__fmul_rn(decode_code(uint8_t(code),integer,bits),
        scales[int64_t(page_id)*strides.scale_elements+index]));
}
__device__ __forceinline__ void load_cache_tile(
    scalar* tile,const void* data,const float* scales,const scalar* incoming,const scalar* tail,
    const int* table,const int* selected,int64_t start,int64_t cutoff,int64_t source,int root,
    int row,int base,int end,int selected_width,int table_pages,int head,
    int layers,int layer,int page,int heads,int dim,int padded_dim,int group,int integer,int bits,int column,int window,
    IndexDivisor page_div,IndexDivisor group_div,IndexDivisor dim_div,CacheStrides strides) {
    column=column && group;
    const int vectors_per_token=padded_dim/8;
    for(int vector=threadIdx.x;vector<2*padded_dim;vector+=128) {
        const int token=column?(vector%2)*8:dim_div.divide(vector*8);
        const int channel=column?vector/2:(vector-token*vectors_per_token)*8;
        bool packed=group && (bits==4 || bits==6 || bits==8) && group%8==0 && base+token<end && channel<dim;
        int position=0,physical=0;
        if(packed) {
            position=selected_width?selected[int64_t(row)*selected_width+base+token]:base+token;
            physical=table[int64_t(row)*table_pages+page_div.divide(position)];
            const int member=group_div.remainder(column?page_div.remainder(position):channel);
            packed=column?(page>=8):member+8<=group;
            if(column) {
                packed=packed && base+token+7<end;
                if(selected_width && packed) {
                    #pragma unroll
                    for(int i=1;i<8;++i)
                        packed=packed && selected[int64_t(row)*selected_width+base+token+i]==position+i;
                }
            }
            if(window && packed) {
                packed=int64_t(position)+(column?7:0)<cutoff;
            }
            if(packed) {
                const int count=group_div.divide((column?page:dim)+group-1);
                const int64_t groups=column?int64_t(heads)*dim*count:int64_t(page)*heads*count;
                const int64_t index=int64_t(layer)*groups+(column?
                    (int64_t(head)*dim+channel)*count+group_div.divide(page_div.remainder(position)):
                    (int64_t(page_div.remainder(position))*heads+head)*count+group_div.divide(channel));
                const int group_bytes=(group*bits+7)/8;
                const auto* group_codes=static_cast<const uint8_t*>(data)+int64_t(physical)*strides.page_bytes+
                    index*group_bytes;
                if(column && (member+8>group || page_div.remainder(position)+8>page)) {
                    // One contiguous vector can cross one group/page boundary.
                    // Share its two descriptors; each code keeps its own scale
                    // and the exact scalar decode/multiply/rounding boundary.
                    const int first=min(group-member,page-int(page_div.remainder(position)));
                    const int second_position=position+first;
                    const int second_token=page_div.remainder(second_position);
                    const int second_page=table[int64_t(row)*table_pages+page_div.divide(second_position)];
                    const int64_t second_index=int64_t(layer)*groups+
                        (int64_t(head)*dim+channel)*count+group_div.divide(second_token);
                    const auto* second_codes=static_cast<const uint8_t*>(data)+
                        int64_t(second_page)*strides.page_bytes+second_index*group_bytes;
                    const float scale0=scales[int64_t(physical)*strides.scale_elements+index];
                    const float scale1=scales[int64_t(second_page)*strides.scale_elements+second_index];
                    unsigned pairs[4];
                    #pragma unroll
                    for(int pair=0;pair<4;++pair) {
                        float values[2];
                        #pragma unroll
                        for(int half=0;half<2;++half) {
                            const int i=2*pair+half;
                            const bool next=i>=first;
                            const int offset=(next?i-first:member+i)*bits;
                            const auto* codes=next?second_codes:group_codes;
                            const int shift=offset%8;
                            unsigned code=codes[offset/8]>>shift;
                            if(shift+bits>8)code|=unsigned(codes[offset/8+1])<<(8-shift);
                            values[half]=__fmul_rn(decode_code(uint8_t(code&((1u<<bits)-1)),integer,bits),
                                                  next?scale1:scale0);
                        }
                        pairs[pair]=native_pair(values[0],values[1]);
                    }
                    *reinterpret_cast<uint4*>(tile+cache_tile_offset(token,channel,column))=
                        make_uint4(pairs[0],pairs[1],pairs[2],pairs[3]);
                    continue;
                }
                const auto* codes=group_codes+member*bits/8;
                uint64_t bytes;
                if(column && member%8) {
                    // A contiguous span may start between packed words. Read
                    // only its bytes, without crossing the proven group bound.
                    const int shift=member*bits%8;
                    const int count=(8*bits+shift+7)/8;
                    bytes=0;
                    #pragma unroll
                    for(int byte=0;byte<8;++byte)
                        if(byte<count)bytes|=uint64_t(codes[byte])<<(8*byte);
                    bytes>>=shift;
                }
                else if(bits==8)bytes=*reinterpret_cast<const uint64_t*>(codes);
                else if(bits==4)bytes=*reinterpret_cast<const uint32_t*>(codes);
                else {
                    // Six bytes are only guaranteed to be two-byte aligned.
                    const auto* words=reinterpret_cast<const uint16_t*>(codes);
                    bytes=uint64_t(words[0])|(uint64_t(words[1])<<16)|(uint64_t(words[2])<<32);
                }
                const unsigned mask=(1u<<bits)-1;
                const float scale=scales[int64_t(physical)*strides.scale_elements+index];
                unsigned pairs[4];
                #pragma unroll
                for(int pair=0;pair<4;++pair) {
                    const float a=(column || channel+pair*2<dim)?
                        __fmul_rn(decode_code(uint8_t((bytes>>(pair*2*bits))&mask),integer,bits),scale):0.f;
                    const float b=(column || channel+pair*2+1<dim)?
                        __fmul_rn(decode_code(uint8_t((bytes>>((pair*2+1)*bits))&mask),integer,bits),scale):0.f;
                    pairs[pair]=native_pair(a,b);
                }
                *reinterpret_cast<uint4*>(tile+cache_tile_offset(token,channel,column))=
                    make_uint4(pairs[0],pairs[1],pairs[2],pairs[3]);
            }
        }
        if(!packed) {
            #pragma unroll
            for(int i=0;i<8;++i) {
                const int t=token+(column?i:0),c=channel+(column?0:i);
                scalar current=scalar(0.f);
                if(base+t<end && c<dim) {
                    const int p=selected_width?selected[int64_t(row)*selected_width+base+t]:base+t;
                    const int id=table[int64_t(row)*table_pages+page_div.divide(p)];
                    current=read_cache(data,scales,incoming,tail,start,cutoff,source,root,row,p,id,head,c,
                        layers,layer,page,heads,dim,group,integer,bits,column,window,page_div,group_div,strides);
                }
                tile[cache_tile_offset(t,c,column)]=current;
            }
        }
    }
}
__device__ float half_warp_max(float value) {
    for(int d=8;d;d>>=1)value=fmaxf(value,__shfl_xor_sync(0xffffffff,value,d,16));
    return value;
}
__device__ float half_warp_sum(float value) {
    for(int d=8;d;d>>=1)value=__fadd_rn(value,__shfl_xor_sync(0xffffffff,value,d,16));
    return value;
}
__device__ float probability_exp(float value) {
#if ACT_KIND == 0
    return expf(value);
#else
    return activation_exp_nonpositive(activation_round(
        __fmul_rn(activation_round(value),activation_round(1.4426950408889634f))));
#endif
}

#if VALUE_ROTATION
__device__ __forceinline__ void restore_value_basis(
    const float* values,const float* owned,scalar* output,int row,int first_head,
    int live_heads,int query_heads,int dim,int padded_dim,IndexDivisor dim_div) {
    // Each warp owns whole heads. Low butterfly stages exchange within lanes;
    // higher stages pair registers. Preserve stage order and FP32 RN boundaries.
    constexpr int items=REGISTER_TILES*2;
    const int lane=threadIdx.x&31,warp=threadIdx.x>>5;
    const int group=dim&-dim,exponent=__ffs(group)-1;
    const float normalization=ldexpf((exponent&1)?0.7071067690849304f:1.f,-(exponent/2));
    for(int head=warp;head<live_heads;head+=blockDim.x/32) {
        if(!owned[head])continue;
        float x[items];
        #pragma unroll
        for(int item=0;item<items;++item) {
            const int channel=lane+32*item;
            x[item]=channel<dim?values[head*padded_dim+channel]:0.f;
        }
        #pragma unroll
        for(int step=1;step<32;step*=2)if(step<group) {
            #pragma unroll
            for(int item=0;item<items;++item) {
                const float peer=__shfl_xor_sync(0xffffffff,x[item],step);
                x[item]=(lane&step)?__fsub_rn(peer,x[item]):__fadd_rn(x[item],peer);
            }
        }
        #pragma unroll
        for(int stride=1;stride<items;stride*=2)if(32*stride<group) {
            #pragma unroll
            for(int pair=0;pair<items/2;++pair) {
                const int first=(pair/stride)*(2*stride)+pair%stride;
                const float a=x[first],b=x[first+stride];
                x[first]=__fadd_rn(a,b);x[first+stride]=__fsub_rn(a,b);
            }
        }
        #pragma unroll
        for(int item=0;item<items;++item) {
            const int channel=lane+32*item;
            if(channel<dim)
                output[(int64_t(row)*query_heads+first_head+head)*dim+channel]=
                    scalar(__fmul_rn(x[item],normalization));
        }
    }
}
#endif

extern "C" __global__ void paged_decode_native(
    const scalar* __restrict__ query,const void* __restrict__ key,const void* __restrict__ value,
    const int* __restrict__ table,const int* __restrict__ lengths,
    scalar* __restrict__ output,float* __restrict__ partial,float* __restrict__ statistics,
    int* __restrict__ arrivals,const int* __restrict__ selected,
    const float* __restrict__ key_scales,const float* __restrict__ value_scales,
    const scalar* __restrict__ incoming_key,const scalar* __restrict__ incoming_value,
    const scalar* __restrict__ key_tail,const scalar* __restrict__ value_tail,
    const int64_t* __restrict__ starts,const int* __restrict__ roots,const int64_t* __restrict__ sources,
    const int* __restrict__ row_activity,
    int layers,int layer,int page,int table_pages,int query_heads,int kv_heads,int dim,int padded_dim,
    int splits,int selected_width,int key_group,int value_group,int key_column,int value_column,
    int window,float scale,unsigned page_m,int page_s,unsigned dim_m,int dim_s,
    unsigned key_m,int key_s,unsigned value_m,int value_s,
    int64_t key_page_stride,int64_t value_page_stride,
    int64_t key_scale_stride,int64_t value_scale_stride,
    int64_t key_tail_stride,int64_t value_tail_stride) {
    window=CACHE_WINDOW;key_column=KEY_COLUMN;value_column=VALUE_COLUMN;
    key_group=KEY_ENCODED?key_group:0;value_group=VALUE_ENCODED?value_group:0;
    const IndexDivisor page_div{unsigned(page),page_m,unsigned(page_s)};
    const IndexDivisor dim_div{unsigned(padded_dim),dim_m,unsigned(dim_s)};
    const IndexDivisor key_div{unsigned(key_group?key_group:1),key_m,unsigned(key_s)};
    const IndexDivisor value_div{unsigned(value_group?value_group:1),value_m,unsigned(value_s)};
    const int tid=threadIdx.x,lane=tid&31,warp=tid>>5,row=blockIdx.x,split=blockIdx.z;
    // Window ownership and causal sealing are invariant over all K/V tiles.
    // Keep absolute positions in INT64; calculate each boundary once per CTA,
    // rather than dividing again for every scalar/vector cache read.
    const int root=window?roots[row]:0;
    const int64_t start=window?starts[row]:0,source=window?sources[row]:0;
    const int64_t query_position=start+row-root;
    const int64_t key_cutoff=window?(key_group?
        (key_column?query_position/key_group*key_group:query_position):start):0;
    const int64_t value_cutoff=window?(value_group?
        (value_column?query_position/value_group*value_group:query_position):start):0;
    const int head_group=query_heads/kv_heads,tiles=(head_group+15)/16;
    const int kv_head=blockIdx.y/tiles,head_tile=blockIdx.y%tiles;
    const int first_head=kv_head*head_group+head_tile*16;
    const int live_heads=min(16,head_group-head_tile*16);
#if CACHE_ACTIVITY
    if(!row_activity[row]) {
        // Padding has no request identity. Never read its page table, native
        // span or tail. Only split zero owns the private output write.
        if(split==0)for(int item=tid;item<live_heads*dim;item+=128)
            output[(int64_t(row)*query_heads+first_head)*dim+item]=scalar(0.f);
        return;
    }
#endif
    // Eight native elements skew each row over the WMMA load banks. The
    // accumulator store uses a four-float skew for the same reason.
    const int tile_stride=20,probability_stride=24;
    extern __shared__ __align__(32) unsigned char storage[];
    auto* maxima=reinterpret_cast<float*>(storage);
    float* denominator=maxima+16;float* alpha=denominator+16;
    float* tile_output=alpha+16;
#if VALUE_ROTATION
    float* rotated_output=tile_output+1024;
#endif
    auto* kv=reinterpret_cast<scalar*>(tile_output+1280);
    scalar* probabilities=kv+24*padded_dim;
    unsigned query_fragments[REGISTER_TILES][4];
    float accumulated[REGISTER_TILES][2][4]={};
    #pragma unroll
    for(int tile=0;tile<REGISTER_TILES;++tile) {
        const int channel=warp*16+tile*64+(lane%4)*2,head=lane/4;
        query_fragments[tile][0]=query_pair(query,row,first_head,head,channel,query_heads,dim,live_heads);
        query_fragments[tile][1]=query_pair(query,row,first_head,head+8,channel,query_heads,dim,live_heads);
        query_fragments[tile][2]=query_pair(query,row,first_head,head,channel+8,query_heads,dim,live_heads);
        query_fragments[tile][3]=query_pair(query,row,first_head,head+8,channel+8,query_heads,dim,live_heads);
    }
    if(tid<16) {maxima[tid]=-CUDART_INF_F;denominator[tid]=0.f;}
    __syncthreads();
    const int length=lengths[row],span=(length+splits-1)/splits;
    const int begin=split*span,end=min(begin+span,length);
    for(int base=begin;base<end;base+=16) {
        load_cache_tile(kv,key,key_scales,incoming_key,key_tail,table,selected,start,key_cutoff,source,root,
            row,base,end,selected_width,table_pages,kv_head,layers,layer,page,kv_heads,dim,padded_dim,
            key_group,KEY_INTEGER,KEY_BITS,key_column,window,page_div,key_div,dim_div,
            {key_page_stride,key_scale_stride,key_tail_stride});
        __syncthreads();
        float scores[2][4]={};
        #pragma unroll
        for(int tile=0;tile<REGISTER_TILES;++tile) {
            const int channel=warp*16+tile*64;
            if(channel>=padded_dim)continue;
            #pragma unroll
            for(int half=0;half<2;++half) {
                unsigned b[2];
                if(KEY_ENCODED && KEY_COLUMN)cubic_load_mma_b_row(b,kv+channel*24+half*8,24,lane);
                else cubic_load_mma_b_col(b,kv+channel*24+half*8*24,24,lane);
                cubic_mma16816(scores[half],query_fragments[tile],b);
            }
        }
        #pragma unroll
        for(int half=0;half<2;++half) {
            #pragma unroll
            for(int i=0;i<4;++i)
                tile_output[warp*320+(lane/4+(i/2)*8)*tile_stride+half*8+(lane%4)*2+i%2]=scores[half][i];
        }
        __syncthreads();
        for(int pass=0;pass<2;++pass) {
            const int head=warp*2+lane/16+pass*8,token=lane%16;
            const int offset=head*tile_stride+token;
            const float score=__fadd_rn(__fadd_rn(__fadd_rn(tile_output[offset],
                tile_output[320+offset]),tile_output[640+offset]),tile_output[960+offset]);
            const float logit=base+token<end?__fmul_rn(score,scale):-CUDART_INF_F;
            const float next=fmaxf(maxima[head],half_warp_max(logit));
            const float p=probability_exp(__fsub_rn(logit,next));
            const float sum=half_warp_sum(p);
            probabilities[head*probability_stride+token]=scalar(p);
            if(token==0) {
                const float old=expf(__fsub_rn(maxima[head],next));
                alpha[head]=old;maxima[head]=next;
                denominator[head]=__fadd_rn(__fmul_rn(denominator[head],old),sum);
            }
        }
        __syncthreads();
        load_cache_tile(kv,value,value_scales,incoming_value,value_tail,table,selected,start,value_cutoff,source,root,
            row,base,end,selected_width,table_pages,kv_head,layers,layer,page,kv_heads,dim,padded_dim,
            value_group,VALUE_INTEGER,VALUE_BITS,value_column,window,page_div,value_div,dim_div,
            {value_page_stride,value_scale_stride,value_tail_stride});
        __syncthreads();
        unsigned probability_fragment[4];
        cubic_load_mma_a(probability_fragment,probabilities,probability_stride,lane);
        #pragma unroll
        for(int tile=0;tile<REGISTER_TILES;++tile) {
            const int channel=warp*16+tile*64;
            if(channel>=padded_dim)continue;
            #pragma unroll
            for(int half=0;half<2;++half) {
                unsigned b[2];
                if(VALUE_ENCODED && VALUE_COLUMN)cubic_load_mma_b_col(b,kv+channel*24+half*8*24,24,lane);
                else cubic_load_mma_b_row(b,kv+channel*24+half*8,24,lane);
                float c[4]={};cubic_mma16816(c,probability_fragment,b);
                #pragma unroll
                for(int i=0;i<4;++i) {
                    const int head=lane/4+(i/2)*8;
                    accumulated[tile][half][i]=__fadd_rn(
                        __fmul_rn(accumulated[tile][half][i],alpha[head]),c[i]);
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int tile=0;tile<REGISTER_TILES;++tile) {
        #pragma unroll
        for(int half=0;half<2;++half) {
            #pragma unroll
            for(int i=0;i<4;++i) {
                const int head=lane/4+(i/2)*8,d=warp*16+tile*64+half*8+(lane%4)*2+i%2;
                if(head<live_heads && d<dim) {
                    const int64_t index=(int64_t(row)*query_heads+first_head+head)*dim+d;
                    if(splits==1) {
#if VALUE_ROTATION
                        rotated_output[head*padded_dim+d]=__fdiv_rn(accumulated[tile][half][i],denominator[head]);
#else
                        output[index]=scalar(__fdiv_rn(accumulated[tile][half][i],denominator[head]));
#endif
                    }
                    else partial[((int64_t(row)*query_heads+first_head+head)*splits+split)*dim+d]=accumulated[tile][half][i];
                }
            }
        }
    }
    if(splits==1) {
#if VALUE_ROTATION
        if(tid<live_heads)alpha[tid]=1.f;
        __syncthreads();
        restore_value_basis(rotated_output,alpha,output,row,first_head,live_heads,query_heads,dim,padded_dim,dim_div);
#endif
        return;
    }
    if(tid<live_heads) {
        const int64_t stat=((int64_t(row)*query_heads+first_head+tid)*splits+split)*2;
        statistics[stat]=maxima[tid];statistics[stat+1]=denominator[tid];
    }
    __threadfence();__syncthreads();
    if(tid<16) {
        alpha[tid]=0.f;
        if(tid<live_heads) {
            cuda::atomic_ref<int,cuda::thread_scope_device> counter(arrivals[int64_t(row)*query_heads+first_head+tid]);
            alpha[tid]=counter.fetch_add(1,cuda::memory_order_acq_rel)==splits-1;
        }
    }
    __syncthreads();
    // K/V, Q and the warp-private tile are dead. Reuse tile_output for the
    // fixed-order merge coefficients; every head acquires its own publication.
    float* old_scales=tile_output;float* part_scales=tile_output+512;
    if(tid<live_heads && alpha[tid]) {
        float m=-CUDART_INF_F,den=0.f;
        for(int part=0;part<splits;++part) {
            const int64_t stat=((int64_t(row)*query_heads+first_head+tid)*splits+part)*2;
            const float pm=statistics[stat],pd=statistics[stat+1],next=fmaxf(m,pm);
            const float old=expf(__fsub_rn(m,next)),weight=expf(__fsub_rn(pm,next));
            old_scales[tid*32+part]=old;part_scales[tid*32+part]=weight;
            den=__fadd_rn(__fmul_rn(den,old),__fmul_rn(pd,weight));m=next;
        }
        denominator[tid]=den;
    }
    __syncthreads();
    for(int item=tid;item<16*padded_dim;item+=128) {
        const int head=dim_div.divide(item),d=item-head*padded_dim;
        if(head<live_heads && d<dim && alpha[head]) {
            float sum=0.f;
            for(int part=0;part<splits;++part) {
                const int64_t index=((int64_t(row)*query_heads+first_head+head)*splits+part)*dim+d;
                sum=__fadd_rn(__fmul_rn(sum,old_scales[head*32+part]),
                             __fmul_rn(partial[index],part_scales[head*32+part]));
            }
#if VALUE_ROTATION
            rotated_output[head*padded_dim+d]=__fdiv_rn(sum,denominator[head]);
#else
            output[(int64_t(row)*query_heads+first_head+head)*dim+d]=scalar(__fdiv_rn(sum,denominator[head]));
#endif
        }
    }
#if VALUE_ROTATION
    __syncthreads();
    restore_value_basis(rotated_output,alpha,output,row,first_head,live_heads,query_heads,dim,padded_dim,dim_div);
#endif
    __threadfence();__syncthreads();
    if(tid<live_heads && alpha[tid]) {
        cuda::atomic_ref<int,cuda::thread_scope_device> counter(arrivals[int64_t(row)*query_heads+first_head+tid]);
        counter.store(0,cuda::memory_order_release);
    }
}
