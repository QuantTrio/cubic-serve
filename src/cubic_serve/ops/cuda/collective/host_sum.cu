// Ordered mapped-host transport. Inspected references:
// llama.cpp 8b4b355 ggml-cuda/allreduce.cu (mapped memory and system fences),
// vLLM 89af4f1 custom_collective_common.cuh and SGLang ba433bb
// custom_all_reduce.cuh (parallel peer polling).
// Arithmetic remains Cubic's ascending-rank FP32 sum and one final RNE cast.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

template<class T> __device__ float to_float(T value);
template<> __device__ float to_float(__nv_bfloat16 value) {
    return __bfloat162float(value);
}
template<> __device__ float to_float(__half value) {
    return __half2float(value);
}
template<class T> __device__ T from_float(float value);
template<> __device__ __nv_bfloat16 from_float(float value) {
    return __float2bfloat16_rn(value);
}
template<> __device__ __half from_float(float value) {
    return __float2half_rn(value);
}
#if HOST_SUM_FP16
using Element = __half;
#else
using Element = __nv_bfloat16;
#endif

#if !HOST_SUM_PARTITIONED
extern "C" __global__ void ordered_host_sum(
    const Element* input, Element* output,
    const unsigned long long* arena_address,
    int count, int stride, int ranks, int rank, unsigned* epochs) {
    using T = Element;
    char* arena = reinterpret_cast<char*>(arena_address[0]);
    const int tid = threadIdx.x, bid = blockIdx.x, blocks = gridDim.x;
    volatile unsigned* flags =
        reinterpret_cast<volatile unsigned*>(arena + size_t(stride) * ranks * 2);
    auto flag = [&](int phase, int peer) -> volatile unsigned* {
        return flags + ((phase * ranks + peer) * blocks + bid) * 16;
    };
    __shared__ unsigned token;
    if (tid == 0) token = epochs[bid] + 1;
    __syncthreads();
    // Publishing g+1 proves that rank finished reading g. No rank can reach
    // g+2 until every peer published g+1, so g's payload AND ready flag remain
    // intact for every reader. Calls retain topology-wide lease/stream order.
    const int slot = token & 1;
    T* mine = reinterpret_cast<T*>(arena + size_t(stride) * (slot * ranks + rank));
    const int vectors = count / 8;
    for (int i = bid * blockDim.x + tid; i < vectors; i += blocks * blockDim.x) {
        reinterpret_cast<uint4*>(mine)[i] = reinterpret_cast<const uint4*>(input)[i];
    }
    if (bid == 0) {
        for (int i = vectors * 8 + tid; i < count; i += blockDim.x) mine[i] = input[i];
    }
    __threadfence_system();
    __syncthreads();
    if (tid == 0) {
        *flag(slot, rank) = token;
        __threadfence_system();
    }
    __syncthreads();
    for (int peer = tid; peer < ranks; peer += blockDim.x) {
        while (*flag(slot, peer) != token) __nanosleep(64);
    }
    __syncthreads();
    __threadfence_system();
    for (int i = bid * blockDim.x + tid; i < vectors; i += blocks * blockDim.x) {
        float sum[8] = {};
        for (int peer = 0; peer < ranks; ++peer) {
            const T* source = peer == rank ? input :
                reinterpret_cast<const T*>(arena + size_t(stride) * (slot * ranks + peer));
            union { uint4 vector; T elements[8]; } value;
            value.vector = reinterpret_cast<const uint4*>(source)[i];
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                sum[j] = __fadd_rn(sum[j], to_float(value.elements[j]));
            }
        }
        union { uint4 vector; T elements[8]; } value;
        #pragma unroll
        for (int j = 0; j < 8; ++j) value.elements[j] = from_float<T>(sum[j]);
        reinterpret_cast<uint4*>(output)[i] = value.vector;
    }
    if (bid == 0) {
        for (int i = vectors * 8 + tid; i < count; i += blockDim.x) {
            float sum = 0;
            for (int peer = 0; peer < ranks; ++peer) {
                const T* source = peer == rank ? input :
                    reinterpret_cast<const T*>(arena + size_t(stride) * (slot * ranks + peer));
                sum = __fadd_rn(sum, to_float(source[i]));
            }
            output[i] = from_float<T>(sum);
        }
    }
    __syncthreads();
    if (tid == 0) epochs[bid] = token;
}

#else
extern "C" __global__ void ordered_host_sum(
    const Element* input, Element* output,
    const unsigned long long* arena_address,
    int count, int stride, int ranks, int rank, unsigned* epochs) {
    using T = Element;
    char* arena = reinterpret_cast<char*>(arena_address[0]);
    const int tid = threadIdx.x, bid = blockIdx.x, blocks = gridDim.x;
    volatile unsigned* flags =
        reinterpret_cast<volatile unsigned*>(arena + size_t(stride) * ranks * 2);
    auto flag = [&](int phase, int peer) -> volatile unsigned* {
        return flags + ((phase * ranks + peer) * blocks + bid) * 16;
    };
    __shared__ unsigned token;
    if (tid == 0) token = epochs[bid] + 1;
    __syncthreads();
    // Never overwrite a prior generation that any rank is still reading.
    for (int peer = tid; peer < ranks; peer += blockDim.x) {
        while (*flag(1, peer) != token - 1) __nanosleep(64);
    }
    __syncthreads();
    T* mine = reinterpret_cast<T*>(arena + size_t(stride) * rank);
    const int vectors = (count + 7) / 8;
    const int chunk = blockDim.x * ranks;
    for (int base = bid * chunk; base < vectors; base += blocks * chunk) {
      for (int part=0; part<ranks; ++part) {
        // The owner reduces this stripe from local input. No peer reads its
        // host input copy; peers consume the separately published final sum.
        if (part == rank) continue;
        int v=base+part*blockDim.x+tid, i=v*8;
        if(i+7<count) reinterpret_cast<uint4*>(mine)[v]=reinterpret_cast<const uint4*>(input)[v];
        else for(int j=0;i+j<count;++j) mine[i+j]=input[i+j];
      }
    }
    __threadfence_system();
    __syncthreads();
    if (tid == 0) {
        *flag(0, rank) = token;
        __threadfence_system();
    }
    __syncthreads();
    for (int peer = tid; peer < ranks; peer += blockDim.x) {
        while (*flag(0, peer) != token) __nanosleep(64);
    }
    __syncthreads();
    __threadfence_system();
    for(int base=bid*chunk;base<vectors;base+=blocks*chunk){
      int v=base+rank*blockDim.x+tid,i=v*8;
      if(i+7<count){
        float sum[8]={};
        for(int peer=0;peer<ranks;++peer){
          const T* source=peer==rank?input:reinterpret_cast<const T*>(arena+size_t(stride)*peer);
          union{uint4 vector;T elements[8];}value;
          value.vector=reinterpret_cast<const uint4*>(source)[v];
          #pragma unroll
          for(int j=0;j<8;++j)sum[j]=__fadd_rn(sum[j],to_float(value.elements[j]));
        }
        union{uint4 vector;T elements[8];}value;
        #pragma unroll
        for(int j=0;j<8;++j)value.elements[j]=from_float<T>(sum[j]);
        // This rank owns the completed stripe. Keep its final bits locally;
        // all peers consume the already-published input, so in-place is safe.
        reinterpret_cast<uint4*>(output)[v]=value.vector;
        reinterpret_cast<uint4*>(arena+size_t(stride)*(ranks+rank))[v]=value.vector;
      }else for(int j=0;i+j<count;++j){
        float sum=0;
        for(int peer=0;peer<ranks;++peer){
          const T* source=peer==rank?input:reinterpret_cast<const T*>(arena+size_t(stride)*peer);
          sum=__fadd_rn(sum,to_float(source[i+j]));
        }
        T value=from_float<T>(sum);
        output[i+j]=value;
        reinterpret_cast<T*>(arena+size_t(stride)*(ranks+rank))[i+j]=value;
      }
    }
    __threadfence_system();
    __syncthreads();
    if (tid == 0) { *flag(2, rank) = token; __threadfence_system(); }
    __syncthreads();
    for (int peer = tid; peer < ranks; peer += blockDim.x) {
        while (*flag(2, peer) != token) __nanosleep(64);
    }
    __syncthreads();
    __threadfence_system();
    for(int base=bid*chunk;base<vectors;base+=blocks*chunk){
      for(int part=0;part<ranks;++part){
        if(part==rank)continue;
        int v=base+part*blockDim.x+tid,i=v*8;
        const T* source=reinterpret_cast<const T*>(arena+size_t(stride)*(ranks+part));
        if(i+7<count)reinterpret_cast<uint4*>(output)[v]=reinterpret_cast<const uint4*>(source)[v];
        else for(int j=0;i+j<count;++j)output[i+j]=source[i+j];
      }
    }
    __syncthreads();
    __threadfence_system();
    if (tid == 0) {
        epochs[bid] = token;
        *flag(1, rank) = token;
        __threadfence_system();
    }
}

#endif
