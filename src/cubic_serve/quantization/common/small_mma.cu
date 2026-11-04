// Organization adapted from T-Crypt/ninfer-4090 q4_small_t_mma.cuh at b9a6b5c (Apache-2.0).
// Row-major packed transport and transposed small-M MMA.
// Format-owned decode and storage rounding precede every product.
// This primitive does not qualify a serving graph.
#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <numeric>
template<int RawBytes>
__device__ __forceinline__ void packed_group_copy(std::uint8_t* target,
    const std::uint8_t* original, int column, int tile, int lane) {
    constexpr int bytes = TILE_K / GI * RawBytes;
    constexpr int alignment = std::gcd(bytes, INPUT_K / GI * RawBytes);
    constexpr int vector = alignment % 16 == 0 ? 16 : alignment % 8 == 0 ? 8 : alignment % 4 == 0 ? 4 : 2;
    const auto* source = original +
        (std::int64_t(column) * (INPUT_K / GI) + tile * (TILE_K / GI)) * RawBytes;
    for (int v = lane; v < bytes / vector; v += 32) {
        if constexpr (vector == 2)
            reinterpret_cast<unsigned short*>(target)[v] = reinterpret_cast<const unsigned short*>(source)[v];
        else __pipeline_memcpy_async(target + v * vector, source + v * vector, vector);
    }
}
#include "decoder.cuh"
#if IS_BF16
using scalar = __nv_bfloat16;
#else
using scalar = __half;
#endif
__device__ __forceinline__ scalar rounded(float x) {
#if IS_BF16
    return __float2bfloat16_rn(x);
#else
    return __float2half_rn(x);
#endif
}
constexpr int k_per_warp = TILE_K / WARPS;
constexpr int input_stride = ((k_per_warp + 63) / 64) * 64;
constexpr int tile_m = ((ROWS + 7) / 8) * 8;
constexpr int nt = tile_m / 8;
constexpr int code_vectors = TILE_K * BITS / 128;
constexpr int raw_vectors = (TILE_K / GI * FORMAT_RAW_BYTES + 15) / 16;
static_assert(16 % WARPS == 0 && k_per_warp % 16 == 0 && INPUT_K % TILE_K == 0);
__device__ __forceinline__ unsigned weight_pair(const std::uint8_t* row, int offset,
                                               float2 metadata, FormatLevels levels, int lane,
                                               const unsigned* lut) {
    const int bit = offset * BITS;
    unsigned packed;
    if constexpr (BITS == 8) packed = *reinterpret_cast<const unsigned short*>(row + offset);
    else if constexpr (BITS == 4) packed = row[offset / 2];
    else {
        const unsigned* words = reinterpret_cast<const unsigned*>(row);
        packed = words[bit / 32] >> (bit % 32);
        if (bit % 32 + 2 * BITS > 32) packed |= words[bit / 32 + 1] << (32 - bit % 32);
    }
    constexpr unsigned mask = (1u << BITS) - 1;
    const unsigned q0 = packed & mask, q1 = (packed >> BITS) & mask;
    const unsigned v0 = (lut[q0 / 2] >> (16 * (q0 & 1))) & 65535;
    const unsigned v1 = (lut[q1 / 2] >> (16 * (q1 & 1))) & 65535;
    return v0 | (v1 << 16);
}
__device__ __forceinline__ void mma(float* c, unsigned a0, unsigned a1,
    unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
#if IS_BF16
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
#else
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
#endif
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}
extern "C" __launch_bounds__(WARPS*32,6) __global__ void packed_small_m_mma(
    const scalar* __restrict__ x,
    const std::uint8_t* __restrict__ codes0, const std::uint8_t* __restrict__ codes1,
    const std::uint8_t* __restrict__ metadata0, const std::uint8_t* __restrict__ metadata1,
    scalar* __restrict__ out0, scalar* __restrict__ out1) {
    union Storage {
      struct { uint4 code[16][code_vectors+1]; uint4 raw[16][raw_vectors];
        float2 metadata[16][TILE_K/GI];
        unsigned levels[16][(TILE_K/GI)*(1 << (BITS-1))+4];
        scalar inputs[WARPS][tile_m][input_stride]; } staging;
      float partial[WARPS][nt][32][4];
    };
    __shared__ __align__(16) Storage storage;
    auto &code=storage.staging.code; auto &raw=storage.staging.raw;
    auto &metadata=storage.staging.metadata; auto &inputs=storage.staging.inputs;
    auto &partial=storage.partial;
    auto &lut=storage.staging.levels;
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int member = blockIdx.x / ((OUTPUT_N + 15) / 16);
    const int row0 = blockIdx.x % ((OUTPUT_N + 15) / 16) * 16;
    const auto* codes = member == 0 ? codes0 : codes1;
    const auto* blob = member == 0 ? metadata0 : metadata1;
    scalar* output = member == 0 ? out0 : out1;
    FormatContext contexts[16 / WARPS];
    #pragma unroll
    for (int ri = 0; ri < 16 / WARPS; ++ri)
        contexts[ri] = format_context(blob, row0 + warp * (16 / WARPS) + ri);
    float acc[nt][4] = {};
    for(int v=lane;v<(tile_m-ROWS)*(k_per_warp/8);v+=32){
        int m=ROWS+v/(k_per_warp/8),k8=v%(k_per_warp/8);
        *reinterpret_cast<uint4*>(&inputs[warp][m][(k8*8)^((m&7)*8)])=make_uint4(0,0,0,0);
    }

    const int b_rin=lane&7,b_koff=((lane>>3)&1)<<3;
    #pragma unroll
    for (int tile = 0; tile < INPUT_K / TILE_K; ++tile) {
        #pragma unroll
        for (int ri = 0; ri < 16 / WARPS; ++ri) {
            const int row = warp * (16 / WARPS) + ri;
            if (true) {
                for (int v = lane; v < code_vectors; v += 32) {
                    const unsigned addr=static_cast<unsigned>(__cvta_generic_to_shared(&code[row][v]));
                    const auto* source=codes+std::int64_t(row0+row)*(INPUT_K*BITS/8)+tile*(TILE_K*BITS/8)+v*16;
                    asm volatile("cp.async.cg.shared.global [%0],[%1],16;"::"r"(addr),"l"(source));
                }
                format_copy(reinterpret_cast<std::uint8_t*>(raw[row]), blob,
                            row0 + row, tile, lane);
            }
        }
        for (int v = lane; v < ROWS * (k_per_warp / 8); v += 32) {
            int m = v / (k_per_warp / 8), k8 = v % (k_per_warp / 8);
            auto* dst = reinterpret_cast<uint4*>(&inputs[warp][m][(k8*8)^((m&7)*8)]);
            __pipeline_memcpy_async(dst,x+m*INPUT_K+tile*TILE_K+warp*k_per_warp+k8*8,16);
        }
        __pipeline_commit(); __pipeline_wait_prior(0); __syncthreads();
        constexpr int pairs = 1 << (BITS - 1);
        for (int item = threadIdx.x; item < 16*(TILE_K/GI)*pairs; item += WARPS*32) {
            const int pair = item % pairs;
            const int group = (item / pairs) % (TILE_K/GI);
            const int row = item / (pairs * (TILE_K/GI));
            format_emit_lut(lut[row] + group*pairs,
                reinterpret_cast<const std::uint8_t*>(raw[row]), blob,
                format_context(blob, row0 + row), row0 + row, tile, group, pair);
        }
        __syncthreads();
        const int gid = lane / 4, lid = lane % 4;
        constexpr int group_span = std::gcd(GI, k_per_warp);
        static_assert(group_span % 16 == 0 && k_per_warp % group_span == 0);
        #pragma unroll
        for (int group_start = 0; group_start < k_per_warp; group_start += group_span) {
        const float2 top_meta = {}, bottom_meta = {};
        const FormatLevels top_levels = {}, bottom_levels = {};
        const unsigned* top_lut = lut[gid] + ((warp*k_per_warp+group_start)/GI)*pairs;
        const unsigned* bottom_lut = lut[gid+8] + ((warp*k_per_warp+group_start)/GI)*pairs;
        #pragma unroll
        for (int kk = 0; kk < group_span; kk += 16) {
            const int ks = group_start + kk;
            const int k = warp * k_per_warp + ks + lid * 2;
            unsigned a0=0, a1=0, a2=0, a3=0;
            if (true) {
                a0 = weight_pair(reinterpret_cast<const std::uint8_t*>(code[gid]), k,
                                 top_meta, top_levels, lane, top_lut);
                a2 = weight_pair(reinterpret_cast<const std::uint8_t*>(code[gid]), k + 8,
                                 top_meta, top_levels, lane, top_lut);
            }
            if (true) {
                a1 = weight_pair(reinterpret_cast<const std::uint8_t*>(code[gid + 8]), k,
                                 bottom_meta, bottom_levels, lane, bottom_lut);
                a3 = weight_pair(reinterpret_cast<const std::uint8_t*>(code[gid + 8]), k + 8,
                                 bottom_meta, bottom_levels, lane, bottom_lut);
            }
            #pragma unroll
            for (int t = 0; t < nt; ++t) {
                unsigned b0,b1;
                const int br=t*8+b_rin;
                const auto addr=static_cast<unsigned>(__cvta_generic_to_shared(&inputs[warp][br][(ks+b_koff)^((br&7)*8)]));
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(b0),"=r"(b1):"r"(addr));
                mma(acc[t], a0, a1, a2, a3, b0, b1);
            }
        }
        }
        __syncthreads();
    }
    if(warp&1){
        #pragma unroll
        for(int t=0;t<nt;++t)*reinterpret_cast<float4*>(&partial[warp][t][lane][0])=make_float4(acc[t][0],acc[t][1],acc[t][2],acc[t][3]);
    }
    __syncthreads();
    if(!(warp&1)){
        #pragma unroll
        for(int t=0;t<nt;++t){
            float4 v=*reinterpret_cast<float4*>(&partial[warp+1][t][lane][0]);
            acc[t][0]=__fadd_rn(acc[t][0],v.x);acc[t][1]=__fadd_rn(acc[t][1],v.y);
            acc[t][2]=__fadd_rn(acc[t][2],v.z);acc[t][3]=__fadd_rn(acc[t][3],v.w);
            if(warp!=0)*reinterpret_cast<float4*>(&partial[warp][t][lane][0])=make_float4(acc[t][0],acc[t][1],acc[t][2],acc[t][3]);
        }
    }
    __syncthreads();
    if(warp==0){
        #pragma unroll
        for(int t=0;t<nt;++t){
            #pragma unroll
            for(int w=2;w<WARPS;w+=2){
                float4 v=*reinterpret_cast<float4*>(&partial[w][t][lane][0]);
                acc[t][0]=__fadd_rn(acc[t][0],v.x);acc[t][1]=__fadd_rn(acc[t][1],v.y);
                acc[t][2]=__fadd_rn(acc[t][2],v.z);acc[t][3]=__fadd_rn(acc[t][3],v.w);
            }
            #pragma unroll
            for(int j=0;j<4;++j){
                const int n=row0+lane/4+(j/2)*8,m=t*8+(lane%4)*2+j%2;
                if(n<OUTPUT_N&&m<ROWS)output[m*OUTPUT_N+n]=rounded(acc[t][j]);
            }
        }
    }
}
