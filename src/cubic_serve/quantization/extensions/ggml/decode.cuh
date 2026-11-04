#pragma once
// Original GGML blocks -> register/shared 16-bit operands -> FP32 tensor-core dot.
// Portable SM80 baseline. Geometry, format and expert ownership are runtime inputs.
// No full decoded weight allocation, activation quantization or model dispatch.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cstdint>

#ifndef GGML_VECTOR_ROWS
#define GGML_VECTOR_ROWS 0
#endif
constexpr int tile_rows = GGML_VECTOR_ROWS ? GGML_VECTOR_ROWS : 16;

#if IS_BF16
using scalar = __nv_bfloat16;
__device__ __forceinline__ scalar rounded(float x) { return __float2bfloat16_rn(x); }
#else
using scalar = __half;
__device__ __forceinline__ scalar rounded(float x) { return __float2half_rn(x); }
#endif

__device__ __forceinline__ unsigned short u16(const unsigned char* p) {
    return unsigned(p[0]) | (unsigned(p[1]) << 8);
}
__device__ __forceinline__ float half_value(const unsigned char* p) {
    return __half2float(__ushort_as_half(u16(p)));
}

__device__ __forceinline__ int iq4_level(unsigned code) {
    // Same immutable GGML table, held in registers. A dynamically indexed
    // function-local array otherwise generates local-memory stores/loads even
    // for other formats in the shared decoding loop.
    const unsigned low = code < 4 ? 0xbfad9881u : 0xf6eaddcfu;
    const unsigned high = code < 12 ? 0x26190d01u : 0x71594535u;
    const unsigned word = code < 8 ? low : high;
    return static_cast<signed char>(word >> (8*(code & 3)));
}

__device__ __forceinline__ float original_weight(
    const unsigned char* row, int k, int type, const unsigned char* grid) {
    if (type == 0) return reinterpret_cast<const float*>(row)[k];
    if (type == 1) return half_value(row + 2LL*k);
    if (type == 30) return __bfloat162float(__ushort_as_bfloat16(u16(row + 2LL*k)));
    if (type == 8) {
        const auto* b = row + (k / 32)*34LL;
        return __fmul_rn(half_value(b), float(reinterpret_cast<const signed char*>(b + 2)[k % 32]));
    }
    if (type == 20) {
        const auto* b = row + (k / 32)*18LL;
        const int p = k % 32;
        const int q = (b[2 + p%16] >> (4*(p/16))) & 15;
        return __fmul_rn(half_value(b), float(iq4_level(q)));
    }
    const int p = k % 256;
    if (type == 14) {
        const auto* b = row + (k / 256)*210LL;
        const int half = p / 128, quarter = (p % 128) / 32, lane = p % 32;
        const int low = (b[half*64 + (quarter%2)*32 + lane] >> ((quarter/2)*4)) & 15;
        const int high = (b[128 + half*32 + lane] >> (quarter*2)) & 3;
        const int scale = reinterpret_cast<const signed char*>(b + 192)[p/16];
        return __fmul_rn(__fmul_rn(half_value(b + 208), float(scale)), float((low | (high << 4)) - 32));
    }
    if (type == 23) {
        const auto* b = row + (k / 256)*136LL;
        const int group = p / 32;
        const int low = (b[4 + group/2] >> (4*(group%2))) & 15;
        const int high = (u16(b + 2) >> (2*group)) & 3;
        const int q = (b[8 + group*16 + p%16] >> (4*((p%32)/16))) & 15;
        return __fmul_rn(__fmul_rn(half_value(b), float((low | (high << 4)) - 32)), float(iq4_level(q)));
    }
    // IQ3_S: 9-bit four-element grid index, individual signs, odd group scale.
    const auto* b = row + (k / 256)*110LL;
    const int vector = p/4, group = p/32;
    const int index = b[2 + vector] | (((b[66 + vector/8] >> (vector%8)) & 1) << 8);
    const int scale = 1 + 2*((b[106 + group/2] >> (4*(group%2))) & 15);
    const int sign = (b[74 + p/8] >> (p%8)) & 1;
    const float magnitude = __fmul_rn(__fmul_rn(half_value(b), float(scale)), float(grid[index*4 + p%4]));
    return sign ? -magnitude : magnitude;
}

__device__ __forceinline__ void original_fragment8(
    const unsigned char* row, int k, int type, const unsigned char* grid,
    int count, scalar (&values)[8]) {
    if (type == 8 && (k & 7) == 0 && count == 8) {
        const auto* b = row + (k / 32)*34LL;
        const float d = half_value(b);
        const auto* codes = reinterpret_cast<const signed char*>(b + 2 + k%32);
        #pragma unroll
        for (int i = 0; i < 8; ++i)
            values[i] = rounded(__fmul_rn(d, float(codes[i])));
        return;
    }
    // IQ3_S owns an eight-value sign packet and two four-value codebook words.
    // Decode that packet once, as upstream getrows.cu does, retaining the exact
    // original multiplication order and the same 16-bit operand boundary.
    if (type == 21 && (k & 7) == 0 && count == 8) {
        const auto* b = row + (k / 256)*110LL;
        const int p = k % 256, vector = p/4, group = p/32;
        const unsigned high = b[66 + vector/8];
        const unsigned index0 = b[2 + vector] | (((high >> (vector%8)) & 1) << 8);
        const unsigned index1 = b[3 + vector] | (((high >> ((vector+1)%8)) & 1) << 8);
        const unsigned grid0 = reinterpret_cast<const unsigned*>(grid)[index0];
        const unsigned grid1 = reinterpret_cast<const unsigned*>(grid)[index1];
        const int scale = 1 + 2*((b[106 + group/2] >> (4*(group%2))) & 15);
        const float d = __fmul_rn(half_value(b), float(scale));
        const unsigned signs = b[74 + p/8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const unsigned packed_grid = i < 4 ? grid0 : grid1;
            const float value = __fmul_rn(d, float((packed_grid >> (8*(i & 3))) & 255));
            values[i] = rounded((signs >> i) & 1 ? -value : value);
        }
    } else {
        #pragma unroll
        for (int i = 0; i < 8; ++i)
            values[i] = i < count ? rounded(original_weight(row, k + i, type, grid)) : rounded(0.0f);
    }
}
