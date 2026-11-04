// SPDX-License-Identifier: MIT
// Lossless Q8_0 transport. Every expert owns its original logical N axis.
// Codes: [ceil(N/16)*16, K/32, 8] int32 in original logical row order.
// Scales follow the codes: [ceil(N/16)*16, K/32] half.
#pragma once
#include <cuda_fp16.h>
#include <stdint.h>

static __device__ __forceinline__ int q8_packed_word(
    const unsigned char* weight, int n, int chunks, int row, int kb, int word) {
    return reinterpret_cast<const int*>(weight)[(int64_t(row) * chunks + kb) * 8 + word];
}

static __device__ __forceinline__ half q8_packed_scale(
    const unsigned char* weight, int n, int chunks, int row, int kb) {
    const int nt = (n + 15) / 16;
    const auto* scales = reinterpret_cast<const half*>(weight + int64_t(chunks) * nt * 512);
    return scales[int64_t(row) * chunks + kb];
}

static __device__ __forceinline__ const unsigned char* q8_packed_expert(
    const unsigned char* weight, int n, int chunks, int expert) {
    return weight + int64_t(expert) * chunks * ((n + 15) / 16) * 544;
}
