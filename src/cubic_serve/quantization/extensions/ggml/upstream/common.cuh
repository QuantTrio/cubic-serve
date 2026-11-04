#pragma once
// Cubic binding shim: upstream block layouts and pointwise integer dots are
// unchanged. No GGML allocator, execution graph or architecture selector.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>
#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"
static __device__ __forceinline__ int ggml_cuda_dp4a(int a, int b, int c) {
    return __dp4a(a, b, c);
}
static __device__ __forceinline__ float ggml_cuda_e8m0_to_fp32(uint8_t x) {
#if CUDART_VERSION >= 12080
    const nv_bfloat16 e = __nv_cvt_e8m0_to_bf16raw(x);
    return (float) e;
#else
    uint32_t bits;
    if (x == 0) {
        bits = 0x00400000;
    } else {
        bits = (uint32_t) x << 23;
    }

    float result;
    memcpy(&result, &bits, sizeof(float));
    return result;
#endif // CUDART_VERSION >= 12050
}

static __device__ __forceinline__ float ggml_cuda_ue4m3_to_fp32(uint8_t x) {
#if defined(GGML_USE_HIP) && defined(CDNA3) && defined(FP8_AVAILABLE) && HIP_VERSION >= 60200000
    // ROCm does not support fp8 in software on devices with fp8 hardware,
    // but CDNA3 supports only e4m3_fnuz (no inf).
    const uint32_t bits = x * (x != 0x7F && x != 0xFF); // Convert NaN to 0.0f to match CPU implementation.
    const __hip_fp8_e4m3_fnuz xf = *reinterpret_cast<const __hip_fp8_e4m3_fnuz *>(&bits);
    return static_cast<float>(xf) / 2;
#else
#if defined(FP8_AVAILABLE) && !defined(GGML_USE_HIP)
    const uint32_t bits = x * (x != 0x7F && x != 0xFF); // Convert NaN to 0.0f to match CPU implementation.
    const __nv_fp8_e4m3 xf = *reinterpret_cast<const __nv_fp8_e4m3 *>(&bits);
    return static_cast<float>(xf) / 2;
#else
    if (x == 0 || (x == 0x7F && x != 0xFF)) { // Convert NaN to 0.0f
        return 0.0f;
    }
    const int exp = (x >> 3) & 0xF;
    const int man = x & 0x7;
    float raw;
    if (exp == 0) {
        raw = ldexpf((float) man, -9);
    } else {
        raw = ldexpf(1.0f + (float) man / 8.0f, exp - 7);
    }
    return static_cast<float>(raw / 2);
#endif // defined(FP8_AVAILABLE) && !defined(GGML_USE_HIP)
#endif // defined(GGML_USE_HIP) && defined(CDNA3) && defined(FP8_AVAILABLE) && HIP_VERSION >= 60200000
}
