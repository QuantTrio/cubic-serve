// Packed row-local SIMT contraction. The format owns decoder.cuh; this file
// owns only packed transport, activation reuse, FP32 FMA and ordered reduction.
// Pipeline structure derived from T-Crypt/ninfer-4090 (Apache-2.0).
#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <numeric>

template<int RawBytes>
__device__ __forceinline__ void packed_group_copy(std::uint8_t* target,
    const std::uint8_t* original, int column, int tile, int lane, int input_k) {
    constexpr int bytes = TILE_K / GI * RawBytes;
    constexpr int vector = METADATA_VECTOR_BYTES;
    const auto* source = original +
        (std::int64_t(column) * (input_k / GI) + tile * (TILE_K / GI)) * RawBytes;
    for (int v = lane; v < bytes / vector; v += 32) {
        if constexpr (vector == 2)
            reinterpret_cast<unsigned short*>(target)[v] = reinterpret_cast<const unsigned short*>(source)[v];
        else __pipeline_memcpy_async(target + v * vector, source + v * vector, vector);
    }
}

struct RowGeometry { int n, k; long long zero_offset, raw_offset; int descriptors; };
#include "decoder.cuh"

constexpr int tile_k = TILE_K, stages = 2;
constexpr int code_vectors = tile_k * BITS / 128;
constexpr int metadata_groups = tile_k / GI;
constexpr int raw_metadata_vectors = (metadata_groups * FORMAT_RAW_BYTES + 15) / 16;
constexpr int staging_bytes = WARPS * MEMBERS *
    (stages * (code_vectors * 16 + raw_metadata_vectors * 16) + metadata_groups * 8);
constexpr bool full_x = X_STORAGE == 0;
constexpr bool tiled_x = X_STORAGE == 1;
static_assert(GI >= ELEMENTS && tile_k % GI == 0);

#if IS_BF16
using scalar = __nv_bfloat16;
#else
using scalar = __half;
#endif
__device__ __forceinline__ scalar scalar_round(float value) {
#if IS_BF16
    return __float2bfloat16_rn(value);
#else
    return __float2half_rn(value);
#endif
}

template<int Count>
__device__ __forceinline__ void copy(void* to, const void* from) {
    __pipeline_memcpy_async(to, from, Count);
}

__device__ __forceinline__ float reduce(float value) {
    #pragma unroll
    for (int delta = 16; delta; delta >>= 1)
        value = __fadd_rn(value, __shfl_down_sync(0xffffffff, value, delta));
    return value;
}

constexpr int fragment_span = ELEMENTS * BITS + 16 - std::gcd(ELEMENTS * BITS, 16);
using Fragment = std::conditional_t<fragment_span <= 32, unsigned, unsigned long long>;
__device__ __forceinline__ Fragment packed_fragment(const unsigned short* halves, int index) {
    const int bit = index * BITS, half = bit / 16, shift = bit % 16;
    if constexpr (ELEMENTS * BITS == 32) {
        return reinterpret_cast<const unsigned*>(halves)[index * BITS / 32];
    } else if constexpr (ELEMENTS * BITS == 64) {
        return reinterpret_cast<const unsigned long long*>(halves)[index * BITS / 64];
    } else {
        // Only this register fragment widens; global/shared codes retain BITS.
        Fragment value = halves[half];
        #pragma unroll
        for (int i = 1; i < (ELEMENTS * BITS + 15) / 16 + 1; ++i)
            if (16 * i < ELEMENTS * BITS + shift)
                value |= static_cast<Fragment>(halves[half + i]) << (16 * i);
        return value >> shift;
    }
}

__device__ __forceinline__ float2 scalar_pair(unsigned word) {
#if IS_BF16
    union { unsigned u; __nv_bfloat162 values; } bits;
    bits.u = word;
    return __bfloat1622float2(bits.values);
#else
    union { unsigned u; __half2 values; } bits;
    bits.u = word;
    return __half22float2(bits.values);
#endif
}

extern "C" __global__ void packed_small_rows(
    const scalar* __restrict__ x,
    const std::uint8_t* __restrict__ codes0, const std::uint8_t* __restrict__ codes1,
    const std::uint8_t* __restrict__ metadata0, const std::uint8_t* __restrict__ metadata1,
    scalar* __restrict__ out0, scalar* __restrict__ out1,
    int output_n, int input_k, long long stride0, long long stride1, int partitions,
    long long zero_offset, long long raw_offset, int descriptors) {
    const RowGeometry geometry{output_n, input_k, zero_offset, raw_offset, descriptors};
    const int partition_shift = __ffs(partitions)-1;
    const int columns_per_block = WARPS >> partition_shift;
    const int packed_row_bytes = input_k * BITS / 8;
    const int tile_count = input_k / tile_k;
    __shared__ uint4 code[WARPS][stages][MEMBERS][code_vectors];
    __shared__ uint4 raw_metadata[WARPS][stages][MEMBERS][raw_metadata_vectors];
    __shared__ float2 metadata[WARPS][MEMBERS][metadata_groups];
    extern __shared__ __align__(16) scalar all_x[];
    __shared__ __align__(16) scalar part_x[stages][tiled_x ? tile_k * ROWS : 1];
    __shared__ float partials[WARPS][MEMBERS][ROWS];
    const int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
    const int partition = warp & (partitions-1);
    const int column = blockIdx.x * columns_per_block + (warp >> partition_shift);
    const bool owned = column < output_n;
    const int begin = (tile_count * partition) >> partition_shift;
    const int end = (tile_count * (partition + 1)) >> partition_shift;
    const std::uint8_t* codes[2] = {codes0, codes1};
    const std::uint8_t* metas[2] = {metadata0, metadata1};
    scalar* outputs[2] = {out0, out1};
    const long long output_strides[2] = {stride0, stride1};
    FormatContext contexts[MEMBERS];
    #pragma unroll
    for (int member = 0; member < MEMBERS; ++member)
        contexts[member] = format_context(metas[member], column, geometry);
    if constexpr (full_x) {
        for (int v = threadIdx.x; v < input_k * ROWS / 8; v += blockDim.x)
            reinterpret_cast<uint4*>(all_x)[v] = reinterpret_cast<const uint4*>(x)[v];
        __syncthreads();
    }
    auto issue = [&](int tile) {
        const int buffer = tile % stages;
        if (owned) {
            #pragma unroll
            for (int member = 0; member < MEMBERS; ++member) {
                const auto* row = codes[member] + std::int64_t(column) * packed_row_bytes;
                for (int v = lane; v < code_vectors; v += 32)
                    copy<16>(&code[warp][buffer][member][v],
                        reinterpret_cast<const uint4*>(row + tile * tile_k * BITS / 8) + v);
                format_copy(reinterpret_cast<std::uint8_t*>(raw_metadata[warp][buffer][member]),
                            metas[member], column, tile, lane, geometry);
            }
        }
        if constexpr (tiled_x) {
            for (int v = threadIdx.x; v < tile_k * ROWS / 8; v += blockDim.x) {
                const int row = v / (tile_k / 8), col = v % (tile_k / 8);
                copy<16>(reinterpret_cast<uint4*>(part_x[buffer]) + v,
                    reinterpret_cast<const uint4*>(x + row * input_k + tile * tile_k) + col);
            }
        }
        __pipeline_commit();
    };
    issue(begin);
    float sums[MEMBERS][ROWS] = {};
    #pragma unroll 1
    for (int tile = begin; tile < end; ++tile) {
        if (tile + 1 < end) issue(tile + 1);
        else __pipeline_commit();
        __pipeline_wait_prior(1);
        if constexpr (tiled_x) __syncthreads(); else __syncwarp();
        const int buffer = tile % stages;
        if (owned) {
            #pragma unroll
            for (int member = 0; member < MEMBERS; ++member)
                format_expand(metadata[warp][member],
                    reinterpret_cast<const std::uint8_t*>(raw_metadata[warp][buffer][member]),
                    metas[member], contexts[member], column, tile, lane);
            __syncwarp();
            #pragma unroll
            for (int first = 0; first < tile_k; first += 32 * ELEMENTS) {
                const int offset = first + lane * ELEMENTS, group = offset / GI;
                FormatLevels levels[MEMBERS];
                float2 data[MEMBERS];
                Fragment packed[MEMBERS];
                #pragma unroll
                for (int member = 0; member < MEMBERS; ++member) {
                    data[member] = metadata[warp][member][group];
                    levels[member] = format_levels(data[member], lane);
                    packed[member] = packed_fragment(
                        reinterpret_cast<const unsigned short*>(code[warp][buffer][member]), offset);
                }
                float2 input[ROWS][ELEMENTS / 2];
                #pragma unroll
                for (int row = 0; row < ROWS; ++row) {
                    const scalar* source = tiled_x ? part_x[buffer] + row * tile_k + offset :
                        (full_x ? all_x : x) + row * input_k + tile * tile_k + offset;
                    if constexpr (ELEMENTS == 8) {
                        const uint4 words = *reinterpret_cast<const uint4*>(source);
                        input[row][0] = scalar_pair(words.x); input[row][1] = scalar_pair(words.y);
                        input[row][2] = scalar_pair(words.z); input[row][3] = scalar_pair(words.w);
                    } else if constexpr (ELEMENTS == 4) {
                        const uint2 words = *reinterpret_cast<const uint2*>(source);
                        input[row][0] = scalar_pair(words.x); input[row][1] = scalar_pair(words.y);
                    } else input[row][0] = scalar_pair(*reinterpret_cast<const unsigned*>(source));
                }
                #pragma unroll
                for (int e = 0; e < ELEMENTS; ++e) {
                    float weights[MEMBERS];
                    #pragma unroll
                    for (int member = 0; member < MEMBERS; ++member)
                        weights[member] = format_decode((packed[member] >> (e * BITS)) & ((1u << BITS) - 1),
                            levels[member], data[member], lane);
                    #pragma unroll
                    for (int row = 0; row < ROWS; ++row) {
                        const float value = (e & 1) ? input[row][e / 2].y : input[row][e / 2].x;
                        #pragma unroll
                        for (int member = 0; member < MEMBERS; ++member)
                            sums[member][row] = __fmaf_rn(weights[member], value, sums[member][row]);
                    }
                }
            }
        }
        if constexpr (tiled_x) __syncthreads(); else __syncwarp();
    }
    #pragma unroll
    for (int member = 0; member < MEMBERS; ++member) {
        #pragma unroll
        for (int row = 0; row < ROWS; ++row) {
            const float value = reduce(sums[member][row]);
            if (partitions == 1) {
                if (owned && lane == 0) outputs[member][row * output_strides[member] + column] = scalar_round(value);
            } else if (lane == 0) partials[warp][member][row] = value;
        }
    }
    if (partitions > 1) {
        __syncthreads();
        if (owned && lane == 0 && partition == 0) {
            #pragma unroll
            for (int member = 0; member < MEMBERS; ++member) {
                #pragma unroll
                for (int row = 0; row < ROWS; ++row) {
                    float value = partials[warp][member][row];
                    #pragma unroll
                    for (int p = 1; p < partitions; ++p)
                        value = __fadd_rn(value, partials[warp + p][member][row]);
                    outputs[member][row * output_strides[member] + column] = scalar_round(value);
                }
            }
        }
    }
}
