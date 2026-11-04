#include "decode.cuh"

extern "C" __global__ void ggml_matmul(
    const scalar* __restrict__ x, const unsigned char* __restrict__ packed,
    const unsigned char* __restrict__ grid, scalar* __restrict__ output,
    const unsigned char* __restrict__ activity,
    const int* __restrict__ counts, const int* __restrict__ starts,
    const int* __restrict__ block_starts, const int* __restrict__ total_blocks,
    const int* __restrict__ row_map,
    int m, int n, int k, int row_bytes, int type, int experts, int input_top_k,
    long long output_stride, int activity_bytes, int column_offset) {
    namespace wmma = nvcuda::wmma;
    int expert = 0, row_begin = int(blockIdx.y)*tile_rows;
    if (counts) {
        if (int(blockIdx.y) >= total_blocks[0]) return;
        int lo = 0, hi = experts;
        while (lo + 1 < hi) {
            const int middle = (lo + hi)/2;
            if (block_starts[middle] <= int(blockIdx.y)) lo = middle; else hi = middle;
        }
        expert = lo;
        row_begin = (int(blockIdx.y) - block_starts[expert])*tile_rows;
    }
    #if !GGML_VECTOR_ROWS
    constexpr int transport_k = GGML_TRANSPORT_K;
    constexpr int shared_stride = transport_k + 8;
    __shared__ __align__(32) scalar a[16*shared_stride];
    __shared__ __align__(32) scalar b[4][16*shared_stride];
    __shared__ __align__(32) float c[4][16*16];
    #endif
    __shared__ int routes[tile_rows];
    const int warp = threadIdx.x/32;
    if (threadIdx.x < tile_rows) {
        const int row = row_begin + threadIdx.x;
        int route = counts ? (row < counts[expert] ? row_map[starts[expert] + row] : -1)
                           : (row < m ? row : -1);
        if (route >= 0 && activity) {
            const int input = route/input_top_k;
            const bool owned = activity_bytes == 4
                ? reinterpret_cast<const int*>(activity)[input] > 0 : activity[input] != 0;
            if (!owned) route = -1;
        }
        routes[threadIdx.x] = route;
    }
    __syncthreads();
    #if GGML_VECTOR_ROWS
    // The upstream MMVQ organization reuses one decoded fragment across rows.
    // This variant keeps A16 operands rather than introducing Q8_1 activation
    // rounding; each output column has one fixed FP32 warp reduction.
    constexpr int column_warps = GGML_COLUMN_WARPS;
    const int column = int(blockIdx.x)*(4/column_warps) + warp/column_warps;
    if (column >= n) return;
    const int lane = threadIdx.x % 32;
    #if GGML_COLUMN_WARPS > 1
    __shared__ float partials[tile_rows][4];
    #endif
    const auto* weight = packed + (std::int64_t(expert)*n + column)*row_bytes;
    float acc[tile_rows] = {};
    for (int first = ((warp%column_warps)*32 + lane)*8; first < k;
         first += column_warps*32*8) {
        scalar fragment[8];
        original_fragment8(weight, first + column_offset, GGML_WEIGHT_TYPE,
                           grid, min(8, k - first), fragment);
        #pragma unroll
        for (int element = 0; element < 8; ++element) {
            const int kk = first + element;
            if (kk < k) {
                const float w = float(fragment[element]);
                #pragma unroll
                for (int row = 0; row < tile_rows; ++row) {
                    const int route = routes[row];
                    if (route >= 0) {
                        const float value = float(x[(std::int64_t(route)/input_top_k)*k + kk]);
                        acc[row] = __fmaf_rn(w, value, acc[row]);
                    }
                }
            }
        }
    }
    #pragma unroll
    for (int row = 0; row < tile_rows; ++row) {
        #pragma unroll
        for (int delta = 16; delta; delta >>= 1)
            acc[row] = __fadd_rn(acc[row], __shfl_down_sync(0xffffffff, acc[row], delta));
        #if GGML_COLUMN_WARPS > 1
        if (lane == 0) partials[row][warp] = acc[row];
        __syncthreads();
        float complete = 0.0f;
        #pragma unroll
        for (int part = 0; part < column_warps; ++part)
            complete = __fadd_rn(complete, partials[row][part]);
        if (threadIdx.x == 0 && routes[row] >= 0)
            output[std::int64_t(routes[row])*output_stride + column] = rounded(complete);
        #else
        if (lane == 0 && routes[row] >= 0)
            output[std::int64_t(routes[row])*output_stride + column] = rounded(acc[row]);
        #endif
    }
    #else
    wmma::fragment<wmma::accumulator,16,16,16,float> acc;
    wmma::fill_fragment(acc, 0.0f);
    for (int first = 0; first < k; first += transport_k) {
        for (int p = threadIdx.x; p < 16*transport_k; p += blockDim.x) {
            const int route = routes[p/transport_k];
            const int kk = first + p%transport_k;
            a[(p/transport_k)*shared_stride + p%transport_k] = route < 0 || kk >= k ? rounded(0.0f)
                 : x[(std::int64_t(route)/input_top_k)*k + kk];
        }
        // One warp reads consecutive fragments of one original weight row.
        // The K-contiguous shared layout is consumed as column-major B, so
        // transport is coalesced without a decoded global matrix or transpose.
        for (int packet = threadIdx.x; packet < 64*transport_k/8; packet += blockDim.x) {
            const int local_column = packet/(transport_k/8);
            const int w = local_column/16;
            const int column = int(blockIdx.x)*64 + local_column;
            const int local_k = (packet%(transport_k/8))*8;
            const int kk = first + local_k;
            const auto* row = packed + (std::int64_t(expert)*n + column)*row_bytes;
            scalar fragment[8];
            if (column < n && kk < k) {
                original_fragment8(row, kk + column_offset, GGML_WEIGHT_TYPE,
                                   grid, min(8, k - kk), fragment);
            } else {
                #pragma unroll
                for (int i = 0; i < 8; ++i) fragment[i] = rounded(0.0f);
            }
            #pragma unroll
            for (int i = 0; i < 8; ++i)
                b[w][(local_column%16)*shared_stride + local_k + i] = fragment[i];
        }
        __syncthreads();
        #pragma unroll
        for (int inner = 0; inner < transport_k; inner += 16) {
            wmma::fragment<wmma::matrix_a,16,16,16,scalar,wmma::row_major> af;
            wmma::fragment<wmma::matrix_b,16,16,16,scalar,wmma::col_major> bf;
            wmma::load_matrix_sync(af, a + inner, shared_stride);
            wmma::load_matrix_sync(bf, b[warp] + inner, shared_stride);
            wmma::mma_sync(acc, af, bf, acc);
        }
        __syncthreads();
    }
    wmma::store_matrix_sync(c[warp], acc, 16, wmma::mem_row_major);
    __syncthreads();
    for (int p = threadIdx.x; p < 1024; p += blockDim.x) {
        const int w = p/256, local = p%256;
        const int column = int(blockIdx.x)*64 + w*16 + local%16;
        const int route = routes[local/16];
        if (route >= 0 && column < n)
            output[std::int64_t(route)*output_stride + column] = rounded(c[w][local]);
    }
    #endif
}
