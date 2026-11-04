// SPDX-License-Identifier: MIT
// llama.cpp 0c1e57098bba43ac29e6e3b677cdceebdd22334f; see manifest.json.
#pragma once
#include "vecdotq.cuh"
// Extracted from pinned llama.cpp; same scalar/block/scale arithmetic.
// Uniform SM80+ integer MMA baseline; host/SKU selectors are not imported.
using ggml_type = int;
constexpr int GGML_TYPE_Q8_0=8, GGML_TYPE_Q6_K=14, GGML_TYPE_IQ4_NL=20, GGML_TYPE_IQ3_S=21, GGML_TYPE_IQ4_XS=23;
constexpr int MMQ_ITER_K=256, MMQ_TILE_NE_K=32, MMQ_TILE_Y_K=36;
enum mmq_q8_1_ds_layout { MMQ_Q8_1_DS_LAYOUT_D4, MMQ_Q8_1_DS_LAYOUT_DS4 };
constexpr __device__ int ggml_cuda_get_physical_warp_size() { return 32; }
constexpr __device__ int ggml_cuda_mmq_get_nthreads(int,int,bool) { return 128; }
constexpr __device__ int ggml_cuda_mmq_get_I(int,int,bool) { return 64; }
constexpr __device__ int ggml_cuda_mmq_get_rows_per_warp(int,int,bool) { return 16; }
constexpr __device__ int ggml_cuda_mmq_get_sram_stride(int type,int,bool) { return 76; }
enum data_layout { DATA_LAYOUT_I_MAJOR = 0 };
    template <int I_, int J_, typename T, data_layout ds_=DATA_LAYOUT_I_MAJOR>
    struct tile {};
    template <int I_, int J_, typename T>
    struct tile<I_, J_, T, DATA_LAYOUT_I_MAJOR> {
        static constexpr int I = I_;
        static constexpr int J = J_;
        static constexpr data_layout dl = DATA_LAYOUT_I_MAJOR;
        static constexpr int ne = I * J / 32;
        T x[ne] = {0};
        static constexpr __device__ bool supported() {
            if (I == 8 && J == 4) return true;
            if (I == 8 && J == 8) return true;
            if (I == 16 && J == 8) return true;
            if (I == 16 && J == 16) return true;
            if (I == 32 && J == 8) return true;
            return false;
        }
        static __device__ __forceinline__ int get_i(const int l) {
            if constexpr (I == 8 && J == 4) {
                return threadIdx.x / 4;
            } else if constexpr (I == 8 && J == 8) {
                return threadIdx.x / 4;
            } else if constexpr (I == 16 && J == 8) {
                return ((l / 2) * 8) + (threadIdx.x / 4);
            } else if constexpr (I == 16 && J == 16) {
                return (((l / 2) % 2) * 8) + (threadIdx.x / 4);
            } else if constexpr (I == 32 && J == 8) {
                return tile<16, 8, T>::get_i(l);
            } else {
                asm("trap;");
                return -1;
            }
        }
        static __device__ __forceinline__ int get_j(const int l) {
            if constexpr (I == 8 && J == 4) {
                return threadIdx.x % 4;
            } else if constexpr (I == 8 && J == 8) {
                return (l * 4) + (threadIdx.x % 4);
            } else if constexpr (I == 16 && J == 8) {
                return ((threadIdx.x % 4) * 2) + (l % 2);
            } else if constexpr (I == 16 && J == 16) {
                return ((l / 4) * 8) + ((threadIdx.x % 4) * 2) + (l % 2);
            } else if constexpr (I == 32 && J == 8) {
                return tile<16, 8, T>::get_j(l);
            } else {
                asm("trap;");
                return -1;
            }
        }
    };
template <int I, int J, typename T, data_layout dl>
    static __device__ __forceinline__ void load_generic(tile<I, J, T, dl> & t, const T * __restrict__ xs0, const int stride) {
#pragma unroll
        for (int l = 0; l < t.ne; ++l) {
            t.x[l] = xs0[t.get_i(l)*stride + t.get_j(l)];
        }
    }
    template <typename T, data_layout dl>
    static __device__ __forceinline__ void load_ldmatrix(
            tile<16, 4, T, dl> & t, const T * __restrict__ xs0, const int stride) {
        int * xi = (int *) t.x;
        const int * xs = (const int *) xs0 + (threadIdx.x % t.I) * stride;
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];"
            : "=r"(xi[0]), "=r"(xi[1])
            : "l"(xs));
    }

    template <typename T, data_layout dl>
    static __device__ __forceinline__ void load_ldmatrix(
            tile<16, 8, T, dl> & t, const T * __restrict__ xs0, const int stride) {
        int * xi = (int * ) t.x;
        const int * xs = (const int *) xs0 + (threadIdx.x % t.I) * stride + (threadIdx.x / t.I) * (t.J / 2);
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
            : "=r"(xi[0]), "=r"(xi[1]), "=r"(xi[2]), "=r"(xi[3])
            : "l"(xs));
    }

    static __device__ __forceinline__ void mma(
            tile<16, 8, int> & D, const tile<16, 4, int> & A, const tile<8, 4, int> & B) {
        asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
            : "+r"(D.x[0]), "+r"(D.x[1]), "+r"(D.x[2]), "+r"(D.x[3])
            : "r"(A.x[0]), "r"(A.x[1]), "r"(B.x[0]));
    }

    static __device__ __forceinline__ void mma(
            tile<16, 8, int> & D, const tile<16, 8, int> & A, const tile<8, 8, int> & B) {
        asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
            : "+r"(D.x[0]), "+r"(D.x[1]), "+r"(D.x[2]), "+r"(D.x[3])
            : "r"(A.x[0]), "r"(A.x[1]), "r"(A.x[2]), "r"(A.x[3]), "r"(B.x[0]), "r"(B.x[1]));
    }

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_q8_0(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);
    int * x_qs = (int *) x_tile;
    float * x_df = (float *) (x_tile + 2*MMQ_TILE_NE_K);
    constexpr int threads_per_row = 32;
    constexpr int nrows = warp_size / threads_per_row;
    const int txi = warp_size > threads_per_row ? threadIdx.x % threads_per_row : threadIdx.x;
    const int kbx = txi / QI8_0;
    const int kqsx = txi % QI8_0;
#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nrows*nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row);
        if (fallback) {
            i = min(i, i_max);
        }
        const block_q8_0 * bxi = (const block_q8_0 *) x + kbx0 + i*stride + kbx;
        x_qs[i*sram_stride + 0 + txi] = get_int_b2(bxi[0].qs, kqsx);
        x_qs[i*sram_stride + MMQ_TILE_NE_K + txi] = get_int_b2(bxi[MMQ_TILE_NE_K/QI8_0].qs, kqsx);
    }
    constexpr int blocks_per_tile_x_row = 2*MMQ_TILE_NE_K / QI8_0;
    constexpr int rows_per_warp = warp_size / blocks_per_tile_x_row;
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;
#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * rows_per_warp) {
        int i = i0 + threadIdx.y * rows_per_warp + threadIdx.x / blocks_per_tile_x_row;
        if (fallback) {
            i = min(i, i_max);
        }
        const block_q8_0 * bxi = (const block_q8_0 *) x + kbx0 + i*stride + kbxd;
        x_df[i*sram_stride + kbxd] = bxi->d;
    }
}
template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_q6_K(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);
    int * x_qs = (int *) x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);
    int * x_sc = (int *) (x_df + MMQ_TILE_NE_K/QI6_K);
    constexpr int threads_per_row = MMQ_ITER_K / (4 * QR6_K);
    constexpr int nrows = warp_size / threads_per_row;
    const int txi = warp_size > threads_per_row ? threadIdx.x % threads_per_row : threadIdx.x;
#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nrows*nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row);
        if (fallback) {
            i = min(i, i_max);
        }
        const block_q6_K * bxi = (const block_q6_K *) x + kbx0 + i*stride;
        const int ql = get_int_b2(bxi->ql, txi);
        const int ql0 = (ql >> 0) & 0x0F0F0F0F;
        const int ql1 = (ql >> 4) & 0x0F0F0F0F;
        const int qh = get_int_b2(bxi->qh, (QI6_K/4) * (txi / (QI6_K/2)) + txi % (QI6_K/4));
        const int qh0 = ((qh >> ((txi & 0x08) >> 2)) << 4) & 0x30303030;
        const int qh1 = (qh >> ((txi & 0x08) >> 2)) & 0x30303030;
        const int kq0 = 2*txi - txi % (QI6_K/2) + 0;
        const int kq1 = 2*txi - txi % (QI6_K/2) + QI6_K/2;
        x_qs[i*sram_stride + kq0] = __vsub4(ql0 | qh0, 0x20202020);
        x_qs[i*sram_stride + kq1] = __vsub4(ql1 | qh1, 0x20202020);
    }
#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps*warp_size) {
        int i = (i0 + threadIdx.y*warp_size + threadIdx.x) % I;
        if (fallback) {
            i = min(i, i_max);
        }
        const block_q6_K * bxi = (const block_q6_K *) x + kbx0 + i*stride;
        x_df[i*sram_stride] = bxi->d;
    }
    constexpr int rows_per_warp = warp_size / 4;
#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps*rows_per_warp) {
        int i = (i0 + threadIdx.y*rows_per_warp + threadIdx.x/(MMQ_TILE_NE_K/8)) % I;
        if (fallback) {
            i = min(i, i_max);
        }
        const block_q6_K * bxi = (const block_q6_K *) x + kbx0 + i*stride + (threadIdx.x % (MMQ_TILE_NE_K/8)) / 4;
        x_sc[i*sram_stride + threadIdx.x%4] = get_int_b2(bxi->scales, threadIdx.x % (MMQ_TILE_NE_K/8));
    }
}
template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_iq3_s(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);
    int * x_qs = (int *) x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);
    constexpr int threads_per_row = (MMQ_ITER_K / (4 * QR3_S)) / 2;
    constexpr int nrows = warp_size / threads_per_row;
    const int kqsx = threadIdx.x % threads_per_row;
#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * nrows) {
        int i = i0 + threadIdx.y*nrows + threadIdx.x/threads_per_row;
        if (fallback) {
            i = min(i, i_max);
        }
        const block_iq3_s * bxi = (const block_iq3_s *) x + kbx0 + i*stride;
        const int2 qs_packed = make_int2(get_int_b2(bxi->qs, 2*kqsx+0), get_int_b2(bxi->qs, 2*kqsx+1));
        const uint8_t * qs = (const uint8_t *) &qs_packed;
        const int qh = bxi->qh[kqsx];
        const int signs_packed_32 = get_int_b2(bxi->signs, kqsx);
        const uint8_t * signs_packed_8 = (const uint8_t *) &signs_packed_32;
#pragma unroll
        for (int l = 0; l < QR3_S; ++l) {
            const int2 grid_pos = make_int2(
                iq3s_grid[qs[2*l+0] | ((qh << (8 - 2*l)) & 0x100)],
                iq3s_grid[qs[2*l+1] | ((qh << (7 - 2*l)) & 0x100)]);
            const int signs0 = __vcmpne4(((signs_packed_8[l] & 0x03) << 7) | ((signs_packed_8[l] & 0x0C) << 21), 0x00000000);
            const int signs1 = __vcmpne4(((signs_packed_8[l] & 0x30) << 3) | ((signs_packed_8[l] & 0xC0) << 17), 0x00000000);
            const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
            const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);
            x_qs[i*sram_stride + 8*kqsx + (2*l+0)] = grid_l;
            x_qs[i*sram_stride + 8*kqsx + (2*l+1)] = grid_h;
        }
        const int ls = 1 + 2*((bxi->scales[kqsx/2] >> (((2*kqsx) << 1) & 0x04)) & 0x0F);
        const float d = bxi->d;
        x_df[i*sram_stride + kqsx] = ls*d;
    }
}
template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_iq4_xs(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);
    int * x_qs = (int *) x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);
    constexpr int threads_per_row = MMQ_ITER_K / (4 * QR4_XS);
    constexpr int nrows = warp_size / threads_per_row;
    const int kqsx = threadIdx.x % threads_per_row;
#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nrows*nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row);
        if (fallback) {
            i = min(i, i_max);
        }
        const block_iq4_xs * bxi = (const block_iq4_xs *) x + kbx0 + i*stride;
        const int aux_q4 = get_int_b4(bxi->qs, kqsx);
        const int2 v = get_int_from_table_16(aux_q4, kvalues_iq4nl);
        const int k0 = 8 * (kqsx / 4) + kqsx % 4;
        x_qs[i*sram_stride + k0 + 0] = v.x;
        x_qs[i*sram_stride + k0 + 4] = v.y;
    }
    constexpr int rows_per_warp = warp_size / 8;
#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * rows_per_warp) {
        int i = i0 + threadIdx.y * rows_per_warp + threadIdx.x / (MMQ_TILE_NE_K/4);
        if (fallback) {
            i = min(i, i_max);
        }
        const block_iq4_xs * bxi = (const block_iq4_xs *) x + kbx0 + i*stride;
        const float d = __half2float(bxi->d);
        const int ls = ((bxi->scales_l[(threadIdx.x % 8)/2] >> (4*(threadIdx.x % 2))) & 0x0F)
            | (((bxi->scales_h >> (2*(threadIdx.x % 8))) & 0x03) << 4);
        x_df[i*sram_stride + threadIdx.x % 8] = d * (ls - 32);
    }
}
template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_iq4_nl(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);
    int * x_qs = (int *) x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);
    constexpr int threads_per_row = MMQ_ITER_K / (4 * QR4_NL);
    constexpr int nrows = warp_size / threads_per_row;
    const int txi = warp_size > threads_per_row ? threadIdx.x % threads_per_row : threadIdx.x;
    const int kbx = txi / QI4_NL;
    const int kqsx = txi % QI4_NL;
#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nrows*nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row);
        if (fallback) {
            i = min(i, i_max);
        }
        const block_iq4_nl * bxi = (const block_iq4_nl *) x + kbx0 + i*stride + kbx;
        const int aux_q4 = get_int_b2(bxi->qs, kqsx);
        const int2 v = get_int_from_table_16(aux_q4, kvalues_iq4nl);
        const int k0 = kbx * (2 * QI4_NL) + kqsx;
        x_qs[i*sram_stride + k0 + 0] = v.x;
        x_qs[i*sram_stride + k0 + QI4_NL] = v.y;
    }
    constexpr int blocks_per_tile_x_row = MMQ_TILE_NE_K / QI4_NL;
    constexpr int rows_per_warp = warp_size / blocks_per_tile_x_row;
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;
#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * rows_per_warp) {
        int i = i0 + threadIdx.y * rows_per_warp + threadIdx.x / blocks_per_tile_x_row;
        if (fallback) {
            i = min(i, i_max);
        }
        const block_iq4_nl * bxi = (const block_iq4_nl *) x + kbx0 + i*stride + kbxd;
        x_df[i*sram_stride + kbxd] = __half2float(bxi->d);
    }
}
template <ggml_type type, int J, bool fallback, mmq_q8_1_ds_layout ds_layout>
static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
    typedef tile<16, 8, int> tile_A;
    typedef tile< 8, 8, int> tile_B;
    typedef tile<16, 8, int> tile_C;
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);
    constexpr int rows_per_warp = ggml_cuda_mmq_get_rows_per_warp(type, J, fallback);
    constexpr int ntx = rows_per_warp/tile_C::I;
    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K);
    const int * x_qs = (const int *) x;
    const float * x_df = (const float *) x_qs + 2*MMQ_TILE_NE_K;
    const int * y_qs = (const int *) y + 4;
    const float * y_df = (const float *) y;
    const half2 * y_ds = (const half2 *) y;
    tile_A A[ntx][MMQ_TILE_NE_K/QI8_0];
    float dA[ntx][tile_C::ne/2][MMQ_TILE_NE_K/QI8_0];
    const int i0 = (threadIdx.y/ntx)*rows_per_warp;
#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_0) {
            const int k0 = k00 + k01;
            load_ldmatrix(A[n][k01/QI8_0], x_qs + (i0 + n*tile_A::I)*sram_stride + k0, sram_stride);
        }
#pragma unroll
        for (int l = 0; l < tile_C::ne/2; ++l) {
            const int i = i0 + n*tile_A::I + tile_C::get_i(2*l);
#pragma unroll
            for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_0) {
                const int k0 = k00 + k01;
                dA[n][l][k01/QI8_0] = x_df[i*sram_stride + k0/QI8_0];
            }
        }
    }
#pragma unroll
    for (int j0 = 0; j0 < J; j0 += ntx*tile_C::J) {
#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_0) {
            tile_B B;
            float dB[tile_C::ne/2];
            load_generic(B, y_qs + j0*MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K);
#pragma unroll
            for (int l = 0; l < tile_C::ne/2; ++l) {
                const int j = j0 + tile_C::get_j(l);
                if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D4) {
                    dB[l] = y_df[j*MMQ_TILE_Y_K + k01/QI8_1];
                } else {
                    dB[l] = __low2float(y_ds[j*MMQ_TILE_Y_K + k01/QI8_1]);
                }
            }
#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C C;
                mma(C, A[n][k01/QI8_0], B);
#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += C.x[l]*dA[n][l/2][k01/QI8_0]*dB[l%2];
                }
            }
        }
    }
}
template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_q6_K_q8_1_mma(
        const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
    typedef tile<16, 4, int> tile_A;
    typedef tile< 8, 4, int> tile_B;
    typedef tile<16, 8, int> tile_C;
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);
    constexpr int rows_per_warp = ggml_cuda_mmq_get_rows_per_warp(type, J, fallback);
    constexpr int ntx = rows_per_warp/tile_C::I;
    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K);
    const int * x_qs = (const int *) x;
    const float * x_df = (const float *) x_qs + MMQ_TILE_NE_K*2;
    const int * x_sc = (const int *) x_df + MMQ_TILE_NE_K/QI6_K;
    const int * y_qs = (const int *) y + 4;
    const float * y_df = (const float *) y;
    const int i0 = (threadIdx.y / ntx) * (ntx*tile_A::I);
    tile_A A[ntx][8];
    int scA[ntx][tile_C::ne/2][8];
    float dA[ntx][tile_C::ne/2];
#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += 8) {
            const int k0 = k00 + k01;
            load_ldmatrix(A[n][k01/4 + 0], x_qs + (i0 + n*tile_A::I)*sram_stride + (k0 + 0), sram_stride);
            load_ldmatrix(A[n][k01/4 + 1], x_qs + (i0 + n*tile_A::I)*sram_stride + (k0 + tile_A::J), sram_stride);
        }
#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += 16) {
            const int k0 = k00 + k01;
#pragma unroll
            for (int l = 0; l < tile_C::ne/2; ++l) {
                const int i = i0 + n*tile_C::I + tile_C::get_i(2*l);
                const int sc_packed = x_sc[i*sram_stride + k0/16];
                const int8_t * sc = (const int8_t *) &sc_packed;
#pragma unroll
                for (int ksc = 0; ksc < sizeof(int); ++ksc) {
                    scA[n][l][k01/4 + ksc] = sc[ksc];
                }
            }
        }
#pragma unroll
        for (int l = 0; l < tile_C::ne/2; ++l) {
            const int i = i0 + n*tile_C::I + tile_C::get_i(2*l);
            dA[n][l] = x_df[i*sram_stride];
        }
    }
#pragma unroll
    for (int j0 = 0; j0 < J; j0 += ntx*tile_C::J) {
        float tmp[ntx][tile_C::ne] = {{0.0f}};
#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += 8) {
            tile_B B[2];
            float dB[tile_C::ne/2];
            load_generic(B[0], y_qs + j0*MMQ_TILE_Y_K + 0 + k01, MMQ_TILE_Y_K);
            load_generic(B[1], y_qs + j0*MMQ_TILE_Y_K + tile_B::J + k01, MMQ_TILE_Y_K);
#pragma unroll
            for (int l = 0; l < tile_C::ne/2; ++l) {
                const int j = j0 + tile_C::get_j(l);
                dB[l] = y_df[j*MMQ_TILE_Y_K + k01/QI8_1];
            }
#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C C[2];
                mma(C[0], A[n][k01/4 + 0], B[0]);
                mma(C[1], A[n][k01/4 + 1], B[1]);
#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    tmp[n][l] += (C[0].x[l]*scA[n][l/2][k01/4 + 0] + C[1].x[l]*scA[n][l/2][k01/4 + 1])*dB[l%2];
                }
            }
        }
#pragma unroll
        for (int n = 0; n < ntx; ++n) {
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                sum[(j0/tile_C::J + n)*tile_C::ne + l] += tmp[n][l]*dA[n][l/2];
            }
        }
    }
}
