#pragma once
/*
 * Modified by Neural Magic
 * Copyright (C) Marlin.2024 Elias Frantar
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Adapted from https://github.com/IST-DASLab/marlin
 */

#include "primitives.cuh"
namespace MARLIN_NAMESPACE_NAME {

template <typename Config, typename Ref_sh_red, typename Ref_frag_c>
__device__ __forceinline__ void marlin_thread_block_reduce(Ref_sh_red& sh_red, Ref_frag_c& frag_c) {
  using FragC = typename Config::FragC;
  static constexpr auto is_a_8bit = Config::is_a_8bit;
  static constexpr auto b_sh_stride_threads = Config::b_sh_stride_threads;
  static constexpr auto threads = Config::threads;
  static constexpr auto thread_m_blocks = Config::thread_m_blocks;
  static constexpr auto m_block_size_8 = Config::m_block_size_8;

  constexpr int red_off = threads / b_sh_stride_threads / 2;
  if (red_off >= 1) {
    auto red_idx = threadIdx.x / b_sh_stride_threads;
    constexpr int red_sh_stride = b_sh_stride_threads * (is_a_8bit ? 2 : 4) * 2;
    constexpr int red_sh_delta = b_sh_stride_threads;
    int red_sh_rd =
        red_sh_stride * (threadIdx.x / b_sh_stride_threads) + (threadIdx.x % b_sh_stride_threads);

    // Parallel logarithmic shared memory reduction. We make sure to avoid any
    // unnecessary read or write iterations, e.g., for two warps we write only
    // once by warp 1 and read only once by warp 0.

#pragma unroll
    for (int m_block = 0; m_block < thread_m_blocks; m_block++) {
#pragma unroll
      for (int i = red_off; i > 0; i /= 2) {
        if (i <= red_idx && red_idx < 2 * i) {
#pragma unroll
          for (int j = 0; j < (is_a_8bit ? 2 : 4) * 2; j += (m_block_size_8 ? 2 : 1)) {
            int red_sh_wr = red_sh_delta * j + (red_sh_rd - red_sh_stride * i);
            if (i < red_off) {
              float* c_rd = reinterpret_cast<float*>(&sh_red[red_sh_delta * j + red_sh_rd]);
              float* c_wr = reinterpret_cast<float*>(&sh_red[red_sh_wr]);
#pragma unroll
              for (int k = 0; k < 4; k++)
                reinterpret_cast<FragC*>(frag_c)[(is_a_8bit ? 2 : 4) * 2 * m_block + j][k] +=
                    c_rd[k] + c_wr[k];
            }
            sh_red[red_sh_wr] =
                reinterpret_cast<int4*>(&frag_c)[(is_a_8bit ? 2 : 4) * 2 * m_block + j];
          }
        }
        __syncthreads();
      }
      if (red_idx == 0) {
#pragma unroll
        for (int i = 0; i < (is_a_8bit ? 2 : 4) * 2; i += (m_block_size_8 ? 2 : 1)) {
          float* c_rd = reinterpret_cast<float*>(&sh_red[red_sh_delta * i + red_sh_rd]);
#pragma unroll
          for (int j = 0; j < 4; j++)
            reinterpret_cast<FragC*>(frag_c)[(is_a_8bit ? 2 : 4) * 2 * m_block + i][j] += c_rd[j];
        }
      }
      __syncthreads();
    }
  }
}

template <typename Config, typename Ref_C, typename Ref_prob_n, typename Ref_slice_col,
          typename Ref_sh_block_sorted_ids, typename Ref_block_num_valid_tokens,
          typename Ref_sh_red, typename Ref_frag_c>
__device__ __forceinline__ void marlin_global_reduce_fp16(
    Ref_C& C, Ref_prob_n& prob_n, Ref_slice_col& slice_col,
    Ref_sh_block_sorted_ids& sh_block_sorted_ids,
    Ref_block_num_valid_tokens& block_num_valid_tokens, Ref_sh_red& sh_red, Ref_frag_c& frag_c,
    bool first, bool last) {
  using Cdtype = typename Config::Cdtype;
  using c_scalar_t = typename Config::c_scalar_t;
  static constexpr auto is_a_8bit = Config::is_a_8bit;
  static constexpr auto tb_n_warps = Config::tb_n_warps;
  static constexpr auto thread_m_blocks = Config::thread_m_blocks;
  static constexpr auto thread_n_blocks = Config::thread_n_blocks;
  static constexpr auto m_block_size_8 = Config::m_block_size_8;

  // We are very careful here to reduce directly in the output buffer to
  // maximize L2 cache utilization in this step. To do this, we write out
  // results in FP16 (but still reduce with FP32 compute).
  constexpr int active_threads = 32 * tb_n_warps;
  bool is_th_active = threadIdx.x < active_threads;
  if (!is_th_active) {
    return;
  }

  int c_gl_stride = prob_n / 8 * (is_a_8bit ? 2 : 1);
  int c_gl_wr_delta_o = 8 * c_gl_stride;
  int c_gl_wr_delta_i = 4 * (active_threads / 32);
  int c_gl_wr;
  if constexpr (m_block_size_8) {
    c_gl_wr =
        c_gl_stride * ((threadIdx.x % 4) * 2) + 4 * (threadIdx.x / 32) + (threadIdx.x % 32) / 8;
    c_gl_wr += (2 * thread_n_blocks) * slice_col;
  } else {
    c_gl_wr = c_gl_stride * ((threadIdx.x % 32) / 4) + 4 * (threadIdx.x / 32) + threadIdx.x % 4;
    c_gl_wr += (2 * thread_n_blocks) * slice_col * (is_a_8bit ? 2 : 1);
  }
  constexpr int c_sh_wr_delta = active_threads;
  int c_sh_wr = threadIdx.x;

  if (!first) {
#pragma unroll
    for (int i = 0; i < (m_block_size_8 ? 2 : thread_m_blocks * 4); i++) {
      int c_idx;
      if constexpr (m_block_size_8)
        c_idx = c_gl_wr + i * c_gl_stride + (threadIdx.x % 8) / 4 * c_gl_wr_delta_i;
      else
        c_idx = c_gl_wr + c_gl_wr_delta_o * (i / 2) + c_gl_wr_delta_i * (i % 2);
      if (c_idx / c_gl_stride < block_num_valid_tokens) {
        int64_t sorted_row = sh_block_sorted_ids[c_idx / c_gl_stride];
        int64_t true_idx = sorted_row * c_gl_stride + c_idx % c_gl_stride;
        if constexpr (is_a_8bit) {
          int2* sh_red_int2 = reinterpret_cast<int2*>(sh_red);
          int2* c_int2 = reinterpret_cast<int2*>(C);
          sh_red_int2[c_sh_wr + c_sh_wr_delta * i] = c_int2[true_idx];
        } else {
          sh_red[c_sh_wr + c_sh_wr_delta * i] = C[true_idx];
        }
      }
    }
  }

#pragma unroll
  for (int i = 0; i < (m_block_size_8 ? 2 : thread_m_blocks * 4); i++) {
    if (!first) {
      c_scalar_t* c_red_f16;
      if constexpr (is_a_8bit) {
        int2 tmp = reinterpret_cast<int2*>(sh_red)[c_sh_wr + i * c_sh_wr_delta];
        c_red_f16 = reinterpret_cast<c_scalar_t*>(&tmp);
      } else {
        int4 tmp = sh_red[c_sh_wr + i * c_sh_wr_delta];
        c_red_f16 = reinterpret_cast<c_scalar_t*>(&tmp);
      }
#pragma unroll
      for (int j = 0; j < 2 * (is_a_8bit ? 2 : 4); j++) {
        int delta = 0;
        if constexpr (m_block_size_8) {
          delta = j % 2 == 1 ? -2 : 0;
        }
        reinterpret_cast<float*>(
            &frag_c)[(is_a_8bit ? 2 : 4) * 2 * 4 * (i / 4) + 4 * j + (i % 4) + delta] +=
            Cdtype::num2float(c_red_f16[j]);
      }
    }
    if (!last) {
      c_scalar_t c_f16[is_a_8bit ? 4 : 8];
#pragma unroll
      for (int j = 0; j < 2 * (is_a_8bit ? 2 : 4); j++) {
        int delta = 0;
        if constexpr (m_block_size_8) {
          delta = j % 2 == 1 ? -2 : 0;
        }
        c_f16[j] = Cdtype::float2num(reinterpret_cast<float*>(
            &frag_c)[(is_a_8bit ? 2 : 4) * 2 * 4 * (i / 4) + 4 * j + (i % 4) + delta]);
      }

      int c_idx;
      if constexpr (m_block_size_8)
        c_idx = c_gl_wr + i * c_gl_stride + (threadIdx.x % 8) / 4 * c_gl_wr_delta_i;
      else
        c_idx = c_gl_wr + c_gl_wr_delta_o * (i / 2) + c_gl_wr_delta_i * (i % 2);
      if (c_idx / c_gl_stride < block_num_valid_tokens) {
        int64_t sorted_row = sh_block_sorted_ids[c_idx / c_gl_stride];
        int64_t true_idx = sorted_row * c_gl_stride + c_idx % c_gl_stride;
        if constexpr (is_a_8bit) {
          int2* c_int2 = reinterpret_cast<int2*>(C);
          c_int2[true_idx] = *reinterpret_cast<int2*>(c_f16);
        } else {
          C[true_idx] = *reinterpret_cast<int4*>(c_f16);
        }
      }
    }
  }
}

template <typename Config, typename Ref_C_tmp, typename Ref_block_num_valid_tokens,
          typename Ref_locks_off, typename Ref_sh_red, typename Ref_frag_c>
__device__ __forceinline__ void marlin_global_reduce_fp32(
    Ref_C_tmp& C_tmp, Ref_block_num_valid_tokens& block_num_valid_tokens, Ref_locks_off& locks_off,
    Ref_sh_red& sh_red, Ref_frag_c& frag_c, bool first, bool last) {
  static constexpr auto is_a_8bit = Config::is_a_8bit;
  static constexpr auto tb_n_warps = Config::tb_n_warps;
  static constexpr auto thread_m_blocks = Config::thread_m_blocks;
  static constexpr auto thread_n_blocks = Config::thread_n_blocks;
  static constexpr auto m_block_size_8 = Config::m_block_size_8;

  constexpr bool compact_m8 = Config::fixed_partitions && m_block_size_8;
  constexpr int tb_m = compact_m8 ? 8 : thread_m_blocks * 16;
  constexpr int tb_n = thread_n_blocks * 16;

  constexpr int c_size = tb_m * tb_n * sizeof(float) / 16;

  constexpr int active_threads = 32 * tb_n_warps;
  bool is_th_active = threadIdx.x < active_threads;

  constexpr int num_floats = thread_m_blocks * (is_a_8bit ? 2 : 4) * 2 * 4;
  constexpr int th_size = num_floats * sizeof(float) / 16;

  int c_cur_offset = locks_off * c_size;

  if (!is_th_active) {
    return;
  }

  if (!first) {
    float* frag_c_ptr = reinterpret_cast<float*>(&frag_c);
#pragma unroll
    for (int k = 0; k < th_size; k++) {
      if constexpr (m_block_size_8) {
        if (k % 2) continue;
      } else {
        if (k / 8 * 16 + (threadIdx.x % 32) / 4 >= block_num_valid_tokens) continue;
      }

      sh_red[threadIdx.x] = C_tmp[c_cur_offset + active_threads * (compact_m8 ? k / 2 : k) + threadIdx.x];

      float* sh_c_ptr = reinterpret_cast<float*>(&sh_red[threadIdx.x]);
#pragma unroll
      for (int f = 0; f < 4; f++) {
        frag_c_ptr[k * 4 + f] += sh_c_ptr[f];
      }
    }
  }

  if (!last) {
    int4* frag_c_ptr = reinterpret_cast<int4*>(&frag_c);
#pragma unroll
    for (int k = 0; k < th_size; k++) {
      if constexpr (m_block_size_8) {
        if (k % 2) continue;
      } else {
        if (k / 8 * 16 + (threadIdx.x % 32) / 4 >= block_num_valid_tokens) continue;
      }

      C_tmp[c_cur_offset + active_threads * (compact_m8 ? k / 2 : k) + threadIdx.x] = frag_c_ptr[k];
    }
  }
}

}  // namespace MARLIN_NAMESPACE_NAME
