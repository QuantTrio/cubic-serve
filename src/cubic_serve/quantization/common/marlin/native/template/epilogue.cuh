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

template <typename Config, typename Ref_C, typename Ref_mul_topk_weights, typename Ref_prob_n,
          typename Ref_has_bias, typename Ref_use_atomic_add, typename Ref_global_scale_f32,
          typename Ref_slice_col, typename Ref_slice_count, typename Ref_sh_block_sorted_ids,
          typename Ref_sh_block_topk_weights, typename Ref_block_num_valid_tokens,
          typename Ref_sh_red, typename Ref_frag_c, typename Ref_frag_s, typename Ref_frag_bias>
__device__ __forceinline__ void marlin_write_result(
    Ref_C& C, Ref_mul_topk_weights& mul_topk_weights, Ref_prob_n& prob_n, Ref_has_bias& has_bias,
    Ref_use_atomic_add& use_atomic_add, Ref_global_scale_f32& global_scale_f32,
    Ref_slice_col& slice_col, Ref_slice_count& slice_count,
    Ref_sh_block_sorted_ids& sh_block_sorted_ids, Ref_sh_block_topk_weights& sh_block_topk_weights,
    Ref_block_num_valid_tokens& block_num_valid_tokens, Ref_sh_red& sh_red, Ref_frag_c& frag_c,
    Ref_frag_s& frag_s, Ref_frag_bias& frag_bias, bool last) {
  using Cdtype = typename Config::Cdtype;
  using scalar_t = typename Config::scalar_t;
  using c_scalar_t = typename Config::c_scalar_t;
  using c_scalar_t2 = typename Config::c_scalar_t2;
  using FragS = typename Config::FragS;
  static constexpr auto b_type = Config::b_type;
  static constexpr auto s_type = Config::s_type;
  static constexpr auto is_a_8bit = Config::is_a_8bit;
  static constexpr auto has_zp = Config::has_zp;
  static constexpr auto dequant_skip_flop = Config::dequant_skip_flop;
  static constexpr auto has_act_order = Config::has_act_order;
  static constexpr auto tb_n_warps = Config::tb_n_warps;
  static constexpr auto threads = Config::threads;
  static constexpr auto thread_m_blocks = Config::thread_m_blocks;
  static constexpr auto thread_n_blocks = Config::thread_n_blocks;
  static constexpr auto m_block_size_8 = Config::m_block_size_8;
  static constexpr auto group_blocks = Config::group_blocks;

  int c_gl_stride = prob_n / 8;
  constexpr int c_sh_stride = 2 * thread_n_blocks + 1;
  int c_gl_wr_delta = c_gl_stride * (threads / (2 * thread_n_blocks));
  constexpr int c_sh_rd_delta = c_sh_stride * (threads / (2 * thread_n_blocks));

  int c_gl_wr =
      c_gl_stride * (threadIdx.x / (2 * thread_n_blocks)) + (threadIdx.x % (2 * thread_n_blocks));
  c_gl_wr += (2 * thread_n_blocks) * slice_col;
  int c_sh_wr;
  if constexpr (m_block_size_8) {
    c_sh_wr = (8 * c_sh_stride) * ((threadIdx.x % 32) % 4 * 2) + (threadIdx.x % 32) / 4;
    c_sh_wr += 64 * (threadIdx.x / 32);
  } else {
    c_sh_wr = (4 * c_sh_stride) * ((threadIdx.x % 32) / 4) + (threadIdx.x % 32) % 4;
    c_sh_wr += (is_a_8bit ? 16 : 32) * (threadIdx.x / 32);
  }

  int c_sh_rd =
      c_sh_stride * (threadIdx.x / (2 * thread_n_blocks)) + (threadIdx.x % (2 * thread_n_blocks));

  // We first reorder in shared memory to guarantee the most efficient final
  // global write patterns
  auto write = [&](int idx, float c0, float c1, FragS& s, FragS& b_bias) {
    if constexpr (b_type == vllm::kFE2M1f && s_type == vllm::kFE4M3fn) {
      if (!mul_topk_weights) {
        c0 *= global_scale_f32;
        c1 *= global_scale_f32;
      }
    }

#if CUBIC_MARLIN_FP32_OUTPUT
    static_assert(!m_block_size_8);
    // Export the accumulator before any activation-dtype rounding. The
    // format binding owns subsequent scales and its single output cast.
    int row = idx / (4 * c_sh_stride);
    int col = 2 * (idx % (4 * c_sh_stride)) + slice_col * (16 * thread_n_blocks);
    if (row < block_num_valid_tokens && col + 1 < prob_n) {
      int logical = sh_block_sorted_ids[row];
      float* output = reinterpret_cast<float*>(C);
      output[logical * prob_n + col] = c0;
      output[logical * prob_n + col + 1] = c1;
    }
#else
    c_scalar_t2 res = Cdtype::nums2num2(Cdtype::float2num(c0), Cdtype::float2num(c1));

    // For per-column quantization we finally apply the scale here (only for
    // 4-bit)
    if constexpr (!has_act_order && group_blocks == -1 && !is_a_8bit && b_type.size_bits() == 4 &&
                  (has_zp && dequant_skip_flop || !has_zp)) {
      c_scalar_t2 tmp_scale = s[0];
      if constexpr (m_block_size_8) {
        tmp_scale = Cdtype::num2num2(reinterpret_cast<scalar_t*>(&s[0])[(threadIdx.x % 8) / 4]);
      }
      res = __hmul2(res, tmp_scale);
    }

    if (has_bias && last) {
      c_scalar_t2 tmp_bias = b_bias[0];
      if constexpr (m_block_size_8) {
        tmp_bias = Cdtype::num2num2(reinterpret_cast<scalar_t*>(&b_bias[0])[(threadIdx.x % 8) / 4]);
      }
      res = __hadd2(res, tmp_bias);
    }

    if constexpr (m_block_size_8) {
      ((c_scalar_t*)sh_red)[idx] = res.x;
      ((c_scalar_t*)sh_red)[idx + 8 * c_sh_stride] = res.y;
    } else {
      ((c_scalar_t2*)sh_red)[idx] = res;
    }
#endif
  };

  if (threadIdx.x / 32 < tb_n_warps) {
#pragma unroll
    for (int i = 0; i < thread_m_blocks; i++) {
#pragma unroll
      for (int j = 0; j < (is_a_8bit ? 2 : 4); j++) {
        if constexpr (m_block_size_8) {
          int wr = c_sh_wr + 16 * j;
          write(wr, frag_c[i][j][0][0], frag_c[i][j][0][1], frag_s[j / 2][2 * (j % 2) + 0],
                frag_bias[j / 2][2 * (j % 2) + 0]);
          write(wr + 8, frag_c[i][j][0][2], frag_c[i][j][0][3], frag_s[j / 2][2 * (j % 2) + 1],
                frag_bias[j / 2][2 * (j % 2) + 1]);
        } else {
          int wr = c_sh_wr + 8 * j;
          write(wr + (4 * c_sh_stride) * 0 + 0, frag_c[i][j][0][0], frag_c[i][j][0][1],
                frag_s[j / 2][2 * (j % 2) + 0], frag_bias[j / 2][2 * (j % 2) + 0]);
          write(wr + (4 * c_sh_stride) * 8 + 0, frag_c[i][j][0][2], frag_c[i][j][0][3],
                frag_s[j / 2][2 * (j % 2) + 0], frag_bias[j / 2][2 * (j % 2) + 0]);
          write(wr + (4 * c_sh_stride) * 0 + 4, frag_c[i][j][1][0], frag_c[i][j][1][1],
                frag_s[j / 2][2 * (j % 2) + 1], frag_bias[j / 2][2 * (j % 2) + 1]);
          write(wr + (4 * c_sh_stride) * 8 + 4, frag_c[i][j][1][2], frag_c[i][j][1][3],
                frag_s[j / 2][2 * (j % 2) + 1], frag_bias[j / 2][2 * (j % 2) + 1]);
        }
      }
      c_sh_wr += 16 * (4 * c_sh_stride);
    }
  }
#if CUBIC_MARLIN_FP32_OUTPUT
  // Direct accumulator stores still read the current tile's shared row map.
  // Finish those reads before any warp starts loading the next tile's map.
  __syncthreads();
  return;
#endif
  __syncthreads();

#pragma unroll
  for (int i = 0; i < div_ceil(16 * thread_m_blocks, threads / (2 * thread_n_blocks)); i++) {
    int row = c_gl_wr / c_gl_stride;
    if (row < block_num_valid_tokens) {
      int64_t sorted_row = sh_block_sorted_ids[row];
      int64_t true_idx = sorted_row * c_gl_stride + c_gl_wr % c_gl_stride;
      c_scalar_t2 topk_weight_score;
      if (mul_topk_weights) topk_weight_score = sh_block_topk_weights[row];
      if (use_atomic_add && slice_count > 1 || mul_topk_weights) {
        c_scalar_t2* C_half2 = reinterpret_cast<c_scalar_t2*>(&C[true_idx]);
        c_scalar_t2* sh_red_half2 = reinterpret_cast<c_scalar_t2*>(&sh_red[c_sh_rd]);
        if (mul_topk_weights) {
#pragma unroll
          for (int a = 0; a < 4; a++) {
            sh_red_half2[a] = __hmul2(sh_red_half2[a], topk_weight_score);
          }
        }

        if (use_atomic_add && slice_count > 1) {
#pragma unroll
          for (int a = 0; a < 4; a++) {
            atomicAdd(&C_half2[a], sh_red_half2[a]);
          }
        } else {
          C[true_idx] = *reinterpret_cast<int4*>(sh_red_half2);
        }
      } else {
        C[true_idx] = sh_red[c_sh_rd];
      }
      c_gl_wr += c_gl_wr_delta;
      c_sh_rd += c_sh_rd_delta;
    }
  }
  __syncthreads();
}

}  // namespace MARLIN_NAMESPACE_NAME
