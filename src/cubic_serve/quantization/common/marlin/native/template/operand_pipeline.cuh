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

template <typename Config, typename Ref_A, typename Ref_B, typename Ref_scales_ptr,
          typename Ref_global_scale_ptr, typename Ref_zp_ptr, typename Ref_g_idx,
          typename Ref_prob_k, typename Ref_scales_expert_stride, typename Ref_expert_id,
          typename Ref_sh_rd_block_sorted_ids, typename Ref_block_num_valid_tokens,
          typename Ref_a_gl_stride, typename Ref_a_gl_rd_delta_i, typename Ref_b_gl_stride,
          typename Ref_b_gl_rd_delta_o, typename Ref_s_gl_rd_delta, typename Ref_zp_gl_rd_delta,
          typename Ref_a_gl_rd_row, typename Ref_a_gl_rd_col, typename Ref_b_gl_rd,
          typename Ref_slice_k_finish, typename Ref_slice_k_start_shared_fetch,
          typename Ref_s_gl_rd, typename Ref_s_sh_wr, typename Ref_s_sh_wr_pred,
          typename Ref_zp_gl_rd, typename Ref_zp_sh_wr, typename Ref_zp_sh_wr_pred,
          typename Ref_a_sh_wr_trans, typename Ref_sh_b, typename Ref_sh_g_idx, typename Ref_sh_zp,
          typename Ref_sh_s, typename Ref_sh_a>
__device__ __forceinline__ void marlin_fetch_to_shared(
    Ref_A& A, Ref_B& B, Ref_scales_ptr& scales_ptr, Ref_global_scale_ptr& global_scale_ptr,
    Ref_zp_ptr& zp_ptr, Ref_g_idx& g_idx, Ref_prob_k& prob_k,
    Ref_scales_expert_stride& scales_expert_stride, Ref_expert_id& expert_id,
    Ref_sh_rd_block_sorted_ids& sh_rd_block_sorted_ids,
    Ref_block_num_valid_tokens& block_num_valid_tokens, Ref_a_gl_stride& a_gl_stride,
    Ref_a_gl_rd_delta_i& a_gl_rd_delta_i, Ref_b_gl_stride& b_gl_stride,
    Ref_b_gl_rd_delta_o& b_gl_rd_delta_o, Ref_s_gl_rd_delta& s_gl_rd_delta,
    Ref_zp_gl_rd_delta& zp_gl_rd_delta, Ref_a_gl_rd_row& a_gl_rd_row, Ref_a_gl_rd_col& a_gl_rd_col,
    Ref_b_gl_rd& b_gl_rd, Ref_slice_k_finish& slice_k_finish,
    Ref_slice_k_start_shared_fetch& slice_k_start_shared_fetch, Ref_s_gl_rd& s_gl_rd,
    Ref_s_sh_wr& s_sh_wr, Ref_s_sh_wr_pred& s_sh_wr_pred, Ref_zp_gl_rd& zp_gl_rd,
    Ref_zp_sh_wr& zp_sh_wr, Ref_zp_sh_wr_pred& zp_sh_wr_pred, Ref_a_sh_wr_trans& a_sh_wr_trans,
    Ref_sh_b& sh_b, Ref_sh_g_idx& sh_g_idx, Ref_sh_zp& sh_zp, Ref_sh_s& sh_s, Ref_sh_a& sh_a,
    int pipe, int a_off, bool pred, int n_offset, int problem_n) {
  static constexpr auto moe_block_size = Config::moe_block_size;
  static constexpr auto metadata_planes = Config::metadata_planes;
  static constexpr auto metadata_scale = Config::metadata_scale;
  static constexpr auto has_zp = Config::has_zp;
  static constexpr auto has_act_order = Config::has_act_order;
  static constexpr auto a_sh_stride = Config::a_sh_stride;
  static constexpr auto a_gl_rd_delta_o = Config::a_gl_rd_delta_o;
  static constexpr auto a_sh_wr_iters = Config::a_sh_wr_iters;
  static constexpr auto b_sh_stride = Config::b_sh_stride;
  static constexpr auto b_thread_vecs = Config::b_thread_vecs;
  static constexpr auto b_sh_stage = Config::b_sh_stage;
  static constexpr auto b_sh_wr_iters = Config::b_sh_wr_iters;
  static constexpr auto s_tb_groups = Config::s_tb_groups;
  static constexpr auto s_sh_stage = Config::s_sh_stage;
  static constexpr auto tb_k = Config::tb_k;
  static constexpr auto g_idx_stage = Config::g_idx_stage;
  static constexpr auto zp_tb_groups = Config::zp_tb_groups;
  static constexpr auto zp_sh_stage = Config::zp_sh_stage;
  static constexpr auto threads = Config::threads;
  static constexpr auto thread_k_blocks = Config::thread_k_blocks;
  static constexpr auto m_block_size_8 = Config::m_block_size_8;
  static constexpr auto group_blocks = Config::group_blocks;

  if (pred) {
    int4* sh_a_stage = sh_a + moe_block_size * a_sh_stride * pipe;
#pragma unroll
    for (int i = 0; i < a_sh_wr_iters; i++) {
      int row = a_gl_rd_delta_i / a_gl_stride * i + a_gl_rd_row;
      int64_t sorted_row = 0;
      if (!m_block_size_8 || row < 8) sorted_row = sh_rd_block_sorted_ids[row];
      int64_t true_idx = sorted_row * a_gl_stride + a_gl_rd_col + a_gl_rd_delta_o * a_off;
      cp_async4_pred(&sh_a_stage[a_sh_wr_trans[i]], &A[true_idx], row < block_num_valid_tokens);
    }

    if constexpr (Config::storage_bits == Config::b_type.size_bits()) {
    int4* sh_b_stage = sh_b + Config::b_storage_stage * pipe;
#pragma unroll
    for (int i = 0; i < (b_sh_wr_iters * b_thread_vecs); i++) {
      constexpr int count = div_ceil(b_sh_stride, threads);
      int b_gl_idx = b_gl_rd + (i % count) * threads +
                     b_gl_stride * (i / count) * div_ceil(threads, b_sh_stride);

      cp_async4(&sh_b_stage[threads * i + threadIdx.x], &B[b_gl_idx]);
    }

    } else {
      constexpr int bits = Config::storage_bits, fragment_bits = Config::b_type.size_bits();
      constexpr int row_vectors = b_sh_stride * bits / fragment_bits;
      constexpr int vectors = Config::b_storage_stage;
      const auto base = b_gl_rd - b_gl_stride * (threadIdx.x / b_sh_stride) -
                        threadIdx.x % b_sh_stride;
      int4* sh_b_stage = sh_b + vectors * pipe;
#pragma unroll
      for (int t = threadIdx.x; t < vectors; t += threads) {
        const auto src = (base + (t / row_vectors) * b_gl_stride) * bits / fragment_bits +
                         t % row_vectors;
        cp_async4(&sh_b_stage[t], &B[src]);
      }
    }

    b_gl_rd += b_gl_rd_delta_o;

    if constexpr (has_act_order) {
      // Fetch g_idx thread-block portion
      int full_pipe = a_off;
      int cur_k = slice_k_start_shared_fetch + tb_k * full_pipe;
      if (cur_k < prob_k && cur_k < slice_k_finish) {
        int4* sh_g_idx_stage = sh_g_idx + g_idx_stage * pipe;

        int4 const* cur_g_idx_stage_ptr = reinterpret_cast<int4 const*>(&g_idx[cur_k]);

        if (threadIdx.x < g_idx_stage) {
          cp_async4_pred(&sh_g_idx_stage[threadIdx.x], &cur_g_idx_stage_ptr[threadIdx.x]);
        }
      }
    } else {
      if constexpr (group_blocks != -1) {
        int4* sh_s_stage = sh_s + s_sh_stage * metadata_scale * pipe;

        // Only fetch scales if this tile starts a new group
        if (pipe % div_ceil(group_blocks, thread_k_blocks) == 0) {
          if constexpr (metadata_planes) {
            Config::GroupDecoder::template fetch<metadata_planes>(
                sh_s_stage, s_sh_stage, s_sh_wr, s_gl_rd, global_scale_ptr, expert_id,
                Config::s_sh_stride, s_gl_rd_delta, n_offset
#ifdef CUBIC_MARLIN_RUNTIME_METADATA
                , problem_n, prob_k
#endif
                );
          } else if (s_sh_wr_pred) {
            cp_async4(&sh_s_stage[s_sh_wr], &scales_ptr[s_gl_rd]);
          }
          s_gl_rd += s_gl_rd_delta * s_tb_groups;
        }
      }

      if constexpr (has_zp && group_blocks != -1) {
        int4* sh_zp_stage = sh_zp + zp_sh_stage * pipe;

        // Only fetch zero points if this tile starts a new group
        if (pipe % div_ceil(group_blocks, thread_k_blocks) == 0) {
          if (zp_sh_wr_pred) {
            cp_async4(&sh_zp_stage[zp_sh_wr], &zp_ptr[zp_gl_rd]);
          }
          zp_gl_rd += zp_gl_rd_delta * zp_tb_groups;
        }
      }
    }
  }
  // Insert a fence even when we are winding down the pipeline to ensure that
  // waiting is also correct at this point.
  cp_async_fence();
}

}  // namespace MARLIN_NAMESPACE_NAME
