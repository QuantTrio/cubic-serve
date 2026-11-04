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

template <typename Config, typename Ref_g_idx, typename Ref_prob_k, typename Ref_slice_iters,
          typename Ref_a_gl_rd_col, typename Ref_slice_k_start,
          typename Ref_slice_k_start_shared_fetch, typename Ref_zero_accums,
          typename Ref_fetch_act_order_scales_to_shared, typename Ref_fetch_to_shared,
          typename Ref_fetch_col_zp_to_shared, typename Ref_fetch_col_scale_to_shared,
          typename Ref_wait_for_stage, typename Ref_fetch_to_registers,
          typename Ref_init_same_group, typename Ref_fetch_scales_to_registers,
          typename Ref_fetch_zp_to_registers>
__device__ __forceinline__ void marlin_start_pipes(
    Ref_g_idx& g_idx, Ref_prob_k& prob_k, Ref_slice_iters& slice_iters,
    Ref_a_gl_rd_col& a_gl_rd_col, Ref_slice_k_start& slice_k_start,
    Ref_slice_k_start_shared_fetch& slice_k_start_shared_fetch, Ref_zero_accums& zero_accums,
    Ref_fetch_act_order_scales_to_shared& fetch_act_order_scales_to_shared,
    Ref_fetch_to_shared& fetch_to_shared, Ref_fetch_col_zp_to_shared& fetch_col_zp_to_shared,
    Ref_fetch_col_scale_to_shared& fetch_col_scale_to_shared, Ref_wait_for_stage& wait_for_stage,
    Ref_fetch_to_registers& fetch_to_registers, Ref_init_same_group& init_same_group,
    Ref_fetch_scales_to_registers& fetch_scales_to_registers,
    Ref_fetch_zp_to_registers& fetch_zp_to_registers) {
  static constexpr auto has_zp = Config::has_zp;
  static constexpr auto dequant_skip_flop = Config::dequant_skip_flop;
  static constexpr auto has_act_order = Config::has_act_order;
  static constexpr auto a_gl_rd_delta_o = Config::a_gl_rd_delta_o;
  static constexpr auto tb_k = Config::tb_k;
  static constexpr auto stages = Config::stages;
  static constexpr auto group_blocks = Config::group_blocks;
  static constexpr auto is_zp_float = Config::is_zp_float;

#pragma unroll
  for (int i = 0; i < stages - 1; i++) {
    if (has_act_order && i == 0) {
      int last_g_idx = slice_k_start + stages * tb_k * 2;
      if (last_g_idx >= prob_k) {
        last_g_idx = prob_k - 1;
      }
      fetch_act_order_scales_to_shared(true, g_idx[slice_k_start], g_idx[last_g_idx]);
    }

    if constexpr (has_zp && !is_zp_float && group_blocks == -1) {
      if (i == 0) {
        fetch_col_zp_to_shared();
        if constexpr (!dequant_skip_flop) {
          fetch_col_scale_to_shared();
        }
      }
    }
    fetch_to_shared(i, i, i < slice_iters);
  }

  zero_accums();
  wait_for_stage();
  init_same_group(0);
  fetch_to_registers(0, 0);
  fetch_scales_to_registers(0, 0);
  fetch_zp_to_registers(0, 0);
  a_gl_rd_col += a_gl_rd_delta_o * (stages - 1);
  if constexpr (has_act_order) {
    slice_k_start_shared_fetch += tb_k * (stages - 1);
  }
}

template <
    typename Config, typename Ref_b_bias_ptr, typename Ref_scales_ptr, typename Ref_g_idx,
    typename Ref_prob_k, typename Ref_locks, typename Ref_has_bias, typename Ref_use_atomic_add,
    typename Ref_use_fp32_reduce, typename Ref_in_part2, typename Ref_slice_row,
    typename Ref_slice_col_par, typename Ref_slice_col, typename Ref_slice_iters,
    typename Ref_slice_count, typename Ref_slice_idx, typename Ref_B_expert_off,
    typename Ref_sh_a_s, typename Ref_locks_off, typename Ref_init_slice, typename Ref_b_gl_stride,
    typename Ref_b_gl_rd_delta_o, typename Ref_s_gl_stride, typename Ref_act_s_col_tb_stride,
    typename Ref_zp_gl_stride, typename Ref_a_gl_rd_col, typename Ref_b_gl_rd,
    typename Ref_slice_k_start, typename Ref_slice_k_finish,
    typename Ref_slice_k_start_shared_fetch, typename Ref_slice_n_offset, typename Ref_s_gl_rd,
    typename Ref_s_sh_wr, typename Ref_s_sh_wr_pred, typename Ref_zp_gl_rd, typename Ref_s_sh_rd,
    typename Ref_bias_sh_rd, typename Ref_bias_sh_wr, typename Ref_bias_gl_rd, typename Ref_sh_bias,
    typename Ref_sh_s, typename Ref_frag_c, typename Ref_frag_s, typename Ref_frag_bias,
    typename Ref_sh_first_group_id, typename Ref_sh_num_groups,
    typename Ref_fetch_act_order_scales_to_shared, typename Ref_fetch_to_shared,
    typename Ref_wait_for_stage, typename Ref_fetch_to_registers, typename Ref_init_same_group,
    typename Ref_fetch_scales_to_registers, typename Ref_fetch_zp_to_registers,
    typename Ref_is_first_matmul_in_slice, typename Ref_matmul, typename Ref_matmul_a8,
    typename Ref_thread_block_reduce, typename Ref_global_reduce_fp16,
    typename Ref_global_reduce_fp32, typename Ref_write_result, typename Ref_start_pipes>
__device__ __forceinline__ void marlin_execute(
    Ref_b_bias_ptr& b_bias_ptr, Ref_scales_ptr& scales_ptr, Ref_g_idx& g_idx, Ref_prob_k& prob_k,
    Ref_locks& locks, Ref_has_bias& has_bias, Ref_use_atomic_add& use_atomic_add,
    Ref_use_fp32_reduce& use_fp32_reduce, Ref_in_part2& in_part2, Ref_slice_row& slice_row,
    Ref_slice_col_par& slice_col_par, Ref_slice_col& slice_col, Ref_slice_iters& slice_iters,
    Ref_slice_count& slice_count, Ref_slice_idx& slice_idx, Ref_B_expert_off& B_expert_off,
    Ref_sh_a_s& sh_a_s, Ref_locks_off& locks_off, Ref_init_slice& init_slice,
    Ref_b_gl_stride& b_gl_stride, Ref_b_gl_rd_delta_o& b_gl_rd_delta_o,
    Ref_s_gl_stride& s_gl_stride, Ref_act_s_col_tb_stride& act_s_col_tb_stride,
    Ref_zp_gl_stride& zp_gl_stride, Ref_a_gl_rd_col& a_gl_rd_col, Ref_b_gl_rd& b_gl_rd,
    Ref_slice_k_start& slice_k_start, Ref_slice_k_finish& slice_k_finish,
    Ref_slice_k_start_shared_fetch& slice_k_start_shared_fetch, Ref_slice_n_offset& slice_n_offset,
    Ref_s_gl_rd& s_gl_rd, Ref_s_sh_wr& s_sh_wr, Ref_s_sh_wr_pred& s_sh_wr_pred,
    Ref_zp_gl_rd& zp_gl_rd, Ref_s_sh_rd& s_sh_rd, Ref_bias_sh_rd& bias_sh_rd,
    Ref_bias_sh_wr& bias_sh_wr, Ref_bias_gl_rd& bias_gl_rd, Ref_sh_bias& sh_bias, Ref_sh_s& sh_s,
    Ref_frag_c& frag_c, Ref_frag_s& frag_s, Ref_frag_bias& frag_bias,
    Ref_sh_first_group_id& sh_first_group_id, Ref_sh_num_groups& sh_num_groups,
    Ref_fetch_act_order_scales_to_shared& fetch_act_order_scales_to_shared,
    Ref_fetch_to_shared& fetch_to_shared, Ref_wait_for_stage& wait_for_stage,
    Ref_fetch_to_registers& fetch_to_registers, Ref_init_same_group& init_same_group,
    Ref_fetch_scales_to_registers& fetch_scales_to_registers,
    Ref_fetch_zp_to_registers& fetch_zp_to_registers,
    Ref_is_first_matmul_in_slice& is_first_matmul_in_slice, Ref_matmul& matmul,
    Ref_matmul_a8& matmul_a8, Ref_thread_block_reduce& thread_block_reduce,
    Ref_global_reduce_fp16& global_reduce_fp16, Ref_global_reduce_fp32& global_reduce_fp32,
    Ref_write_result& write_result, Ref_start_pipes& start_pipes) {
  using Cdtype = typename Config::Cdtype;
  using scalar_t = typename Config::scalar_t;
  using c_scalar_t = typename Config::c_scalar_t;
  using c_scalar_t2 = typename Config::c_scalar_t2;
  static constexpr auto use_fp16_accum = Config::use_fp16_accum;
  static constexpr auto a_type = Config::a_type;
  static constexpr auto b_type = Config::b_type;
  static constexpr auto is_a_8bit = Config::is_a_8bit;
  static constexpr auto has_zp = Config::has_zp;
  static constexpr auto dequant_skip_flop = Config::dequant_skip_flop;
  static constexpr auto has_act_order = Config::has_act_order;
  static constexpr auto a_gl_rd_delta_o = Config::a_gl_rd_delta_o;
  static constexpr auto b_sh_stride = Config::b_sh_stride;
  static constexpr auto b_sh_wr_iters = Config::b_sh_wr_iters;
  static constexpr auto s_sh_stride = Config::s_sh_stride;
  static constexpr auto tb_k = Config::tb_k;
  static constexpr auto tb_n_warps = Config::tb_n_warps;
  static constexpr auto zp_sh_stride = Config::zp_sh_stride;
  static constexpr auto c_type_id = Config::c_type_id;
  static constexpr auto thread_m_blocks = Config::thread_m_blocks;
  static constexpr auto thread_n_blocks = Config::thread_n_blocks;
  static constexpr auto thread_k_blocks = Config::thread_k_blocks;
  static constexpr auto m_block_size_8 = Config::m_block_size_8;
  static constexpr auto stages = Config::stages;
  static constexpr auto group_blocks = Config::group_blocks;

  if (slice_iters) {
    start_pipes();
  }

  // Main loop.
  while (slice_iters) {
    // We unroll over both the global fetch and the register load pipeline to
    // ensure all shared memory accesses are static. Note that both pipelines
    // have even length meaning that the next iteration will always start at
    // index 0.

#pragma unroll
    for (int pipe = 0; pipe < stages;) {
      if constexpr (b_sh_wr_iters == 1) {
        // A single fragment cannot advance through the two-buffer lookahead
        // trigger (k == -1). Consume before recycling its shared stage.
        fetch_to_registers(0, pipe % stages);
        fetch_scales_to_registers(0, pipe);
        fetch_zp_to_registers(0, pipe);
        if constexpr (is_a_8bit) {
          matmul_a8(0);
        } else {
          matmul(0, pipe);
        }
        __syncthreads();
        fetch_to_shared((pipe + stages - 1) % stages, pipe, slice_iters >= stages);
        ++pipe;
        wait_for_stage();
        init_same_group(pipe % stages);
      } else {
#pragma unroll
      for (int k = 0; k < b_sh_wr_iters; k++) {
        if constexpr (Config::compact_registers) {
          fetch_to_registers(k, (pipe - (k == b_sh_wr_iters - 1 ? 1 : 0)) % stages);
        } else {
          fetch_to_registers(k + 1, pipe % stages);
        }
        if constexpr (Config::compact_registers && Config::metadata_planes &&
                      group_blocks >= thread_k_blocks) {
          if (k == 0) fetch_scales_to_registers(0, pipe);
        } else {
          fetch_scales_to_registers(k + 1, pipe);
        }
        fetch_zp_to_registers(k + 1, pipe);
        if (k == b_sh_wr_iters - 2) {
          // Without register lookahead, late warps still read the old stage.
          // Complete those reads before any warp recycles shared input storage.
          if constexpr (Config::compact_registers) __syncthreads();
          fetch_to_shared((pipe + stages - 1) % stages, pipe, slice_iters >= stages);
          pipe++;
          wait_for_stage();
          init_same_group(pipe % stages);
        }

        if constexpr (!is_a_8bit) {
          matmul(k, pipe - (k >= b_sh_wr_iters - 2 ? 1 : 0));
        } else {
          static_assert(group_blocks != 0 && group_blocks != 1);
          matmul_a8(k);
        }
      }
      }
      slice_iters--;
      if (slice_iters == 0) {
        break;
      }
    }

    a_gl_rd_col += a_gl_rd_delta_o * stages;

    if constexpr (has_act_order) {
      slice_k_start += tb_k * stages;

      if (slice_k_start < prob_k) {
        slice_k_start_shared_fetch += tb_k * stages;
        int first_group_id = g_idx[slice_k_start];
        int last_g_idx = slice_k_start + stages * tb_k * 2;
        if (last_g_idx >= prob_k) {
          last_g_idx = prob_k - 1;
        }
        int last_group_id = g_idx[last_g_idx];
        if (last_group_id >= sh_first_group_id + sh_num_groups) {
          fetch_act_order_scales_to_shared(false, first_group_id, last_group_id);
          __syncthreads();
        }
      }
    }

    // Process results and, if necessary, proceed to the next column slice.
    // While this pattern may not be the most readable, other ways of writing
    // the loop seemed to noticeably worse performance after compilation.
    if (slice_iters == 0) {
      // convert fp16 accum to fp32 for reduction
      if constexpr (use_fp16_accum) {
#pragma unroll
        for (int i = 0; i < (thread_m_blocks * (is_a_8bit ? 2 : 4) * 2); i++) {
          float* frag_c_part_float = reinterpret_cast<float*>(frag_c) + i * 4;
          scalar_t* frag_c_part_half = reinterpret_cast<scalar_t*>(frag_c_part_float);

#pragma unroll
          for (int i = 3; i >= 0; i--) {
            frag_c_part_float[i] = Cdtype::num2float(frag_c_part_half[i]);
          }
        }
      }

      if constexpr (is_a_8bit) {
        float frag_a_s[2 * thread_m_blocks];

        for (int i = 0; i < 2 * thread_m_blocks; i++)
          frag_a_s[i] = sh_a_s[i * 8 + (threadIdx.x % 32) / 4];

#pragma unroll
        for (int j = 0; j < 2; j++) {
#pragma unroll
          for (int i = 0; i < thread_m_blocks; i++) {
#pragma unroll
            for (int g = 0; g < 4; g++) {
              float c_val = frag_c[i][j][0][g];

              if constexpr (a_type == vllm::kS8) {
#ifndef CUBIC_MARLIN_INT_A8_FLOAT_SCALE
                c_val = __int2float_rn(*reinterpret_cast<int32_t*>(&c_val));
#endif
              }
              float s_val = frag_a_s[i * 2 + g / 2];
              frag_c[i][j][0][g] = c_val * s_val;
            }
#pragma unroll
            for (int g = 0; g < 4; g++) {
              float c_val = frag_c[i][j][1][g];

              if constexpr (a_type == vllm::kS8) {
#ifndef CUBIC_MARLIN_INT_A8_FLOAT_SCALE
                c_val = __int2float_rn(*reinterpret_cast<int32_t*>(&c_val));
#endif
              }
              float s_val = frag_a_s[i * 2 + g / 2];
              frag_c[i][j][1][g] = c_val * s_val;
            }
          }
        }
      }

      cp_async_wait<0>();
      bool last = slice_idx == slice_count - 1;
      // For per-column scales, we only fetch them here in the final step before
      // write-out
      if constexpr (!has_act_order && group_blocks == -1 &&
                    (has_zp && dequant_skip_flop || !has_zp)) {
        if (b_type.size_bits() == 8 || (last || use_atomic_add) || is_a_8bit) {
          if (s_sh_wr_pred) {
            cp_async4(&sh_s[s_sh_wr], &scales_ptr[s_gl_rd]);
          }
          cp_async_fence();
        }
      }

      // Reduction aliases operand shared storage. With current-subtile loads,
      // every warp must finish its final B read before the first reduction store.
      // An odd number of K tiles can end in the overlapping shared stage.
      if constexpr (Config::compact_registers) __syncthreads();
      thread_block_reduce();

      if (has_bias && last) {
        __syncthreads();
        cp_async4_pred(&sh_bias[bias_sh_wr], &b_bias_ptr[bias_gl_rd],
                       threadIdx.x < 16 * thread_n_blocks / 8);
        cp_async_fence();
      }

      if constexpr (!has_act_order && group_blocks == -1 &&
                    (has_zp && dequant_skip_flop || !has_zp || is_a_8bit)) {
        if constexpr (is_a_8bit) {
          cp_async_wait<0>();
          __syncthreads();
          if (threadIdx.x / 32 < tb_n_warps) {
            reinterpret_cast<int4*>(&frag_s)[0] = sh_s[s_sh_rd + 0];
          }
        } else if (b_type.size_bits() == 8 || (last || use_atomic_add)) {
          cp_async_wait<0>();
          __syncthreads();
          if (threadIdx.x / 32 < tb_n_warps) {
            reinterpret_cast<int4*>(&frag_s)[0] = sh_s[s_sh_rd + 0];
            reinterpret_cast<int4*>(&frag_s)[1] = sh_s[s_sh_rd + 4];
            if constexpr (m_block_size_8) {
              int idx = (threadIdx.x / 4) % 2;
              c_scalar_t2* frag_s_half2 = reinterpret_cast<c_scalar_t2*>(frag_s);
#pragma unroll
              for (int i = 0; i < 8; i++) {
                frag_s_half2[i] =
                    Cdtype::num2num2(reinterpret_cast<c_scalar_t*>(&frag_s_half2[i])[idx]);
              }
            }
          }
        }
      }

      // For 8-bit channelwise, we apply the scale before the global reduction
      // that converts the fp32 results to fp16 (so that we avoid possible
      // overflow in fp16)
      if constexpr (!has_act_order && group_blocks == -1 && is_a_8bit) {
#pragma unroll
        for (int j = 0; j < 2; j++) {
          float2 aa[2];
          aa[0] = Cdtype::num22float2(frag_s[0][j * 2][0]);
          aa[1] = Cdtype::num22float2(frag_s[0][j * 2 + 1][0]);

#pragma unroll
          for (int i = 0; i < thread_m_blocks; i++) {
#pragma unroll
            for (int g = 0; g < 4; g++) {
              float scale = reinterpret_cast<float*>(&aa[0])[g % 2];
              frag_c[i][j][0][g] *= scale;
            }

#pragma unroll
            for (int g = 0; g < 4; g++) {
              float scale = reinterpret_cast<float*>(&aa[1])[g % 2];
              frag_c[i][j][1][g] *= scale;
            }
          }
        }
      } else if (!has_act_order && group_blocks == -1 && b_type.size_bits() == 8 &&
                 (has_zp && dequant_skip_flop || !has_zp)) {
        if (threadIdx.x / 32 < tb_n_warps) {
#pragma unroll
          for (int i = 0; i < thread_m_blocks; i++) {
#pragma unroll
            for (int j = 0; j < 4; j++) {
              scale_float<c_type_id>(reinterpret_cast<float*>(&frag_c[i][j][0][0]),
                                     frag_s[j / 2][2 * (j % 2) + 0]);
              scale_float<c_type_id>(reinterpret_cast<float*>(&frag_c[i][j][0][2]),
                                     frag_s[j / 2][2 * (j % 2) + (m_block_size_8 ? 1 : 0)]);

              if constexpr (!m_block_size_8) {
                scale_float<c_type_id>(reinterpret_cast<float*>(&frag_c[i][j][1][0]),
                                       frag_s[j / 2][2 * (j % 2) + 1]);
                scale_float<c_type_id>(reinterpret_cast<float*>(&frag_c[i][j][1][2]),
                                       frag_s[j / 2][2 * (j % 2) + 1]);
              }
            }
          }
        }
      }

      if constexpr (Config::fixed_partitions) {
        // Each CTA owns one FP32 fragment range. The following kernel merges
        // contiguous K partitions in ascending order and rounds only once.
        locks_off = blockIdx.x;
        global_reduce_fp32(true, false);
        return;
      }
      if (slice_count > 1 && !use_atomic_add) {
        // only globally reduce if there is more than one block in a slice
        barrier_acquire(&locks[locks_off], slice_idx);
        if (use_fp32_reduce) {
          global_reduce_fp32(slice_idx == 0, last);
        } else {
          global_reduce_fp16(slice_idx == 0, last);
        }
        barrier_release(&locks[locks_off], last);
      }

      if (has_bias && last) {
        cp_async_wait<0>();
        __syncthreads();
        reinterpret_cast<int4*>(&frag_bias)[0] = sh_bias[bias_sh_rd];
        if constexpr (!is_a_8bit) reinterpret_cast<int4*>(&frag_bias)[1] = sh_bias[bias_sh_rd + 4];
        __syncthreads();
      }

      if (use_atomic_add && slice_count > 1 && slice_idx != 0)
        wait_negative_and_add(&locks[locks_off]);
      if (last || use_atomic_add)
        // only the last block in a slice actually writes the result
        write_result(last);
      slice_row = 0;
      if (!in_part2) {
        slice_col_par += gridDim.x;
      } else {
        slice_col_par++;
        slice_col++;
      }
      is_first_matmul_in_slice = true;
      init_slice();

      if (slice_iters) {
        a_gl_rd_col = a_gl_rd_delta_o * slice_row + threadIdx.x % a_gl_rd_delta_o;
        b_gl_rd =
            B_expert_off + b_gl_stride * (threadIdx.x / b_sh_stride) + (threadIdx.x % b_sh_stride);
        b_gl_rd += b_sh_stride * slice_col + b_gl_rd_delta_o * slice_row;

        bias_gl_rd = (thread_n_blocks * 16 / 8) * slice_col + threadIdx.x;
        // Update slice k/n for scales loading
        if constexpr (has_act_order) {
          slice_k_start = tb_k * slice_row;
          slice_k_finish = slice_k_start + tb_k * slice_iters;
          slice_k_start_shared_fetch = slice_k_start;
          slice_n_offset = act_s_col_tb_stride * slice_col;
        } else {
          if constexpr (group_blocks == -1) {
            s_gl_rd = s_sh_stride * slice_col + threadIdx.x;
            zp_gl_rd = zp_sh_stride * slice_col + threadIdx.x;
          } else if constexpr (group_blocks >= thread_k_blocks) {
            s_gl_rd = s_gl_stride * ((thread_k_blocks * slice_row) / group_blocks) +
                      s_sh_stride * slice_col + threadIdx.x;
            zp_gl_rd = zp_gl_stride * ((thread_k_blocks * slice_row) / group_blocks) +
                       zp_sh_stride * slice_col + threadIdx.x;
          } else {
            s_gl_rd = s_gl_stride * ((thread_k_blocks * slice_row) / group_blocks +
                                     threadIdx.x / s_sh_stride) +
                      s_sh_stride * slice_col + threadIdx.x % s_sh_stride;
            zp_gl_rd = zp_gl_stride * ((thread_k_blocks * slice_row) / group_blocks +
                                       threadIdx.x / zp_sh_stride) +
                       zp_sh_stride * slice_col + threadIdx.x % zp_sh_stride;
          }
        }
        start_pipes();
      }
    }
  }
}

}  // namespace MARLIN_NAMESPACE_NAME
