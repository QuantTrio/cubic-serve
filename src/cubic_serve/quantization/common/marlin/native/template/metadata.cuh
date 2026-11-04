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

template <typename Config, typename Ref_scales_ptr, typename Ref_num_groups,
          typename Ref_s_gl_stride, typename Ref_slice_n_offset, typename Ref_sh_s,
          typename Ref_sh_first_group_id, typename Ref_sh_num_groups>
__device__ __forceinline__ void marlin_fetch_act_order_scales_to_shared(
    Ref_scales_ptr& scales_ptr, Ref_num_groups& num_groups, Ref_s_gl_stride& s_gl_stride,
    Ref_slice_n_offset& slice_n_offset, Ref_sh_s& sh_s, Ref_sh_first_group_id& sh_first_group_id,
    Ref_sh_num_groups& sh_num_groups, bool is_async, int first_group_id, int last_group_id) {
  static constexpr auto s_sh_stride = Config::s_sh_stride;
  static constexpr auto act_s_max_num_groups = Config::act_s_max_num_groups;

  sh_first_group_id = first_group_id;
  sh_num_groups = last_group_id - first_group_id + 1;

  if (sh_num_groups > act_s_max_num_groups) {
    sh_num_groups = act_s_max_num_groups;
  }

  if (sh_first_group_id + sh_num_groups > num_groups) {
    sh_num_groups = num_groups - sh_first_group_id;
  }

  int row_offset = first_group_id * s_gl_stride;

  if (is_async) {
    for (int i = 0; i < sh_num_groups; i++) {
      if (threadIdx.x < s_sh_stride) {
        cp_async4_pred(&sh_s[(i * s_sh_stride) + threadIdx.x],
                       &scales_ptr[row_offset + (i * s_gl_stride) + slice_n_offset + threadIdx.x]);
      }
    }
  } else {
    for (int i = 0; i < sh_num_groups; i++) {
      if (threadIdx.x < s_sh_stride) {
        sh_s[(i * s_sh_stride) + threadIdx.x] =
            scales_ptr[row_offset + (i * s_gl_stride) + slice_n_offset + threadIdx.x];
      }
    }
  }
}

template <typename Config, typename Ref_prob_k, typename Ref_act_s_col_stride,
          typename Ref_act_s_col_warp_stride, typename Ref_slice_k_start,
          typename Ref_slice_k_finish, typename Ref_s_sh_rd, typename Ref_sh_g_idx,
          typename Ref_sh_s, typename Ref_group_metadata, typename Ref_frag_s,
          typename Ref_act_frag_s, typename Ref_sh_first_group_id, typename Ref_is_same_group,
          typename Ref_same_group_id>
__device__ __forceinline__ void marlin_fetch_scales_to_registers(
    Ref_prob_k& prob_k, Ref_act_s_col_stride& act_s_col_stride,
    Ref_act_s_col_warp_stride& act_s_col_warp_stride, Ref_slice_k_start& slice_k_start,
    Ref_slice_k_finish& slice_k_finish, Ref_s_sh_rd& s_sh_rd, Ref_sh_g_idx& sh_g_idx,
    Ref_sh_s& sh_s, Ref_group_metadata& group_metadata, Ref_frag_s& frag_s,
    Ref_act_frag_s& act_frag_s, Ref_sh_first_group_id& sh_first_group_id,
    Ref_is_same_group& is_same_group, Ref_same_group_id& same_group_id, int k, int full_pipe) {
  static constexpr auto metadata_planes = Config::metadata_planes;
  static constexpr auto metadata_scale = Config::metadata_scale;
  static constexpr auto is_a_8bit = Config::is_a_8bit;
  static constexpr auto is_8bit_scale = Config::is_8bit_scale;
  static constexpr auto dequant_skip_flop = Config::dequant_skip_flop;
  static constexpr auto has_act_order = Config::has_act_order;
  static constexpr auto b_sh_wr_iters = Config::b_sh_wr_iters;
  static constexpr auto s_sh_stride = Config::s_sh_stride;
  static constexpr auto s_sh_stage = Config::s_sh_stage;
  static constexpr auto tb_k = Config::tb_k;
  static constexpr auto g_idx_stage = Config::g_idx_stage;
  static constexpr auto tb_n_warps = Config::tb_n_warps;
  static constexpr auto thread_k_blocks = Config::thread_k_blocks;
  static constexpr auto stages = Config::stages;
  static constexpr auto group_blocks = Config::group_blocks;

  if constexpr (metadata_planes) {
    static_assert(!is_a_8bit && !has_act_order && group_blocks > 0);
    constexpr int g = group_blocks >= thread_k_blocks ? group_blocks / thread_k_blocks : 1;
    if constexpr (g > 1) {
      // Async staging may already contain the next group. Keep both current
      // register buffers until the mathematical group boundary is crossed.
      if ((full_pipe % stages) % g != 0) return;
      if (!Config::compact_registers && k % b_sh_wr_iters != 0) {
#pragma unroll
        for (int p = 0; p < metadata_planes; ++p) {
#pragma unroll
          for (int v = 0; v < 8; ++v) group_metadata[1][p][v] = group_metadata[0][p][v];
        }
        return;
      }
    }
    int stage = g * ((full_pipe % stages) / g);
    int warp_row = (threadIdx.x / 32) / tb_n_warps;
    int group = group_blocks >= thread_k_blocks
                    ? 0
                    : (b_sh_wr_iters * warp_row + k % b_sh_wr_iters) / group_blocks;
    const float* metadata =
        reinterpret_cast<const float*>(sh_s + metadata_scale * s_sh_stage * stage);
#pragma unroll
    for (int p = 0; p < metadata_planes; ++p) {
#pragma unroll
      for (int v = 0; v < 8; ++v)
        group_metadata[(Config::compact_registers && group_blocks >= thread_k_blocks) ? 0 : k % 2][p][v] =
            metadata[p * 8 * s_sh_stage + (group * s_sh_stride + s_sh_rd) * 8 + v];
    }
    return;
  }
  int pipe = full_pipe % stages;
  using IT1 = typename std::conditional_t<is_a_8bit, int2, int4>;
  using IT0 = typename std::conditional_t<is_a_8bit, int, int2>;
  constexpr int group_blocks2 = div_ceil(group_blocks, is_a_8bit ? 2 : 1);

  if constexpr (!has_act_order) {
    // No act-order case
    if constexpr (group_blocks == -1) {
      // load only when starting a new slice
      if (k == 0 && full_pipe == 0 && dequant_skip_flop) {
        reinterpret_cast<int4*>(&frag_s)[0] = sh_s[s_sh_rd];
        reinterpret_cast<int4*>(&frag_s)[1] = sh_s[s_sh_rd + 4];
      }
    } else if constexpr (group_blocks != -1) {
      if constexpr (group_blocks >= thread_k_blocks) {
        constexpr int g = group_blocks / thread_k_blocks;
        if (pipe % g == 0) {
          if (k % b_sh_wr_iters == 0) {
            int4* sh_s_stage = sh_s + s_sh_stage * (g * (pipe / g));
            reinterpret_cast<int4*>(&frag_s[k % 2])[0] = sh_s_stage[s_sh_rd];
          } else {
            reinterpret_cast<int4*>(&frag_s[1])[0] = reinterpret_cast<int4*>(&frag_s[0])[0];
          }
        }
      } else if (group_blocks2 < b_sh_wr_iters || k % b_sh_wr_iters == 0) {
        auto warp_id = threadIdx.x / 32;
        int warp_row = warp_id / tb_n_warps;

        int k_blocks = b_sh_wr_iters * warp_row + k % b_sh_wr_iters;
        int cur_group_id = k_blocks / group_blocks2;

        int4* sh_s_stage = sh_s + s_sh_stage * pipe;

        if constexpr (!is_8bit_scale) {
          reinterpret_cast<int4*>(&frag_s[k % 2])[0] =
              sh_s_stage[s_sh_rd + cur_group_id * s_sh_stride];
        } else {
          reinterpret_cast<int2*>(&frag_s[k % 2])[0] =
              reinterpret_cast<int2*>(sh_s_stage)[s_sh_rd + cur_group_id * (2 * s_sh_stride)];
        }
      } else if (group_blocks >= b_sh_wr_iters) {
        if constexpr (!is_8bit_scale) {
          reinterpret_cast<int4*>(&frag_s[1])[0] = reinterpret_cast<int4*>(&frag_s[0])[0];
        } else {
          reinterpret_cast<int2*>(&frag_s[1])[0] = reinterpret_cast<int2*>(&frag_s[0])[0];
        }
      }
    }

    return;
  }

  // Act-order case

  // Determine K of the "current" thread-block
  int cur_k = slice_k_start + tb_k * full_pipe;
  if (cur_k >= prob_k || cur_k >= slice_k_finish) {
    return;
  }

  // Reset (to current thread-block) since we read g_idx portion from the
  // shared memory
  cur_k = 0;

  // Progress to current iteration
  cur_k += k % b_sh_wr_iters;

  // Determine "position" inside the thread-block (based on warp and
  // thread-id)
  auto warp_id = threadIdx.x / 32;
  int warp_row = warp_id / tb_n_warps;
  int warp_col = warp_id % tb_n_warps;

  cur_k += warp_row * 16 * b_sh_wr_iters;

  auto th_id = threadIdx.x % 32;
  cur_k += (th_id % 4) * 2;  // Due to tensor-core layout for fp16 B matrix

  int s_col_shift =
      /*slice_n_offset +*/ (act_s_col_warp_stride * warp_col) + (th_id / 4) * act_s_col_stride;

  if (is_same_group[pipe]) {
    if (k % 2 == 0) {
      *(reinterpret_cast<int4*>(&(act_frag_s[k % 2][0][0]))) =
          sh_s[(same_group_id[pipe] - sh_first_group_id) * s_sh_stride + s_col_shift];
    } else {
      *(reinterpret_cast<int4*>(&(act_frag_s[k % 2][0][0]))) =
          *(reinterpret_cast<int4*>(&(act_frag_s[(k - 1) % 2][0][0])));
    }

    for (int i = 1; i < 4; i++) {
      *(reinterpret_cast<int4*>(&(act_frag_s[k % 2][i][0]))) =
          *(reinterpret_cast<int4*>(&(act_frag_s[k % 2][0][0])));
    }
    return;
  }

  int4* sh_g_idx_stage = sh_g_idx + g_idx_stage * pipe;
  int* sh_g_idx_int_ptr = reinterpret_cast<int*>(sh_g_idx_stage);

  constexpr int k_frag_offsets[4] = {0, 1, 8, 9};  // Tensor core offsets per thread

#pragma unroll
  for (int i = 0; i < 4; i++) {
    int actual_k = cur_k + k_frag_offsets[i];

    int group_id = sh_g_idx_int_ptr[actual_k];
    int rel_group_id = group_id - sh_first_group_id;

    *(reinterpret_cast<int4*>(&(act_frag_s[k % 2][i][0]))) =
        sh_s[rel_group_id * s_sh_stride + s_col_shift];
  }
}

template <typename Config, typename Ref_zp_sh_rd, typename Ref_sh_zp, typename Ref_frag_qzp,
          typename Ref_frag_zpf>
__device__ __forceinline__ void marlin_fetch_zp_to_registers(Ref_zp_sh_rd& zp_sh_rd,
                                                             Ref_sh_zp& sh_zp,
                                                             Ref_frag_qzp& frag_qzp,
                                                             Ref_frag_zpf& frag_zpf, int k,
                                                             int full_pipe) {
  static constexpr auto is_a_8bit = Config::is_a_8bit;
  static constexpr auto has_zp = Config::has_zp;
  static constexpr auto b_sh_wr_iters = Config::b_sh_wr_iters;
  static constexpr auto tb_n_warps = Config::tb_n_warps;
  static constexpr auto zp_sh_stride = Config::zp_sh_stride;
  static constexpr auto zp_sh_stage = Config::zp_sh_stage;
  static constexpr auto num_ints_per_thread = Config::num_ints_per_thread;
  static constexpr auto thread_k_blocks = Config::thread_k_blocks;
  static constexpr auto stages = Config::stages;
  static constexpr auto group_blocks = Config::group_blocks;
  static constexpr auto is_zp_float = Config::is_zp_float;

  // This code does not handle group_blocks == 0,
  // which signifies act_order.
  // has_zp implies AWQ, which doesn't have act_order,
  static_assert(!has_zp || group_blocks != 0);

  if constexpr (has_zp && !is_zp_float) {
    int pipe = full_pipe % stages;

    if constexpr (group_blocks == -1) {
      // load only when starting a new slice
      if (k == 0 && full_pipe == 0 || is_a_8bit) {
#pragma unroll
        for (int i = 0; i < num_ints_per_thread; i++) {
          frag_qzp[k % 2][i] = (reinterpret_cast<int*>(sh_zp))[zp_sh_rd + i];
        }
      }
    } else if constexpr (group_blocks >= thread_k_blocks) {
      constexpr int g = group_blocks / thread_k_blocks;
      // A single-subtile A8 loop keeps one register zero fragment for the
      // whole group. Re-reading its shared stage after prefetch has recycled
      // that stage can race with the next group's asynchronous zero load.
      if ((pipe % g == 0 && k % b_sh_wr_iters == 0) ||
          (is_a_8bit && b_sh_wr_iters != 1)) {
        int4* sh_zp_stage = sh_zp + zp_sh_stage * (g * (pipe / g));
#pragma unroll
        for (int i = 0; i < num_ints_per_thread; i++) {
          frag_qzp[k % 2][i] = (reinterpret_cast<int*>(sh_zp_stage))[zp_sh_rd + i];
        }
      }
    } else {
      auto warp_id = threadIdx.x / 32;

      int warp_row = warp_id / tb_n_warps;

      int k_blocks = b_sh_wr_iters * warp_row + k % b_sh_wr_iters;
      int cur_group_id = k_blocks / div_ceil(group_blocks, is_a_8bit ? 2 : 1);

      int4* sh_zp_stage = sh_zp + zp_sh_stage * pipe;

      sh_zp_stage += cur_group_id * zp_sh_stride;

#pragma unroll
      for (int i = 0; i < num_ints_per_thread; i++) {
        frag_qzp[k % 2][i] = (reinterpret_cast<int*>(sh_zp_stage))[zp_sh_rd + i];
      }
    }
  }

  else if constexpr (has_zp && is_zp_float) {
    int pipe = full_pipe % stages;

    if constexpr (group_blocks != -1) {
      if constexpr (group_blocks >= thread_k_blocks) {
        constexpr int g = group_blocks / thread_k_blocks;
        if (pipe % g == 0 && k % b_sh_wr_iters == 0) {
          int4* sh_zp_stage = sh_zp + zp_sh_stage * (g * (pipe / g));
          reinterpret_cast<int4*>(&frag_zpf[k % 2])[0] = sh_zp_stage[zp_sh_rd];
        }
      } else if (group_blocks < b_sh_wr_iters || k % b_sh_wr_iters == 0) {
        auto warp_id = threadIdx.x / 32;

        int warp_row = warp_id / tb_n_warps;
        int k_blocks = b_sh_wr_iters * warp_row + k % b_sh_wr_iters;
        int cur_group_id = k_blocks / group_blocks;

        int4* sh_zp_stage = sh_zp + zp_sh_stage * pipe;

        reinterpret_cast<int4*>(&frag_zpf[k % 2])[0] =
            sh_zp_stage[zp_sh_rd + cur_group_id * zp_sh_stride];
      }
    }
  }
}

}  // namespace MARLIN_NAMESPACE_NAME
