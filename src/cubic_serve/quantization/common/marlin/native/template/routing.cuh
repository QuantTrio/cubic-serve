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

template <typename Config, typename Ref_sorted_token_ids_ptr, typename Ref_topk_weights_ptr,
          typename Ref_top_k, typename Ref_mul_topk_weights, typename Ref_global_scale_f32,
          typename Ref_sh_block_sorted_ids_int4, typename Ref_sh_new,
          typename Ref_sh_block_sorted_ids, typename Ref_sh_rd_block_sorted_ids,
          typename Ref_sh_block_topk_weights, typename Ref_block_num_valid_tokens,
          typename Ref_prob_m_top_k>
__device__ __forceinline__ void marlin_read_moe_block_data(
    Ref_sorted_token_ids_ptr& sorted_token_ids_ptr, Ref_topk_weights_ptr& topk_weights_ptr,
    Ref_top_k& top_k, Ref_mul_topk_weights& mul_topk_weights,
    Ref_global_scale_f32& global_scale_f32, Ref_sh_block_sorted_ids_int4& sh_block_sorted_ids_int4,
    Ref_sh_new& sh_new, Ref_sh_block_sorted_ids& sh_block_sorted_ids,
    Ref_sh_rd_block_sorted_ids& sh_rd_block_sorted_ids,
    Ref_sh_block_topk_weights& sh_block_topk_weights,
    Ref_block_num_valid_tokens& block_num_valid_tokens, Ref_prob_m_top_k& prob_m_top_k,
    int block_id) {
  using Cdtype = typename Config::Cdtype;
  using c_scalar_t2 = typename Config::c_scalar_t2;
  static constexpr auto moe_block_size = Config::moe_block_size;
  static constexpr auto b_type = Config::b_type;
  static constexpr auto s_type = Config::s_type;
  static constexpr auto threads = Config::threads;

  block_num_valid_tokens = moe_block_size;

  cp_async4_pred(sh_block_sorted_ids_int4 + threadIdx.x,
                 reinterpret_cast<const int4*>(sorted_token_ids_ptr) +
                     (block_id * moe_block_size / 4 + threadIdx.x),
                 threadIdx.x < moe_block_size / 4);

  cp_async_fence();
  cp_async_wait<0>();

  __syncthreads();

  if (threadIdx.x >= threads - 32) {
    constexpr int size_per_thread = div_ceil(moe_block_size, 32);
    int lane_id = threadIdx.x - (threads - 32);

    int local_count = 0;
#pragma unroll
    for (int i = 0; i < size_per_thread; i++) {
      int j = lane_id * size_per_thread + i;
      if (j < moe_block_size) {
        int idx = sh_block_sorted_ids[j];
        if (idx < prob_m_top_k) local_count++;
      }
    }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 750

    if constexpr (moe_block_size >= 16)
      local_count += __shfl_down_sync(0xFFFFFFFF, local_count, 16);
    if constexpr (moe_block_size >= 8) local_count += __shfl_down_sync(0xFFFFFFFF, local_count, 8);
    if constexpr (moe_block_size >= 4) local_count += __shfl_down_sync(0xFFFFFFFF, local_count, 4);
    if constexpr (moe_block_size >= 2) local_count += __shfl_down_sync(0xFFFFFFFF, local_count, 2);

    local_count += __shfl_down_sync(0xFFFFFFFF, local_count, 1);
    block_num_valid_tokens = local_count;
#else
    block_num_valid_tokens = __reduce_add_sync(0xffffffff, local_count);
#endif

    if (lane_id == 0) reinterpret_cast<int*>(sh_new)[0] = block_num_valid_tokens;
  }

  if (threadIdx.x < moe_block_size) {
    int idx = sh_block_sorted_ids[threadIdx.x];
    sh_rd_block_sorted_ids[threadIdx.x] = idx / top_k;

    if (mul_topk_weights) {
      idx = idx < prob_m_top_k ? idx : 0;
      float topk_weight_tmp = topk_weights_ptr[idx];
      if constexpr (b_type == vllm::kFE2M1f && s_type == vllm::kFE4M3fn) {
        topk_weight_tmp *= global_scale_f32;
      }
      c_scalar_t2 topk_weight_val = Cdtype::num2num2(Cdtype::float2num(topk_weight_tmp));
      sh_block_topk_weights[threadIdx.x] = topk_weight_val;
    }
  }

  __syncthreads();

  block_num_valid_tokens = reinterpret_cast<int*>(sh_new)[0];
  __syncthreads();
}

template <typename Config, typename Ref_C, typename Ref_a_scales_ptr, typename Ref_prob_n,
          typename Ref_locks, typename Ref_use_atomic_add, typename Ref_k_tiles,
          typename Ref_n_tiles, typename Ref_part2_mn_tiles, typename Ref_iters,
          typename Ref_slice_row, typename Ref_slice_col_par, typename Ref_slice_col,
          typename Ref_slice_iters, typename Ref_slice_count, typename Ref_slice_idx,
          typename Ref_par_id, typename Ref_sh_a_s, typename Ref_sh_block_sorted_ids,
          typename Ref_sh_rd_block_sorted_ids, typename Ref_block_num_valid_tokens,
          typename Ref_locks_off, typename Ref_update_next_moe_block_data, typename Ref_first_init>
__device__ __forceinline__ void marlin_init_part2_slice(
    Ref_C& C, Ref_a_scales_ptr& a_scales_ptr, Ref_prob_n& prob_n, Ref_locks& locks,
    Ref_use_atomic_add& use_atomic_add, Ref_k_tiles& k_tiles, Ref_n_tiles& n_tiles,
    Ref_part2_mn_tiles& part2_mn_tiles, Ref_iters& iters, Ref_slice_row& slice_row,
    Ref_slice_col_par& slice_col_par, Ref_slice_col& slice_col, Ref_slice_iters& slice_iters,
    Ref_slice_count& slice_count, Ref_slice_idx& slice_idx, Ref_par_id& par_id, Ref_sh_a_s& sh_a_s,
    Ref_sh_block_sorted_ids& sh_block_sorted_ids,
    Ref_sh_rd_block_sorted_ids& sh_rd_block_sorted_ids,
    Ref_block_num_valid_tokens& block_num_valid_tokens, Ref_locks_off& locks_off,
    Ref_update_next_moe_block_data& update_next_moe_block_data, Ref_first_init& first_init) {
  static constexpr auto is_a_8bit = Config::is_a_8bit;
  static constexpr auto threads = Config::threads;
  static constexpr auto thread_n_blocks = Config::thread_n_blocks;

  slice_iters = iters * (blockIdx.x + 1) - (k_tiles * slice_col_par + slice_row);
  if (slice_iters < 0 || slice_col_par >= part2_mn_tiles) slice_iters = 0;
  if (slice_iters == 0) return;
  if (slice_row + slice_iters > k_tiles) slice_iters = k_tiles - slice_row;
  slice_count = 1;
  slice_idx = 0;
  int col_first = iters * div_ceil(k_tiles * slice_col_par, iters);
  if (col_first <= k_tiles * (slice_col_par + 1)) {
    int col_off = col_first - k_tiles * slice_col_par;
    slice_count = div_ceil(k_tiles - col_off, iters);
    if (col_off > 0) slice_count++;
    int delta_first = iters * blockIdx.x - col_first;
    if (delta_first < 0 || (col_off == 0 && delta_first == 0))
      slice_idx = slice_count - 1;
    else {
      slice_idx = slice_count - 1 - delta_first / iters;
      if (col_off > 0) slice_idx--;
    }
  }
  if (part2_mn_tiles >= gridDim.x) {
    if (slice_count > 1 && slice_idx == slice_count - 1) {
      locks_off++;
    }
  } else {
    locks_off++;
  }

  if (first_init && use_atomic_add && slice_count > 1 && slice_idx == 0) {
    constexpr int threads_per_m = 16 * thread_n_blocks / 8;
    int m_per_thread = div_ceil(block_num_valid_tokens, threads / threads_per_m);
    for (int i = 0; i < m_per_thread; i++) {
      int row = threads / threads_per_m * i + threadIdx.x / threads_per_m;
      if (row < block_num_valid_tokens) {
        int64_t sorted_row = sh_block_sorted_ids[row];
        int col = slice_col * 16 * thread_n_blocks / 8 + threadIdx.x % threads_per_m;
        C[sorted_row * prob_n / 8 + col] = {0, 0, 0, 0};
      }
    }
    // After write zero to output, write a negative value to lock.
    // Every SM that processes the same slice would wait for
    // the negative value, and then atomicAdd 1 to it.
    // After all SMs are processed, the lock value would back to 0 again.
    __syncthreads();
    if (threadIdx.x == 0) locks[locks_off] = 1 - slice_count;
  }

  if (slice_col == n_tiles) {
    slice_col = 0;
    par_id++;
    update_next_moe_block_data();
  }
  if (is_a_8bit && (first_init || slice_col == 0)) {
    __syncthreads();
    cp_async1_ca_pred(&sh_a_s[threadIdx.x], &a_scales_ptr[sh_rd_block_sorted_ids[threadIdx.x]],
                      threadIdx.x < block_num_valid_tokens);
  }
}

}  // namespace MARLIN_NAMESPACE_NAME
