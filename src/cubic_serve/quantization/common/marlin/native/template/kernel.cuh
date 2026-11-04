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

#include "epilogue.cuh"
#include "execute.cuh"
#include "packed_codes.cuh"
#include "metadata.cuh"
#include "mma.cuh"
#include "operand_pipeline.cuh"
#include "reduction.cuh"
#include "routing.cuh"
namespace MARLIN_NAMESPACE_NAME {
template <const vllm::ScalarTypeId a_type_id,  // A ScalarType id
          const vllm::ScalarTypeId b_type_id,  // B ScalarType id
          const vllm::ScalarTypeId c_type_id,  // C ScalarType id
          const vllm::ScalarTypeId s_type_id,  // B_SCALE ScalarType id
          const int threads,                   // number of threads in a threadblock
          const int thread_m_blocks,           // number of 16x16 blocks in the m
                                               // dimension (batchsize) of the
                                               // threadblock
          const int thread_n_blocks,           // same for n dimension (output)
          const int thread_k_blocks,           // same for k dimension (reduction)
          const bool m_block_size_8,           // whether m_block_size == 8
                                               // only works when thread_m_blocks == 1
          const int stages,                    // number of stages for the async global->shared
                                               // fetch pipeline
          const int group_blocks,              // number of consecutive 16x16 blocks
                                               // with a separate quantization scale
          const bool is_zp_float               // is zero point of float16 type?
          >
__global__ void Marlin(const int4* __restrict__ A,  // fp16 input matrix of shape mxk
                       const int4* __restrict__ B,  // 4bit quantized weight matrix of shape kxn
                       int4* __restrict__ C,        // fp16 output buffer of shape mxn
                       int4* __restrict__ C_tmp,    // fp32 tmp output buffer (for reduce)
                       const int4* __restrict__ b_bias_ptr,
                       // float scales of input matrix, only used when is_a_8bit == true.
                       // shape (m,)
                       const float* __restrict__ a_scales_ptr,
                       // fp16 quantization scales. shape (k/groupsize, n)
                       const int4* __restrict__ scales_ptr,
                       // fp16 global scale (for nvfp4// only)
                       const float* __restrict__ global_scale_ptr,
                       // 4bit packed zero-points of shape
                       // (k/groupsize, n/pack_factor)
                       const int4* __restrict__ zp_ptr,
                       // int32 group indices of shape k
                       const int* __restrict__ g_idx,
                       const int32_t* __restrict__ sorted_token_ids_ptr,        // moe sorted_ids
                       const int32_t* __restrict__ expert_ids_ptr,              // moe expert ids
                       const int32_t* __restrict__ num_tokens_past_padded_ptr,  // moe num tokens
                       const float* __restrict__ topk_weights_ptr,              // moe top weights
                       int top_k,              // num of experts per token
                       bool mul_topk_weights,  // mul topk weights or not
                       int num_groups,         // number of scale groups per output channel
                       int prob_m,             // batch dimension m
                       int prob_n,             // output dimension n
                       int prob_k,             // reduction dimension k
                       int* locks,             // extra global storage for barrier synchronization
                       bool has_bias,
                       bool use_atomic_add,  // whether to use atomic add to reduce
                       bool use_fp32_reduce,  // whether to use fp32 global reduce
                       int runtime_partitions  // sealed contiguous K-slice count
) {
  using Config = MarlinConfiguration<a_type_id, b_type_id, c_type_id, s_type_id, threads,
                                     thread_m_blocks, thread_n_blocks, thread_k_blocks,
                                     m_block_size_8, stages, group_blocks, is_zp_float>;
  // Matrix dimensions are launch operands, shared across checkpoint/TP shapes.
#ifdef CUBIC_MARLIN_UNWEIGHTED
  // Route weighting is an external stored boundary for this binding.
  mul_topk_weights = false;
  has_bias = false;
  use_atomic_add = false;
  use_fp32_reduce = true;
#endif
  if constexpr (Config::dense_linear) {
    top_k = 1;
    mul_topk_weights = false;
    has_bias = false;
    use_atomic_add = false;
    use_fp32_reduce = true;
  }
  using scalar_t = typename Config::scalar_t;
  using scalar_32bit_t = typename Config::scalar_32bit_t;
  using c_scalar_t = typename Config::c_scalar_t;
  using c_scalar_t2 = typename Config::c_scalar_t2;
  using FragA = typename Config::FragA;
  using FragC = typename Config::FragC;
  using FragS = typename Config::FragS;
  using FragZP = typename Config::FragZP;
  static constexpr auto moe_block_size = Config::moe_block_size;
  static constexpr auto metadata_planes = Config::metadata_planes;
  static constexpr auto b_type = Config::b_type;
  static constexpr auto s_type = Config::s_type;
  static constexpr auto is_a_8bit = Config::is_a_8bit;
  static constexpr auto has_zp = Config::has_zp;
  static constexpr auto is_8bit_scale = Config::is_8bit_scale;
  static constexpr auto dequant_skip_flop = Config::dequant_skip_flop;
  static constexpr auto has_act_order = Config::has_act_order;
  static constexpr auto pack_factor = Config::pack_factor;
  static constexpr auto a_sh_stride = Config::a_sh_stride;
  static constexpr auto a_gl_rd_delta_o = Config::a_gl_rd_delta_o;
  static constexpr auto a_sh_wr_delta = Config::a_sh_wr_delta;
  static constexpr auto a_sh_rd_delta_i = Config::a_sh_rd_delta_i;
  static constexpr auto a_sh_wr_iters = Config::a_sh_wr_iters;
  static constexpr auto b_sh_stride = Config::b_sh_stride;
  static constexpr auto b_thread_vecs = Config::b_thread_vecs;
  static constexpr auto b_sh_stage = Config::b_sh_stage;
  static constexpr auto b_sh_wr_iters = Config::b_sh_wr_iters;
  static constexpr auto s_sh_stride = Config::s_sh_stride;
  static constexpr auto s_sh_stage = Config::s_sh_stage;
  static constexpr auto tb_k = Config::tb_k;
  static constexpr auto g_idx_stage = Config::g_idx_stage;
  static constexpr auto tb_n_warps = Config::tb_n_warps;
  static constexpr auto zp_sh_stride = Config::zp_sh_stride;
  static constexpr auto zp_sh_stage = Config::zp_sh_stage;
  static constexpr auto num_col_threads = Config::num_col_threads;
  static constexpr auto num_row_threads = Config::num_row_threads;
  static constexpr auto num_ints_per_thread = Config::num_ints_per_thread;
  static constexpr auto sh_size_b_red_min = Config::sh_size_b_red_min;
  static constexpr auto sh_b_red_bias_size = Config::sh_b_red_bias_size;
  static constexpr auto sh_s_size = Config::sh_s_size;
  // Each threadblock processes one "stripe" of the B matrix with (roughly) the
  // same size, which might involve multiple column "slices" (of width 16 *
  // `thread_n_blocks`). Stripes are defined as shown in the 3x3 matrix 5 SM
  // example:
  //   0 1 3
  //   0 2 3
  //   1 2 4
  // While this kind of partitioning makes things somewhat more complicated, it
  // ensures good utilization of all SMs for many kinds of shape and GPU
  // configurations, while requiring as few slow global cross-threadblock
  // reductions as possible.

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 890
  // FP8 computation is only supported for Ada Lovelace or newer architectures.
  if constexpr (a_type_id == vllm::kFE4M3fn.id()) return;
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 750
  // Turing TensorCore only supports fp16 and int8
  if constexpr (a_type_id != vllm::kFloat16.id() && a_type_id != vllm::kS8.id()) return;
#endif

  int num_tokens_past_padded;
  if constexpr (Config::dense_linear) {
    num_tokens_past_padded = div_ceil(prob_m, moe_block_size) * moe_block_size;
  } else {
    num_tokens_past_padded = num_tokens_past_padded_ptr[0];
  }

  extern __shared__ int4 sh[];

  if constexpr (b_type == vllm::kFE2M1f) {
    static_assert(s_type == vllm::kFE4M3fn && group_blocks == 1 ||
                  s_type == vllm::kFE8M0fnu && group_blocks == 2);
  } else if constexpr (b_type == vllm::kFE4M3fn && s_type == vllm::kFE8M0fnu) {
    static_assert(group_blocks == 2);
  } else if constexpr (std::is_same<scalar_t, nv_bfloat16>::value) {
    static_assert(s_type == vllm::kFloat16);
  } else if constexpr (std::is_same<scalar_t, half>::value) {
    static_assert(s_type == vllm::kFloat16);
  }

  if constexpr (!is_a_8bit) {
    static_assert(std::is_same<scalar_t, c_scalar_t>::value);
  }

  // see comments of dequant.h for more details

  float global_scale_f32 = 1.0f;

  static_assert(thread_m_blocks == 1 || !m_block_size_8);
  const int group_size = (!has_act_order && group_blocks == -1) ? prob_k : prob_k / num_groups;
  const int scales_expert_stride = prob_n * prob_k / group_size / (is_8bit_scale ? 16 : 8);
  const int zp_expert_stride = is_zp_float ? prob_n * prob_k / group_size / 8
                                           : prob_n * prob_k / group_size / (pack_factor * 4);
  const int b_bias_expert_stride = prob_n / 8;

  // parallel: num valid moe blocks
  int parallel = num_tokens_past_padded / moe_block_size;

  int k_tiles = prob_k / 16 / thread_k_blocks;
  int n_tiles = prob_n / 16 / thread_n_blocks;

  int global_mn_tiles = parallel * n_tiles;
  int part2_mn_tiles = global_mn_tiles;
  int part1_mn_iters = 0;
  bool in_part2 = false;

  // we use DP + two-tile SK here
  // part1: DP
  // part2: two-tile SK
  // see https://github.com/vllm-project/vllm/pull/24722 for more details
  if (!Config::fixed_partitions && global_mn_tiles > gridDim.x) {
    part2_mn_tiles = global_mn_tiles % gridDim.x;
    if (part2_mn_tiles * 3 <= gridDim.x) part2_mn_tiles += gridDim.x;
    part1_mn_iters = (global_mn_tiles - part2_mn_tiles) / gridDim.x;
  }

  int iters = Config::fixed_partitions
                  ? k_tiles / runtime_partitions
                  : div_ceil(part2_mn_tiles, gridDim.x) * k_tiles;

  if constexpr (!has_act_order && group_blocks != -1) {
    if (group_blocks >= thread_k_blocks) {
      // Ensure that the number of tiles in each stripe is a multiple of the
      // groupsize; this avoids an annoying special case where a stripe starts
      // in the middle of group.
      iters = (group_blocks / thread_k_blocks) * div_ceil(iters, (group_blocks / thread_k_blocks));
    }
  }

  int slice_row = 0;
  int slice_col_par = blockIdx.x;
  int slice_col;
  int slice_iters = k_tiles;  // number of threadblock tiles in the current slice
  // total number of active threadblocks in the current slice
  int slice_count = 1;
  // index of threadblock in current slice; numbered bottom to top
  int slice_idx = 0;

  int par_id = 0;
  int block_id = -1;
  int64_t expert_id = 0;  // use int64 to avoid computation result overflow
  int old_expert_id = 0;
  int64_t B_expert_off = 0;

  float* sh_a_s = reinterpret_cast<float*>(sh);
  int4* sh_block_sorted_ids_int4 = sh + (is_a_8bit ? (4 * thread_m_blocks) : 0);
  int4* sh_rd_block_sorted_ids_int4 = sh_block_sorted_ids_int4 + moe_block_size / 4;
  int4* sh_block_topk_weights_int4 = sh_rd_block_sorted_ids_int4 + moe_block_size / 4;
  // sh_block_topk_weights_int4 only need (moe_block_size / 4);
  // but we pad to align to 256 bytes
  // Dense rows are an affine identity map, not routed expert metadata.
  // Keep activation-scale storage when present; only routed rows need IDs
  // and top-k scores. Shared operand/reduction lifetime barriers stay intact.
  int4* sh_new = Config::dense_linear ? sh_block_sorted_ids_int4
                                    : sh_block_topk_weights_int4 + moe_block_size / 2;
  struct DenseRows {
    int base;
    __device__ __forceinline__ int operator[](int index) const { return base + index; }
  };
  auto row_map = [](int32_t* storage) {
    if constexpr (Config::dense_linear) return DenseRows{0};
    else return storage;
  };
  // Routed helpers retain their original pointer type and storage contract.
  auto sh_block_sorted_ids = row_map(reinterpret_cast<int*>(sh_block_sorted_ids_int4));
  auto sh_rd_block_sorted_ids = row_map(reinterpret_cast<int*>(sh_rd_block_sorted_ids_int4));
  c_scalar_t2* sh_block_topk_weights = reinterpret_cast<c_scalar_t2*>(sh_block_topk_weights_int4);

  int32_t block_num_valid_tokens = 0;
  int32_t locks_off = 0;

  // We can easily implement parallel problem execution by just remapping
  // indices and advancing global pointers
  if (part2_mn_tiles >= gridDim.x) {
    // when part2_mn_tiles >= sms
    // then there are at most $sms$ conflict tile blocks
    locks_off = blockIdx.x;
  } else {
    locks_off = (iters * blockIdx.x) / k_tiles - 1;
  }

  int prob_m_top_k = prob_m * top_k;
  // read moe block data given block_id
  // block_sorted_ids / block_num_valid_tokens / block_topk_weights
  auto read_moe_block_data = [&](int block_id) {
    if constexpr (Config::dense_linear) {
      block_num_valid_tokens = min(moe_block_size, max(0, prob_m - block_id * moe_block_size));
      sh_block_sorted_ids.base = block_id * moe_block_size;
      sh_rd_block_sorted_ids.base = block_id * moe_block_size;
      // The same input/epilogue helpers consume these rows. Keep their shared
      // lifetime barrier, including when this CTA advances to another tile.
      __syncthreads();
    } else {
      marlin_read_moe_block_data<Config>(
          sorted_token_ids_ptr, topk_weights_ptr, top_k, mul_topk_weights, global_scale_f32,
          sh_block_sorted_ids_int4, sh_new, sh_block_sorted_ids, sh_rd_block_sorted_ids,
          sh_block_topk_weights, block_num_valid_tokens, prob_m_top_k, block_id);
    }
  };

  // when move to next moe block, find the next block_id and expert_id
  // and then read moe block data
  auto update_next_moe_block_data = [&]() {
    if (par_id >= parallel) return;

    old_expert_id = expert_id;
    block_id = par_id;
    if constexpr (Config::dense_linear) {
      expert_id = 0;
    } else {
      expert_id = expert_ids_ptr[block_id];
    }

    if constexpr (b_type == vllm::kFE2M1f && s_type == vllm::kFE4M3fn) {
      global_scale_f32 = global_scale_ptr[expert_id];
    }

    B_expert_off = expert_id * prob_n * prob_k / (pack_factor * 4);
    scales_ptr += (expert_id - old_expert_id) * scales_expert_stride;
    if constexpr (has_zp) {
      zp_ptr += (expert_id - old_expert_id) * zp_expert_stride;
    }
    if constexpr (has_act_order) {
      g_idx += (expert_id - old_expert_id) * prob_k;
    }
    if (has_bias) {
      b_bias_ptr += (expert_id - old_expert_id) * b_bias_expert_stride;
    }

    read_moe_block_data(block_id);
  };

  // Compute all information about the current slice which is required for
  // synchronization.
  bool first_init = true;
  auto init_part2_slice = [&]() {
    marlin_init_part2_slice<Config>(
        C, a_scales_ptr, prob_n, locks, use_atomic_add, k_tiles, n_tiles, part2_mn_tiles, iters,
        slice_row, slice_col_par, slice_col, slice_iters, slice_count, slice_idx, par_id, sh_a_s,
        sh_block_sorted_ids, sh_rd_block_sorted_ids, block_num_valid_tokens, locks_off,
        update_next_moe_block_data, first_init);
  };

  auto init_part1_slice = [&]() {
    if (part1_mn_iters) {
      part1_mn_iters--;
      par_id = slice_col_par / n_tiles;
      slice_col = slice_col_par % n_tiles;
      slice_iters = k_tiles;
      update_next_moe_block_data();
      if (is_a_8bit) {
        __syncthreads();
        cp_async1_ca_pred(&sh_a_s[threadIdx.x], &a_scales_ptr[sh_rd_block_sorted_ids[threadIdx.x]],
                          threadIdx.x < block_num_valid_tokens);
      }
    }
  };

  auto init_slice = [&]() {
    if (!in_part2 && !part1_mn_iters) {
      in_part2 = true;
      slice_col_par = (iters * blockIdx.x) / k_tiles;
      slice_row = (iters * blockIdx.x) % k_tiles;
      slice_col = (slice_col_par + global_mn_tiles - part2_mn_tiles) % n_tiles;
      par_id = (slice_col_par + global_mn_tiles - part2_mn_tiles) / n_tiles;
      update_next_moe_block_data();
    }
    if (!in_part2) {
      init_part1_slice();
    } else {
      init_part2_slice();
      first_init = false;
    }
  };

  init_slice();

  // A sizes/strides

  // stride of the A matrix in global memory
  int a_gl_stride = prob_k / (is_a_8bit ? 16 : 8);
  // stride of an A matrix tile in shared memory

  // delta between subsequent A tiles in global memory

  // between subsequent accesses within a tile
  int a_gl_rd_delta_i = a_gl_stride * (threads / a_gl_rd_delta_o);
  // between shared memory writes

  // within a shared memory tile

  // overall size of a tile

  // number of shared write iterations for a tile

  // B sizes/strides
  int b_gl_stride = 16 * prob_n / (pack_factor * (is_a_8bit ? 2 : 4));

  int b_gl_rd_delta_o = b_gl_stride * thread_k_blocks / (is_a_8bit ? 2 : 1);

  static_assert(b_sh_wr_iters == 1 ||
                (b_sh_wr_iters >= 2 && b_sh_wr_iters % 2 == 0),
                "Marlin register pipeline needs an even number of K subtiles");
  if constexpr (group_blocks > 0) {
    static_assert((group_blocks < thread_k_blocks ? thread_k_blocks % group_blocks
                                                  : group_blocks % thread_k_blocks) == 0,
                  "Marlin scale groups and K tiles must have aligned boundaries");
  }
  static_assert(
      group_blocks <= thread_k_blocks ||
          stages % (group_blocks > thread_k_blocks ? group_blocks / thread_k_blocks : 1) == 0,
      "Marlin group reuse must close within the pipeline ring");

  // Scale sizes/strides without act_order
  int s_gl_stride = prob_n / (is_8bit_scale ? 16 : 8);

  int s_gl_rd_delta = s_gl_stride;

  // Scale size/strides with act_order

  // constexpr int act_s_row_stride      = 1;
  // int           act_s_col_stride      = act_s_row_stride * num_groups;

  int act_s_col_stride = 1;
  int act_s_col_warp_stride = act_s_col_stride * 8;

  int act_s_col_tb_stride = act_s_col_warp_stride * tb_n_warps;

  // Zero-points sizes/strides
  int zp_gl_stride = is_zp_float ? prob_n / 8 : (prob_n / pack_factor) / 4;

  int zp_gl_rd_delta = zp_gl_stride;

  // Global A read index of current thread.
  int a_gl_rd_row = threadIdx.x / a_gl_rd_delta_o;
  int a_gl_rd_col = a_gl_rd_delta_o * slice_row + threadIdx.x % a_gl_rd_delta_o;
  // Shared write index of current thread.
  int a_sh_wr = a_sh_stride * (threadIdx.x / a_gl_rd_delta_o) + (threadIdx.x % a_gl_rd_delta_o);
  // Shared read index.
  int a_sh_rd = a_sh_stride * ((threadIdx.x % 32) % (16 / (m_block_size_8 ? 2 : 1))) +
                (threadIdx.x % 32) / (16 / (m_block_size_8 ? 2 : 1));
  a_sh_rd += 2 * ((threadIdx.x / 32) / tb_n_warps) * b_sh_wr_iters;

  int b_gl_rd;
  if (threads <= b_sh_stride) {
    b_gl_rd = threadIdx.x;
  } else {
    b_gl_rd = b_gl_stride * (threadIdx.x / b_sh_stride) + (threadIdx.x % b_sh_stride);
  }

  b_gl_rd += B_expert_off + b_sh_stride * slice_col;
  b_gl_rd += b_gl_rd_delta_o * slice_row;
  auto b_sh_rd = threadIdx.x * b_thread_vecs;
  b_sh_rd += b_sh_rd / b_sh_stride * (b_sh_stride * (b_sh_wr_iters - 1));

  // For act_order
  int slice_k_start = tb_k * slice_row;
  int slice_k_finish = slice_k_start + tb_k * slice_iters;
  int slice_k_start_shared_fetch = slice_k_start;
  int slice_n_offset = act_s_col_tb_stride * slice_col;

  // No act_order
  int s_gl_rd;
  if constexpr (!has_act_order) {
    if constexpr (group_blocks == -1) {
      s_gl_rd = s_sh_stride * slice_col + threadIdx.x;
    } else if constexpr (group_blocks >= thread_k_blocks) {
      s_gl_rd = s_gl_stride * ((thread_k_blocks * slice_row) / group_blocks) +
                s_sh_stride * slice_col + threadIdx.x;
    } else {
      s_gl_rd =
          s_gl_stride * ((thread_k_blocks * slice_row) / group_blocks + threadIdx.x / s_sh_stride) +
          s_sh_stride * slice_col + threadIdx.x % s_sh_stride;
    }
  }
  auto s_sh_wr = threadIdx.x;
  bool s_sh_wr_pred = threadIdx.x < s_sh_stage;

  // Zero-points
  int zp_gl_rd;
  if constexpr (has_zp) {
    if constexpr (group_blocks == -1) {
      zp_gl_rd = zp_sh_stride * slice_col + threadIdx.x;
    } else if constexpr (group_blocks >= thread_k_blocks) {
      zp_gl_rd = zp_gl_stride * ((thread_k_blocks * slice_row) / group_blocks) +
                 zp_sh_stride * slice_col + threadIdx.x;
    } else {
      zp_gl_rd = zp_gl_stride *
                     ((thread_k_blocks * slice_row) / group_blocks + threadIdx.x / zp_sh_stride) +
                 zp_sh_stride * slice_col + threadIdx.x % zp_sh_stride;
    }
  }
  auto zp_sh_wr = threadIdx.x;
  bool zp_sh_wr_pred = zp_sh_stage > 0 && threadIdx.x < zp_sh_stage;

  // We use a different scale layout for grouped and column-wise quantization as
  // we scale a `half2` tile in column-major layout in the former and in
  // row-major in the latter case.
  int s_sh_rd;
  if constexpr (is_a_8bit) {
    s_sh_rd = 4 * ((threadIdx.x / 32) % tb_n_warps) + (threadIdx.x % 4);
  } else if constexpr (group_blocks != -1)
    s_sh_rd = 8 * ((threadIdx.x / 32) % tb_n_warps) + (threadIdx.x % 32) / 4;
  else if constexpr (group_blocks == -1 && (m_block_size_8 || (has_zp && !dequant_skip_flop)))
    s_sh_rd = 8 * ((threadIdx.x / 32) % tb_n_warps) + (threadIdx.x % 32) / 8;
  else
    s_sh_rd = 8 * ((threadIdx.x / 32) % tb_n_warps) + (threadIdx.x % 32) % 4;

  int bias_sh_rd;
  if constexpr (m_block_size_8) {
    bias_sh_rd = 8 * ((threadIdx.x / 32) % tb_n_warps) + (threadIdx.x % 32) / 8;
  } else {
    bias_sh_rd = (is_a_8bit ? 4 : 8) * ((threadIdx.x / 32) % tb_n_warps) + (threadIdx.x % 32) % 4;
  }

  int bias_sh_wr = threadIdx.x;
  int bias_gl_rd = (thread_n_blocks * 16 / 8) * slice_col + threadIdx.x;

  // Zero-points have the same read layout as the scales
  // (without column-wise case)

  int zp_sh_rd;
  if constexpr (has_zp) {
    if constexpr (is_zp_float) {
      if constexpr (group_blocks != -1) {
        zp_sh_rd = 8 * ((threadIdx.x / 32) % tb_n_warps) + (threadIdx.x % 32) / 4;
      }
    } else if (is_a_8bit) {
      zp_sh_rd = num_ints_per_thread * num_col_threads * ((threadIdx.x / 32) % tb_n_warps / 2) +
                 num_ints_per_thread * ((threadIdx.x % 32) / num_row_threads);
    } else {
      zp_sh_rd = num_ints_per_thread * num_col_threads * ((threadIdx.x / 32) % tb_n_warps) +
                 num_ints_per_thread * ((threadIdx.x % 32) / num_row_threads);
    }
  }

  // To ensure that writing and reading A tiles to/from shared memory, the
  // latter in fragment format, is fully bank conflict free, we need to use a
  // rather fancy XOR-based layout. The key here is that neither reads nor
  // writes of the 16-byte `int4` blocks of 8 consecutive threads involve the
  // same shared memory banks. Further, it seems (based on NSight-Compute) that
  // each warp must also write a consecutive memory segment?
  auto transform_a = [&](int i) {
    int row = i / a_gl_rd_delta_o;
    return a_gl_rd_delta_o * row + (i % a_gl_rd_delta_o) ^ (row % 8);
  };
  // Since the computation of this remapping is non-trivial and, due to our main
  // loop unrolls, all shared memory accesses are static, we simply precompute
  // both transformed reads and writes.
  int a_sh_wr_trans[a_sh_wr_iters];
#pragma unroll
  for (int i = 0; i < a_sh_wr_iters; i++)
    a_sh_wr_trans[i] = transform_a(a_sh_wr_delta * i + a_sh_wr);
  int a_sh_rd_trans[b_sh_wr_iters][thread_m_blocks];
#pragma unroll
  for (int i = 0; i < b_sh_wr_iters; i++) {
#pragma unroll
    for (int j = 0; j < thread_m_blocks; j++)
      a_sh_rd_trans[i][j] = transform_a(2 * i + a_sh_rd_delta_i * j + a_sh_rd);
  }

  // Since B-accesses have non-constant stride they have to be computed at
  // runtime; we break dependencies between subsequent accesses with a tile by
  // maintining multiple pointers (we have enough registers), a tiny
  // optimization.

  // Shared memory storage for global fetch pipelines.

  int4* sh_b = sh_new;
  int4* sh_red = sh_new;

  int4* sh_bias = sh_new + sh_size_b_red_min;
  int4* sh_g_idx = sh_new + sh_b_red_bias_size;
  int4* sh_zp = sh_g_idx + (stages * g_idx_stage);

  int4* sh_s = sh_zp + (stages * zp_sh_stage);
  int4* sh_a = sh_s + sh_s_size;

  // Register storage for double buffer of shared memory reads.
  FragA frag_a[Config::operand_buffers][thread_m_blocks];
  I4 frag_b_quant[Config::operand_buffers][b_thread_vecs];
  FragC frag_c[thread_m_blocks][is_a_8bit ? 2 : 4][2];
  FragC frag_c_tmp[thread_m_blocks][is_a_8bit ? 2 : 4][2];
  constexpr bool current_group_only = Config::compact_registers && metadata_planes &&
                                      group_blocks >= thread_k_blocks;
  float group_metadata[current_group_only ? 1 : 2][metadata_planes ? metadata_planes : 1][8];
  FragS frag_s[2][4];  // No act-order
  FragS frag_bias[2][4];
  FragS act_frag_s[2][4][4];             // For act-order
  int frag_qzp[2][num_ints_per_thread];  // Zero-points
  FragZP frag_zp;                        // Zero-points in fp16
  FragZP frag_zpf[2];                    // Zero-points in fp16 in HQQ

  if constexpr (is_a_8bit && group_blocks != -1) {
#pragma unroll
    for (int j = 0; j < 2; j++) {
#pragma unroll
      for (int i = 0; i < thread_m_blocks; i++) {
#pragma unroll
        for (int g = 0; g < 4; g++) {
          frag_c_tmp[i][j][0][g] = 0.0f;
        }

#pragma unroll
        for (int g = 0; g < 4; g++) {
          frag_c_tmp[i][j][1][g] = 0.0f;
        }
      }
    }
  }

  // Zero accumulators.
  auto zero_accums = [&]() {
#pragma unroll
    for (int i = 0; i < sizeof(frag_c) / sizeof(float); i++) reinterpret_cast<float*>(frag_c)[i] = 0;
  };

  int sh_first_group_id = -1;
  int sh_num_groups = -1;

  auto fetch_act_order_scales_to_shared = [&](bool is_async, int first_group_id,
                                              int last_group_id) {
    marlin_fetch_act_order_scales_to_shared<Config>(
        scales_ptr, num_groups, s_gl_stride, slice_n_offset, sh_s, sh_first_group_id, sh_num_groups,
        is_async, first_group_id, last_group_id);
  };
  // Asynchronously fetch the next A, B and s tile from global to the next
  // shared memory pipeline location.
  auto fetch_to_shared = [&](int pipe, int a_off, bool pred = true) {
    marlin_fetch_to_shared<Config>(
        A, B, scales_ptr, global_scale_ptr, zp_ptr, g_idx, prob_k, scales_expert_stride, expert_id,
        sh_rd_block_sorted_ids, block_num_valid_tokens, a_gl_stride, a_gl_rd_delta_i, b_gl_stride,
        b_gl_rd_delta_o, s_gl_rd_delta, zp_gl_rd_delta, a_gl_rd_row, a_gl_rd_col, b_gl_rd,
        slice_k_finish, slice_k_start_shared_fetch, s_gl_rd, s_sh_wr, s_sh_wr_pred, zp_gl_rd,
        zp_sh_wr, zp_sh_wr_pred, a_sh_wr_trans, sh_b, sh_g_idx, sh_zp, sh_s, sh_a, pipe, a_off,
        pred, slice_col * (16 * thread_n_blocks), prob_n);
  };

  auto fetch_col_zp_to_shared = [&]() {
    if (zp_sh_wr_pred) {
      cp_async4(&sh_zp[zp_sh_wr], &zp_ptr[zp_gl_rd]);
    }
  };

  auto fetch_col_scale_to_shared = [&]() {
    if (s_sh_wr_pred) {
      cp_async4(&sh_s[s_sh_wr], &scales_ptr[s_gl_rd]);
    }
  };

  // Wait until the next thread tile has been loaded to shared memory.
  auto wait_for_stage = [&]() {
    // We only have `stages - 2` active fetches since we are double buffering
    // and can only issue the next fetch when it is guaranteed that the previous
    // shared memory load is fully complete (as it may otherwise be
    // overwritten).
    cp_async_wait<stages - 2>();
    __syncthreads();
  };

  // Load the next sub-tile from the current location in the shared memory pipe
  // into the current register buffer.
  auto fetch_to_registers = [&](int k, int pipe) {
    int4* sh_a_stage = sh_a + moe_block_size * a_sh_stride * pipe;
#pragma unroll
    for (int i = 0; i < thread_m_blocks; i++)
      ldsm<m_block_size_8 ? 2 : 4, a_type_id>(frag_a[k % Config::operand_buffers][i],
                                              &sh_a_stage[a_sh_rd_trans[k % b_sh_wr_iters][i]]);
    int4* sh_b_stage = sh_b + Config::b_storage_stage * pipe;

#pragma unroll
    for (int i = 0; i < b_thread_vecs; i++) {
      int4 unpacked = load_code_fragment<Config>(
          sh_b_stage, b_sh_stride * (k % b_sh_wr_iters) + b_sh_rd + i);
      frag_b_quant[k % Config::operand_buffers][i] = reinterpret_cast<I4&>(unpacked);
    }
  };

  bool is_same_group[stages];
  int same_group_id[stages];

  auto init_same_group = [&](int pipe) {
    if constexpr (!has_act_order) {
      return;
    }

    int4* sh_g_idx_stage = sh_g_idx + g_idx_stage * pipe;
    int* sh_g_idx_int_ptr = reinterpret_cast<int*>(sh_g_idx_stage);

    int group_id_1 = sh_g_idx_int_ptr[0];
    int group_id_2 = sh_g_idx_int_ptr[tb_k - 1];

    is_same_group[pipe] = group_id_1 == group_id_2;
    same_group_id[pipe] = group_id_1;
  };

  auto fetch_scales_to_registers = [&](int k, int full_pipe) {
    marlin_fetch_scales_to_registers<Config>(prob_k, act_s_col_stride, act_s_col_warp_stride,
                                             slice_k_start, slice_k_finish, s_sh_rd, sh_g_idx, sh_s,
                                             group_metadata, frag_s, act_frag_s, sh_first_group_id,
                                             is_same_group, same_group_id, k, full_pipe);
  };

  auto fetch_zp_to_registers = [&](int k, int full_pipe) {
    marlin_fetch_zp_to_registers<Config>(zp_sh_rd, sh_zp, frag_qzp, frag_zpf, k, full_pipe);
  };

  auto dequant_data = [&](int q, scalar_32bit_t* frag_b_ptr, int zp = 0) {
    marlin_dequant_data<Config>(q, frag_b_ptr, zp);
  };

  // Execute the actual tensor core matmul of a sub-tile.
  bool is_first_matmul_in_slice = true;
  auto matmul = [&](int k, int pipe) {
    marlin_matmul<Config>(frag_a, frag_b_quant, frag_c, group_metadata, frag_s, act_frag_s,
                          frag_qzp, frag_zp, frag_zpf, dequant_data, is_first_matmul_in_slice, k,
                          pipe);
  };

  auto matmul_a8 = [&](int k) {
    marlin_matmul_a8<Config>(frag_a, frag_b_quant, frag_c, frag_c_tmp, frag_s, frag_qzp,
                             dequant_data, k);
  };

  // Since we slice across the k dimension of a tile in order to increase the
  // number of warps while keeping the n dimension of a tile reasonable, we have
  // multiple warps that accumulate their partial sums of the same output
  // location; which we have to reduce over in the end. We do in shared memory.
  auto thread_block_reduce = [&]() { marlin_thread_block_reduce<Config>(sh_red, frag_c); };

  // Since multiple threadblocks may process parts of the same column slice, we
  // finally have to globally reduce over the results. As the striped
  // partitioning minimizes the number of such reductions and our outputs are
  // usually rather small, we perform this reduction serially in L2 cache.
  auto global_reduce_fp16 = [&](bool first = false, bool last = false) {
    marlin_global_reduce_fp16<Config>(C, prob_n, slice_col, sh_block_sorted_ids,
                                      block_num_valid_tokens, sh_red, frag_c, first, last);
  };

  // Globally reduce over threadblocks that compute the same column block.
  // We use a tmp C buffer to reduce in full fp32 precision.
  auto global_reduce_fp32 = [&](bool first = false, bool last = false) {
    marlin_global_reduce_fp32<Config>(C_tmp, block_num_valid_tokens, locks_off, sh_red, frag_c,
                                      first, last);
  };

  // Write out the reduce final result in the correct layout. We only actually
  // reshuffle matrix fragments in this step, the reduction above is performed
  // in fragment layout.
  auto write_result = [&](bool last) {
    marlin_write_result<Config>(C, mul_topk_weights, prob_n, has_bias, use_atomic_add,
                                global_scale_f32, slice_col, slice_count, sh_block_sorted_ids,
                                sh_block_topk_weights, block_num_valid_tokens, sh_red, frag_c,
                                frag_s, frag_bias, last);
  };

  // Start global fetch and register load pipelines.
  auto start_pipes = [&]() {
    marlin_start_pipes<Config>(
        g_idx, prob_k, slice_iters, a_gl_rd_col, slice_k_start, slice_k_start_shared_fetch,
        zero_accums, fetch_act_order_scales_to_shared, fetch_to_shared, fetch_col_zp_to_shared,
        fetch_col_scale_to_shared, wait_for_stage, fetch_to_registers, init_same_group,
        fetch_scales_to_registers, fetch_zp_to_registers);
  };
  marlin_execute<Config>(
      b_bias_ptr, scales_ptr, g_idx, prob_k, locks, has_bias, use_atomic_add, use_fp32_reduce,
      in_part2, slice_row, slice_col_par, slice_col, slice_iters, slice_count, slice_idx,
      B_expert_off, sh_a_s, locks_off, init_slice, b_gl_stride, b_gl_rd_delta_o, s_gl_stride,
      act_s_col_tb_stride, zp_gl_stride, a_gl_rd_col, b_gl_rd, slice_k_start, slice_k_finish,
      slice_k_start_shared_fetch, slice_n_offset, s_gl_rd, s_sh_wr, s_sh_wr_pred, zp_gl_rd, s_sh_rd,
      bias_sh_rd, bias_sh_wr, bias_gl_rd, sh_bias, sh_s, frag_c, frag_s, frag_bias,
      sh_first_group_id, sh_num_groups, fetch_act_order_scales_to_shared, fetch_to_shared,
      wait_for_stage, fetch_to_registers, init_same_group, fetch_scales_to_registers,
      fetch_zp_to_registers, is_first_matmul_in_slice, matmul, matmul_a8, thread_block_reduce,
      global_reduce_fp16, global_reduce_fp32, write_result, start_pipes);
}
}  // namespace MARLIN_NAMESPACE_NAME
