#pragma once
// Cubic compile-time arithmetic contracts; included in the source receipt.
#ifndef CUBIC_MARLIN_FP32_OUTPUT
#define CUBIC_MARLIN_FP32_OUTPUT 0
#endif
#ifndef CUBIC_MARLIN_ORDERED_K16
#define CUBIC_MARLIN_ORDERED_K16 0
#endif
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

#ifndef MARLIN_NAMESPACE_NAME
#define MARLIN_NAMESPACE_NAME marlin_moe_wna16
#endif

#ifndef CUBIC_MARLIN_DENSE_LINEAR
#define CUBIC_MARLIN_DENSE_LINEAR 0
#endif

// This is a binary arithmetic-mode tag. The actual contiguous K-slice count
// is a sealed launch operand, independent of logical M and compiled tile code.
#ifndef CUBIC_MARLIN_FIXED_PARTITIONS
#define CUBIC_MARLIN_FIXED_PARTITIONS 0
#endif

#include "core/scalar_type.hpp"
#include "libtorch_stable/quantization/marlin/dequant.h"
#include "libtorch_stable/quantization/marlin/marlin.cuh"
#include "libtorch_stable/quantization/marlin/marlin_dtypes.cuh"
#include "libtorch_stable/quantization/marlin/marlin_mma.h"

#define STATIC_ASSERT_SCALAR_TYPE_VALID(scalar_t)                                                  \
  static_assert(std::is_same<scalar_t, half>::value || std::is_same<scalar_t, nv_bfloat16>::value, \
                "only float16 and bfloat16 is supported");

namespace MARLIN_NAMESPACE_NAME {
// Optional format-owned FP32 group metadata. The default keeps the vendor
// scale ABI; a decoder owns arithmetic, while Marlin owns staging/lifetimes.
#ifndef CUBIC_MARLIN_GROUP_DECODER
struct MarlinDefaultGroupDecoder {
  static constexpr int metadata_planes = 0;
  template <vllm::ScalarTypeId type, typename... Args>
  __device__ static void apply(Args&&...) {}
};
#define CUBIC_MARLIN_GROUP_DECODER MarlinDefaultGroupDecoder
#endif

template <vllm::ScalarTypeId P_a_type_id, vllm::ScalarTypeId P_b_type_id,
          vllm::ScalarTypeId P_c_type_id, vllm::ScalarTypeId P_s_type_id, int P_threads,
          int P_thread_m_blocks, int P_thread_n_blocks, int P_thread_k_blocks,
          bool P_m_block_size_8, int P_stages, int P_group_blocks, bool P_is_zp_float>
struct MarlinConfiguration {
  // Dense projections have contiguous rows and one weight, with no routing,
  // expert weighting, bias or atomic accumulation in their binding contract.
  static constexpr bool dense_linear = CUBIC_MARLIN_DENSE_LINEAR;
  static constexpr int fixed_partitions = CUBIC_MARLIN_FIXED_PARTITIONS;
  // Fixed partials retain physical tile order for both contiguous and routed
  // rows. The final ordered fold owns the routed row map and padding mask.
  static constexpr auto a_type_id = P_a_type_id;
  static constexpr auto b_type_id = P_b_type_id;
  static constexpr auto c_type_id = P_c_type_id;
  static constexpr auto s_type_id = P_s_type_id;
  static constexpr auto threads = P_threads;
  static constexpr auto thread_m_blocks = P_thread_m_blocks;
  static constexpr auto thread_n_blocks = P_thread_n_blocks;
  static constexpr auto thread_k_blocks = P_thread_k_blocks;
  static constexpr auto m_block_size_8 = P_m_block_size_8;
  static constexpr auto stages = P_stages;
  static constexpr auto group_blocks = P_group_blocks;
  static constexpr auto is_zp_float = P_is_zp_float;
  using GroupDecoder = CUBIC_MARLIN_GROUP_DECODER;
  using Adtype = MarlinScalarType<a_type_id>;
  using Cdtype = MarlinScalarType<c_type_id>;
  using scalar_t = typename MarlinScalarType<a_type_id>::scalar_t;
  using scalar_t2 = typename MarlinScalarType<a_type_id>::scalar_t2;
  using scalar_32bit_t = typename MarlinScalarType<a_type_id>::scalar_32bit_t;
  using c_scalar_t = typename MarlinScalarType<c_type_id>::scalar_t;
  using c_scalar_t2 = typename MarlinScalarType<c_type_id>::scalar_t2;
  using FragA = typename MarlinScalarType<a_type_id>::FragA;
  using FragB = typename MarlinScalarType<a_type_id>::FragB;
  using FragC = typename MarlinScalarType<a_type_id>::FragC;
  using FragS = typename MarlinScalarType<c_type_id>::FragS;
  using FragZP = typename MarlinScalarType<c_type_id>::FragZP;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 750
  static constexpr auto num_bits = vllm::ScalarType::from_id(b_type_id).size_bits();
  // Disable use_fp16_accum for NVFP4 and cases when group_size == -1 &&
  // num_bits == 4
  static constexpr bool use_fp16_accum =
      a_type_id == vllm::kFloat16.id() &&
      (!(b_type_id == vllm::kFE2M1f.id() && s_type_id == vllm::kFE4M3fn.id()) &&
       !(group_blocks == -1 && num_bits == 4));
#else
  static constexpr bool use_fp16_accum = false;
#endif
  static constexpr int moe_block_size = m_block_size_8 ? 8 : (16 * thread_m_blocks);
  static constexpr int metadata_planes = GroupDecoder::metadata_planes;
  static constexpr int metadata_scale = metadata_planes ? 2 * metadata_planes : 1;
  static constexpr auto a_type = vllm::ScalarType::from_id(a_type_id);
  static constexpr auto b_type = vllm::ScalarType::from_id(b_type_id);
  static constexpr auto c_type = vllm::ScalarType::from_id(c_type_id);
  static constexpr auto s_type = vllm::ScalarType::from_id(s_type_id);
  static constexpr bool is_a_8bit = a_type.size_bits() == 8;
  static constexpr bool has_zp = b_type == vllm::kU4 || b_type == vllm::kU8;
  static constexpr bool is_int_type = b_type == vllm::kU4 || b_type == vllm::kU8 ||
                                      b_type == vllm::kS4 || b_type == vllm::kS8 ||
                                      b_type == vllm::kU4B8 || b_type == vllm::kU8B128;
  static constexpr bool is_8bit_scale = s_type.size_bits() == 8;
  static constexpr bool dequant_skip_flop =
      is_a_8bit || (b_type == vllm::kFE4M3fn && !(s_type == vllm::kFE8M0fnu)) ||
      b_type == vllm::kFE2M1f && s_type == vllm::kFE4M3fn ||
      has_zp && !is_zp_float && !std::is_same<scalar_t, nv_bfloat16>::value ||
      has_zp && !is_zp_float && !(b_type == vllm::kU8);
  static constexpr bool has_act_order = group_blocks == 0;
#ifdef CUBIC_MARLIN_STORAGE_BITS
  static constexpr int storage_bits = CUBIC_MARLIN_STORAGE_BITS;
#else
  static constexpr int storage_bits = b_type.size_bits();
#endif
  static constexpr int pack_factor = 32 / b_type.size_bits();
  static constexpr int a_sh_stride = 16 * thread_k_blocks / (is_a_8bit ? 16 : 8);
  static constexpr int a_gl_rd_delta_o = 16 * thread_k_blocks / (is_a_8bit ? 16 : 8);
  static constexpr int a_sh_wr_delta = a_sh_stride * (threads / a_gl_rd_delta_o);
  static constexpr int a_sh_rd_delta_i = a_sh_stride * 16;
  static constexpr int a_sh_stage = a_sh_stride * (16 * thread_m_blocks);
  static constexpr int a_sh_wr_iters = div_ceil(a_sh_stage, a_sh_wr_delta);
  static constexpr int b_sh_stride =
      ((thread_n_blocks * 16) * 16 / pack_factor) / (is_a_8bit ? 2 : 4);
  static constexpr int b_thread_vecs = b_type.size_bits() == 4 ? 1 : 2;
  static constexpr int b_sh_stride_threads = b_sh_stride / b_thread_vecs;
  static constexpr int b_sh_wr_delta = threads * b_thread_vecs;
  static constexpr int b_sh_stage = b_sh_stride * thread_k_blocks / (is_a_8bit ? 2 : 1);
  static constexpr int b_sh_wr_iters = b_sh_stage / b_sh_wr_delta;
  static constexpr bool compact_registers = fixed_partitions > 0 || b_sh_wr_iters == 1;
  static constexpr int operand_buffers = compact_registers ? 1 : 2;
  static constexpr int s_sh_stride = 16 * thread_n_blocks / (is_8bit_scale ? 16 : 8);
  static constexpr int s_tb_groups =
      !has_act_order && group_blocks != -1 && group_blocks < thread_k_blocks
          ? thread_k_blocks / group_blocks
          : 1;
  static constexpr int s_sh_stage = s_tb_groups * s_sh_stride;
  static constexpr int tb_k = 16 * thread_k_blocks;
  static constexpr int g_idx_stage = has_act_order ? (tb_k * sizeof(int)) / 16 : 0;
  static constexpr int act_s_max_num_groups = 32;
  static constexpr int tb_n_warps = thread_n_blocks / (is_a_8bit ? 2 : 4);
  static constexpr int zp_sh_stride =
      is_zp_float ? 16 * thread_n_blocks / 8 : ((16 * thread_n_blocks) / pack_factor) / 4;
  static constexpr int zp_tb_groups = s_tb_groups;
  static constexpr int zp_sh_stage = has_zp ? zp_tb_groups * zp_sh_stride : 0;
  static constexpr int num_col_threads = 8;
  static constexpr int num_row_threads = 4;
  static constexpr int num_ints_per_thread = 8 / pack_factor;
  static constexpr int sh_red_size = (2 * thread_n_blocks + 1) * 16 * thread_m_blocks;
  static constexpr int b_storage_stage = b_sh_stage * storage_bits / b_type.size_bits();
  static constexpr int sh_b_size = stages * b_storage_stage;
  static constexpr int sh_size_b_red_min = (sh_red_size < sh_b_size ? sh_red_size : sh_b_size);
  static constexpr int sh_size_b_red_max = (sh_red_size > sh_b_size ? sh_red_size : sh_b_size);
  static constexpr int sh_bias_size = (thread_n_blocks * 16 / 8);
  static constexpr int sh_b_red_bias_size = sh_size_b_red_max > (sh_size_b_red_min + sh_bias_size)
                                                ? sh_size_b_red_max
                                                : (sh_size_b_red_min + sh_bias_size);
  static constexpr int sh_s_size =
      has_act_order ? (act_s_max_num_groups * s_sh_stride) : (stages * s_sh_stage * metadata_scale);
};
}  // namespace MARLIN_NAMESPACE_NAME
