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

template <typename Config>
__device__ __forceinline__ void marlin_dequant_data(int q,
                                                    typename Config::scalar_32bit_t* frag_b_ptr,
                                                    int zp) {
  using scalar_32bit_t = typename Config::scalar_32bit_t;
  static constexpr auto a_type = Config::a_type;
  static constexpr auto b_type = Config::b_type;
  static constexpr auto is_a_8bit = Config::is_a_8bit;
  static constexpr auto has_zp = Config::has_zp;
  static constexpr auto dequant_skip_flop = Config::dequant_skip_flop;
  static constexpr auto b_type_id = Config::b_type_id;

  if constexpr (a_type.size_bits() != b_type.size_bits()) {
    if constexpr (is_a_8bit && has_zp) {
      sub_zp_and_dequant<scalar_32bit_t, b_type_id, dequant_skip_flop>(q, frag_b_ptr, zp);
    } else {
      dequant<scalar_32bit_t, b_type_id, dequant_skip_flop>(q, frag_b_ptr);
    }
  }
}

template <typename Config, typename Ref_frag_a, typename Ref_frag_b_quant, typename Ref_frag_c,
          typename Ref_group_metadata, typename Ref_frag_s, typename Ref_act_frag_s,
          typename Ref_frag_qzp, typename Ref_frag_zp, typename Ref_frag_zpf,
          typename Ref_dequant_data, typename Ref_is_first_matmul_in_slice>
__device__ __forceinline__ void marlin_matmul(
    Ref_frag_a& frag_a, Ref_frag_b_quant& frag_b_quant, Ref_frag_c& frag_c,
    Ref_group_metadata& group_metadata, Ref_frag_s& frag_s, Ref_act_frag_s& act_frag_s,
    Ref_frag_qzp& frag_qzp, Ref_frag_zp& frag_zp, Ref_frag_zpf& frag_zpf,
    Ref_dequant_data& dequant_data, Ref_is_first_matmul_in_slice& is_first_matmul_in_slice, int k,
    int pipe) {
  using GroupDecoder = typename Config::GroupDecoder;
  using Adtype = typename Config::Adtype;
  using scalar_t = typename Config::scalar_t;
  using scalar_t2 = typename Config::scalar_t2;
  using scalar_32bit_t = typename Config::scalar_32bit_t;
  using c_scalar_t2 = typename Config::c_scalar_t2;
  using FragB = typename Config::FragB;
  using FragC = typename Config::FragC;
  static constexpr auto use_fp16_accum = Config::use_fp16_accum;
  static constexpr auto metadata_planes = Config::metadata_planes;
  static constexpr auto b_type = Config::b_type;
  static constexpr auto s_type = Config::s_type;
  static constexpr auto is_a_8bit = Config::is_a_8bit;
  static constexpr auto has_zp = Config::has_zp;
  static constexpr auto dequant_skip_flop = Config::dequant_skip_flop;
  static constexpr auto has_act_order = Config::has_act_order;
  static constexpr auto b_sh_wr_iters = Config::b_sh_wr_iters;
  static constexpr auto a_type_id = Config::a_type_id;
  static constexpr auto b_type_id = Config::b_type_id;
  static constexpr auto s_type_id = Config::s_type_id;
  static constexpr auto threads = Config::threads;
  static constexpr auto thread_m_blocks = Config::thread_m_blocks;
  static constexpr auto thread_n_blocks = Config::thread_n_blocks;
  static constexpr auto thread_k_blocks = Config::thread_k_blocks;
  static constexpr auto m_block_size_8 = Config::m_block_size_8;
  static constexpr auto group_blocks = Config::group_blocks;
  static constexpr auto is_zp_float = Config::is_zp_float;

  if (is_a_8bit) return;
  int k2 = k % 2;
  constexpr int g = group_blocks > 0 ? div_ceil(group_blocks, thread_k_blocks) : 1;
  const bool is_new_zp =
      (group_blocks == 0) ||
      ((group_blocks > 0) && (group_blocks < b_sh_wr_iters || k == 0)) && (pipe % g == 0) ||
      (group_blocks == -1 && is_first_matmul_in_slice);
  if constexpr (has_zp && !is_zp_float) {
    if (is_new_zp) {
      if constexpr (group_blocks == -1) is_first_matmul_in_slice = false;
      int zp_quant_0, zp_quant_1;

      if constexpr (b_type.size_bits() == 4) {
        zp_quant_0 = frag_qzp[k2][0];
        zp_quant_1 = zp_quant_0 >> 8;
      } else {
        static_assert(b_type.size_bits() == 8);
        zp_quant_0 = frag_qzp[k2][0];
        zp_quant_1 = frag_qzp[k2][1];
      }

      dequant_data(zp_quant_0, reinterpret_cast<scalar_32bit_t*>(&frag_zp));
      dequant_data(zp_quant_1, reinterpret_cast<scalar_32bit_t*>(&frag_zp) + 2);
    }
  }
  if constexpr (!dequant_skip_flop && has_zp && is_zp_float) {
    if (is_new_zp) {
      reinterpret_cast<int4*>(&frag_zp)[0] = reinterpret_cast<int4*>(&frag_zpf[k2])[0];
    }
  }

  if constexpr (s_type == vllm::kFE4M3fn || s_type == vllm::kFE8M0fnu) {
    int s_quant_0 = reinterpret_cast<int*>(frag_s[k2])[0];
    int s_quant_1 = reinterpret_cast<int*>(frag_s[k2])[1];

    dequant_fp8_scales<c_scalar_t2, s_type_id>(s_quant_0,
                                               reinterpret_cast<c_scalar_t2*>(&frag_s[k2]));
    dequant_fp8_scales<c_scalar_t2, s_type_id>(s_quant_1,
                                               reinterpret_cast<c_scalar_t2*>(&frag_s[k2]) + 2);
  }

// We have the m dimension as the inner loop in order to encourage overlapping
// dequantization and matmul operations.
#pragma unroll
  for (int j = 0; j < 4; j++) {
    FragB frag_b0;
    FragB frag_b1;
    int b_quant_0, b_quant_1;

    if constexpr (b_type_id == vllm::kFE2M1f.id()) {
      b_quant_1 = frag_b_quant[Config::compact_registers ? 0 : k2][0][j];
      b_quant_0 = b_quant_1 << 8;
    } else if constexpr (b_type.size_bits() == 4) {
      b_quant_0 = frag_b_quant[Config::compact_registers ? 0 : k2][0][j];
      b_quant_1 = b_quant_0 >> 8;
    } else {
      static_assert(b_type.size_bits() == 8);
      int* frag_b_quant_ptr = reinterpret_cast<int*>(frag_b_quant[Config::compact_registers ? 0 : k2]);
      b_quant_0 = frag_b_quant_ptr[j * 2 + 0];
      b_quant_1 = frag_b_quant_ptr[j * 2 + 1];
    }

#ifdef CUBIC_MARLIN_FP16_GROUP_FRAGMENT
    // Format-owned decoder closes the original F16 weight boundary before
    // converting to the MMA operand type. All other formats retain their ABI.
    static_assert(metadata_planes > 0 && !dequant_skip_flop && !has_zp && !is_a_8bit);
    dequant<half2, b_type_id, false>(b_quant_0, reinterpret_cast<half2*>(&frag_b0));
    dequant<half2, b_type_id, false>(b_quant_1, reinterpret_cast<half2*>(&frag_b1));
#else
    dequant_data(b_quant_0, reinterpret_cast<scalar_32bit_t*>(&frag_b0));
    dequant_data(b_quant_1, reinterpret_cast<scalar_32bit_t*>(&frag_b1));
#endif

    if constexpr (dequant_skip_flop && has_zp && !is_zp_float && !is_a_8bit) {
      sub_zp<a_type_id>(frag_b0, frag_zp[j], 0);
      sub_zp<a_type_id>(frag_b1, frag_zp[j], 1);
    }

    // Apply the format-owned rule or the ordinary scale contract.
    if constexpr (metadata_planes) {
      GroupDecoder::template apply<a_type_id>(frag_b0, group_metadata[(Config::compact_registers && Config::group_blocks >= Config::thread_k_blocks) ? 0 : k2], 2 * j);
      GroupDecoder::template apply<a_type_id>(frag_b1, group_metadata[(Config::compact_registers && Config::group_blocks >= Config::thread_k_blocks) ? 0 : k2], 2 * j + 1);
    } else if constexpr (has_act_order && !is_a_8bit) {
      static_assert(group_blocks != -1);
      scale4<a_type_id>(frag_b0, act_frag_s[k2][0][j], act_frag_s[k2][1][j], act_frag_s[k2][2][j],
                        act_frag_s[k2][3][j], 0);
      scale4<a_type_id>(frag_b1, act_frag_s[k2][0][j], act_frag_s[k2][1][j], act_frag_s[k2][2][j],
                        act_frag_s[k2][3][j], 1);
    } else if constexpr (!dequant_skip_flop && has_zp && !is_zp_float && group_blocks == -1 &&
                         !is_a_8bit) {
      int idx = (threadIdx.x / 4) % 2;
      scalar_t2 s2 =
          Adtype::nums2num2(reinterpret_cast<scalar_t*>(&frag_s[j / 2][j % 2 * 2 + 0])[idx],
                            reinterpret_cast<scalar_t*>(&frag_s[j / 2][j % 2 * 2 + 1])[idx]);
      if (is_new_zp) frag_zp[j] = __hmul2(frag_zp[j], s2);
      scale_and_sub<a_type_id>(frag_b0, s2.x, frag_zp[j].x);
      scale_and_sub<a_type_id>(frag_b1, s2.y, frag_zp[j].y);
    } else if constexpr (!dequant_skip_flop && has_zp && group_blocks != -1 && !is_a_8bit) {
      if (is_new_zp)
        frag_zp[j] = __hmul2(frag_zp[j], *reinterpret_cast<scalar_t2*>(&frag_s[k2][j]));
      scale_and_sub<a_type_id>(frag_b0, frag_s[k2][j][0].x, frag_zp[j].x);
      scale_and_sub<a_type_id>(frag_b1, frag_s[k2][j][0].y, frag_zp[j].y);
    } else if constexpr (group_blocks != -1 && !is_a_8bit) {
      scale<a_type_id>(frag_b0, frag_s[k2][j], 0);
      scale<a_type_id>(frag_b1, frag_s[k2][j], 1);
    }

#pragma unroll
    for (int i = 0; i < thread_m_blocks; i++) {
      if constexpr (m_block_size_8) {
        mma_trans<a_type_id, use_fp16_accum>(frag_a[Config::compact_registers ? 0 : k2][i], frag_b0, frag_b1, frag_c[i][j][0]);
      } else {
#if CUBIC_MARLIN_ORDERED_K16
        // One exact group partial followed by the prescribed FP32 RN add.
        // N-only warp placement preserves global K16 order for every row.
        static_assert(group_blocks == 1 && !m_block_size_8 && threads / 32 == thread_n_blocks / 4);
        FragC partial0{}, partial1{};
        mma<a_type_id, false>(frag_a[Config::compact_registers ? 0 : k2][i], frag_b0, partial0);
        mma<a_type_id, false>(frag_a[Config::compact_registers ? 0 : k2][i], frag_b1, partial1);
#pragma unroll
        for (int z = 0; z < 4; ++z) {
          frag_c[i][j][0][z] = __fadd_rn(frag_c[i][j][0][z], partial0[z]);
          frag_c[i][j][1][z] = __fadd_rn(frag_c[i][j][1][z], partial1[z]);
        }
#else
        mma<a_type_id, use_fp16_accum>(frag_a[Config::compact_registers ? 0 : k2][i], frag_b0, frag_c[i][j][0]);
        mma<a_type_id, use_fp16_accum>(frag_a[Config::compact_registers ? 0 : k2][i], frag_b1, frag_c[i][j][1]);
#endif
      }
    }
  }
}

template <typename Config, typename Ref_frag_a, typename Ref_frag_b_quant, typename Ref_frag_c,
          typename Ref_frag_c_tmp, typename Ref_frag_s, typename Ref_frag_qzp,
          typename Ref_dequant_data>
__device__ __forceinline__ void marlin_matmul_a8(Ref_frag_a& frag_a, Ref_frag_b_quant& frag_b_quant,
                                                 Ref_frag_c& frag_c, Ref_frag_c_tmp& frag_c_tmp,
                                                 Ref_frag_s& frag_s, Ref_frag_qzp& frag_qzp,
                                                 Ref_dequant_data& dequant_data, int k) {
  using Cdtype = typename Config::Cdtype;
  using scalar_32bit_t = typename Config::scalar_32bit_t;
  using FragB = typename Config::FragB;
  static constexpr auto a_type = Config::a_type;
  static constexpr auto b_type = Config::b_type;
  static constexpr auto s_type = Config::s_type;
  static constexpr auto is_a_8bit = Config::is_a_8bit;
  static constexpr auto has_zp = Config::has_zp;
  static constexpr auto a_type_id = Config::a_type_id;
  static constexpr auto s_type_id = Config::s_type_id;
  static constexpr auto thread_m_blocks = Config::thread_m_blocks;
  static constexpr auto group_blocks = Config::group_blocks;

  int k2 = k % 2;
#pragma unroll
  for (int j = 0; j < 2; j++) {
    FragB frag_b[2];

    if (is_a_8bit && b_type.size_bits() == 4 && !has_zp) {
      dequant_data(frag_b_quant[Config::compact_registers ? 0 : k2][0][j * 2], reinterpret_cast<scalar_32bit_t*>(&frag_b));
      dequant_data(frag_b_quant[Config::compact_registers ? 0 : k2][0][j * 2 + 1], reinterpret_cast<scalar_32bit_t*>(&frag_b) + 2);
    } else if (is_a_8bit && b_type.size_bits() == 4 && has_zp) {
      int off = (threadIdx.x / 32) % 2 * 2 + j;
      int zp = (frag_qzp[k2][0] >> (off * 8)) & 0xF;
      dequant_data(frag_b_quant[Config::compact_registers ? 0 : k2][0][j * 2], reinterpret_cast<scalar_32bit_t*>(&frag_b), zp);
      zp = (frag_qzp[k2][0] >> (off * 8 + 4)) & 0xF;
      dequant_data(frag_b_quant[Config::compact_registers ? 0 : k2][0][j * 2 + 1], reinterpret_cast<scalar_32bit_t*>(&frag_b) + 2,
                   zp);
    } else {
      reinterpret_cast<int2*>(&frag_b)[0] = reinterpret_cast<int2*>(&frag_b_quant[Config::compact_registers ? 0 : k2][j])[0];
      reinterpret_cast<int2*>(&frag_b)[1] = reinterpret_cast<int2*>(&frag_b_quant[Config::compact_registers ? 0 : k2][j])[1];
#ifdef CUBIC_MARLIN_INT_A8_REGISTER_ZERO
      if constexpr (is_a_8bit && b_type == vllm::kU8) {
        const int off = ((threadIdx.x / 32) % 2) * 2 + j;
        const unsigned int z0 = 0x01010101u * ((frag_qzp[k2][0] >> (off * 8)) & 0xff);
        const unsigned int z1 = 0x01010101u * ((frag_qzp[k2][1] >> (off * 8)) & 0xff);
        auto words = reinterpret_cast<unsigned int*>(&frag_b);
#pragma unroll
        for (int i = 0; i < 2; ++i) words[i] = __vsub4(words[i], z0);
#pragma unroll
        for (int i = 2; i < 4; ++i) words[i] = __vsub4(words[i], z1);
      }
#endif
    }

#pragma unroll
    for (int i = 0; i < thread_m_blocks; i++) {
      mma<a_type_id, false, 32>(frag_a[Config::compact_registers ? 0 : k2][i], frag_b[0],
                                (group_blocks == -1 ? frag_c : frag_c_tmp)[i][j][0]);
      mma<a_type_id, false, 32>(frag_a[Config::compact_registers ? 0 : k2][i], frag_b[1],
                                (group_blocks == -1 ? frag_c : frag_c_tmp)[i][j][1]);
    }

    if constexpr (group_blocks != -1) {
      if (group_blocks == 2 || k == Config::b_sh_wr_iters - 1) {
        if constexpr (a_type == vllm::kS8) {
#ifdef CUBIC_MARLIN_INT_A8_FLOAT_SCALE
          // INT checkpoints store a floating scale per K group. Preserve the
          // integer MMA result, then apply that original scale in FP32 before
          // the existing activation-scale and output-rounding boundaries.
          float2 s_vals[2];
          s_vals[0] = __half22float2(*reinterpret_cast<const half2*>(&frag_s[k2][j * 2][0]));
          s_vals[1] = __half22float2(*reinterpret_cast<const half2*>(&frag_s[k2][j * 2 + 1][0]));
#pragma unroll
          for (int i = 0; i < thread_m_blocks; i++) {
#pragma unroll
            for (int b = 0; b < 2; b++) {
#pragma unroll
              for (int g = 0; g < 4; g++) {
                const float scale = reinterpret_cast<float*>(&s_vals[b])[g % 2];
                const int32_t dot = *reinterpret_cast<int32_t*>(&frag_c_tmp[i][j][b][g]);
                frag_c[i][j][b][g] = __fadd_rn(
                    frag_c[i][j][b][g], __fmul_rn(__int2float_rn(dot), scale));
                frag_c_tmp[i][j][b][g] = 0.0f;
              }
            }
          }
#else
          int2 s_vals[2];
          s_vals[0] = {(int)reinterpret_cast<uint16_t*>(&frag_s[k2][j * 2][0])[0],
                       (int)reinterpret_cast<uint16_t*>(&frag_s[k2][j * 2][0])[1]};
          s_vals[1] = {(int)reinterpret_cast<uint16_t*>(&frag_s[k2][j * 2 + 1][0])[0],
                       (int)reinterpret_cast<uint16_t*>(&frag_s[k2][j * 2 + 1][0])[1]};

#pragma unroll
          for (int i = 0; i < thread_m_blocks; i++) {
#pragma unroll
            for (int g = 0; g < 4; g++) {
              int scale = reinterpret_cast<int*>(&s_vals[0])[g % 2];
              *reinterpret_cast<int32_t*>(&frag_c[i][j][0][g]) +=
                  *reinterpret_cast<int32_t*>(&frag_c_tmp[i][j][0][g]) * scale;
              frag_c_tmp[i][j][0][g] = 0.0f;
            }

#pragma unroll
            for (int g = 0; g < 4; g++) {
              int scale = reinterpret_cast<int*>(&s_vals[1])[g % 2];
              *reinterpret_cast<int32_t*>(&frag_c[i][j][1][g]) +=
                  *reinterpret_cast<int32_t*>(&frag_c_tmp[i][j][1][g]) * scale;
              frag_c_tmp[i][j][1][g] = 0.0f;
            }
          }
#endif
        } else {
          float2 s_vals[2];
          if constexpr (s_type_id != vllm::kFE8M0fnu.id()) {
            static_assert(a_type.size_bits() == 16 || s_type.size_bits() == 16);
            s_vals[0] = Cdtype::num22float2(frag_s[k2][j * 2][0]);
            s_vals[1] = Cdtype::num22float2(frag_s[k2][j * 2 + 1][0]);
          } else {
            int32_t* s_vals_int = reinterpret_cast<int32_t*>(&s_vals[0]);
            int32_t s_vals_e8m0 = *reinterpret_cast<int32_t*>(&frag_s[k2][j][0]);

            s_vals_int[0] = (s_vals_e8m0 & 0xFF) << 23;
            s_vals_int[1] = (s_vals_e8m0 & 0xFF00) << 15;
            s_vals_int[2] = (s_vals_e8m0 & 0xFF0000) << 7;
            s_vals_int[3] = (s_vals_e8m0 & 0xFF000000) >> 1;
          }

#pragma unroll
          for (int i = 0; i < thread_m_blocks; i++) {
#pragma unroll
            for (int g = 0; g < 4; g++) {
              float scale = reinterpret_cast<float*>(&s_vals[0])[g % 2];
              frag_c[i][j][0][g] += frag_c_tmp[i][j][0][g] * scale;
              frag_c_tmp[i][j][0][g] = 0.0f;
            }

#pragma unroll
            for (int g = 0; g < 4; g++) {
              float scale = reinterpret_cast<float*>(&s_vals[1])[g % 2];
              frag_c[i][j][1][g] += frag_c_tmp[i][j][1][g] * scale;
              frag_c_tmp[i][j][1][g] = 0.0f;
            }
          }
        }
      }
    }
  }
}

}  // namespace MARLIN_NAMESPACE_NAME
