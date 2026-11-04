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

#include "configuration.cuh"
namespace MARLIN_NAMESPACE_NAME {

template <typename scalar_t,                   // compute dtype, half or nv_float16
          const vllm::ScalarTypeId b_type_id,  // weight MarlinScalarType id
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
          const bool has_act_order,            // whether act_order is enabled
          const int group_blocks,              // number of consecutive 16x16 blocks
                                               // with a separate quantization scale
          const bool is_zp_float               // is zero point of float16 type?
          >
__global__ void Marlin(const int4* __restrict__ A,  // fp16 input matrix of shape mxk
                       const int4* __restrict__ B,  // 4bit quantized weight matrix of shape kxn
                       int4* __restrict__ C,        // fp16 output buffer of shape mxn
                       int4* __restrict__ C_tmp,    // fp32 tmp output buffer (for reduce)
                       const int4* __restrict__ scales_ptr,  // fp16 quantization scales of shape
                                                             // (k/groupsize)xn
                       const int4* __restrict__ zp_ptr,      // 4bit packed zero-points of shape
                                                             // (k/groupsize)x(n/pack_factor)
                       const int* __restrict__ g_idx,        // int32 group indices of shape k
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
                       bool use_atomic_add,    // whether to use atomic add to reduce
                       bool use_fp32_reduce    // whether to use fp32 global reduce
) {}

}  // namespace MARLIN_NAMESPACE_NAME
