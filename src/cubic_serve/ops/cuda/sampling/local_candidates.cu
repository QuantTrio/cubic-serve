#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <limits.h>
#include <math.h>

#if SCORE_TYPE == 0
using score_t = __nv_bfloat16;
#elif SCORE_TYPE == 1
using score_t = __half;
#else
using score_t = float;
#endif

// The unchanged torch.topk owns candidate selection, including cutoff ties.
// This boundary only validates owned logits and publishes its selected pairs.
#if PHASE == 0
extern "C" __global__ void candidate_partial(
    const score_t* logits, const float* temperatures, const bool* activity, const int* metadata,
    float* partial_scores, int* partial_ids, int* partial_finite,
    int width, int stride, int tiles, int all_random
#if HAS_GRAMMAR
    , const uint32_t* token_mask, score_t* masked_logits, int mask_words, int64_t vocab_start,
    int masked_stride
#endif
    ) {
    const int row = blockIdx.x, tile = blockIdx.y;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const bool owned = (!metadata || row < metadata[0]) && (!activity || activity[row]);
    const bool greedy = !all_random && temperatures[row] == 0.f;
    float best = -INFINITY;
    int best_id = INT_MAX, good = 1;
    if (owned) {
        const int end = min(width, (tile + 1) * 4096);
        for (int j = tile * 4096 + threadIdx.x; j < end; j += blockDim.x) {
            const float value = float(logits[(size_t)row * stride + j]);
            good &= isfinite(value);
#if HAS_GRAMMAR
            const int64_t token = vocab_start + j;
            const bool allowed = (token_mask[(size_t)row * mask_words + token / 32]
                                  >> (token % 32)) & 1u;
            masked_logits[(size_t)row * masked_stride + j] = score_t(allowed ? value : -INFINITY);
            if (!allowed) continue;
#endif
            if (greedy && (value > best || (value == best && j < best_id))) {
                best = value; best_id = j;
            }
        }
    }
#if HAS_GRAMMAR && !CONSUME_LOGITS
    else {
        const int end = min(width, (tile + 1) * 4096);
        for (int j = tile * 4096 + threadIdx.x; j < end; j += blockDim.x)
            masked_logits[(size_t)row * masked_stride + j] = score_t(-INFINITY);
    }
#endif
    for (int d = 16; d; d >>= 1) {
        const float value = __shfl_down_sync(0xffffffff, best, d);
        const int id = __shfl_down_sync(0xffffffff, best_id, d);
        good &= __shfl_down_sync(0xffffffff, good, d);
        if (value > best || (value == best && id < best_id)) {
            best = value; best_id = id;
        }
    }
    __shared__ float scores[8];
    __shared__ int ids[8], finite[8];
    if (!lane) { scores[warp] = best; ids[warp] = best_id; finite[warp] = good; }
    __syncthreads();
    if (!warp) {
        best = lane < 8 ? scores[lane] : -INFINITY;
        best_id = lane < 8 ? ids[lane] : INT_MAX;
        good = lane < 8 ? finite[lane] : 1;
        for (int d = 16; d; d >>= 1) {
            const float value = __shfl_down_sync(0xffffffff, best, d);
            const int id = __shfl_down_sync(0xffffffff, best_id, d);
            good &= __shfl_down_sync(0xffffffff, good, d);
            if (value > best || (value == best && id < best_id)) {
                best = value; best_id = id;
            }
        }
        if (!lane) {
            const size_t out = (size_t)row * tiles + tile;
            partial_scores[out] = best;
            partial_ids[out] = best_id;
            partial_finite[out] = good;
        }
    }
}
#else
extern "C" __global__ void candidate_publish(
    const score_t* top_scores, const int64_t* top_ids,
    const float* temperatures, const bool* activity, const int* metadata,
    const float* partial_scores, const int* partial_ids, const int* partial_finite,
    score_t* scores, int64_t* ids,
    int tiles, int count, int capacity, int64_t vocab_start, int all_random) {
    const int row = blockIdx.x, lane = threadIdx.x & 31;
    const bool owned = (!metadata || row < metadata[0]) && (!activity || activity[row]);
    const bool greedy = temperatures[row] == 0.f;
    float best = -INFINITY;
    int best_id = INT_MAX, good = 1;
    for (int tile = lane; tile < tiles; tile += 32) {
        const size_t offset = (size_t)row * tiles + tile;
        good &= partial_finite[offset];
        const float value = partial_scores[offset];
        const int id = partial_ids[offset];
        if (value > best || (value == best && id < best_id)) {
            best = value; best_id = id;
        }
    }
    for (int d = 16; d; d >>= 1) {
        const float value = __shfl_down_sync(0xffffffff, best, d);
        const int id = __shfl_down_sync(0xffffffff, best_id, d);
        good &= __shfl_down_sync(0xffffffff, good, d);
        if (value > best || (value == best && id < best_id)) {
            best = value; best_id = id;
        }
    }
    good = __shfl_sync(0xffffffff, good, 0);
    best = __shfl_sync(0xffffffff, best, 0);
    best_id = __shfl_sync(0xffffffff, best_id, 0);
    for (int j = lane; j < capacity; j += 32) {
        float value = NAN;
        int64_t id = -1;
        if (j < count && owned) {
            if (!good) id = -2;
            else if (!greedy) {
                value = float(top_scores[(size_t)row * count + j]);
                id = top_ids[(size_t)row * count + j] + vocab_start;
#if HAS_GRAMMAR
                if (!isfinite(value)) { value = NAN; id = -1; }
#endif
            } else if (!all_random && j == 0) {
#if HAS_GRAMMAR
                if (best_id != INT_MAX) {
#endif
                value = best; id = (int64_t)best_id + vocab_start;
#if HAS_GRAMMAR
                }
#endif
            }
        }
        scores[(size_t)row * capacity + j] = score_t(value);
        ids[(size_t)row * capacity + j] = id;
    }
}
#endif
