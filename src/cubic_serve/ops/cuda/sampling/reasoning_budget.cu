#include <cuda_runtime.h>
#include <stdint.h>

// Integer-only reasoning ledger. Geometry and token IDs are runtime operands;
// no model, vocabulary, speculative depth, or GPU generation is compiled in.
// State: phase (-1 disabled, -2 invalid, 0 waiting, 1 optional start, 2 thought),
// used tokens, quota, tail length, EOS enabled, then the pending marker tail.
__device__ bool equal_marker(const int64_t* s, const int32_t* markers,
                             const int32_t* offsets, int marker, int n) {
    if (n > offsets[marker + 1] - offsets[marker]) return false;
    for (int j = 0; j < n; ++j)
        if (s[5 + j] != markers[offsets[marker] + j]) return false;
    return true;
}

__device__ int64_t forced_token(const int64_t* s, const int32_t* markers,
                               const int32_t* offsets) {
    if (s[0] <= 0 || s[0] == 3 || s[1] < s[2]) return -1;
    const int n = int(s[3]), end_len = offsets[2] - offsets[1];
    const int prefix = n && n < end_len && equal_marker(s, markers, offsets, 1, n) ? n : 0;
    return markers[offsets[1] + prefix];
}

__device__ void thought_token(int64_t* s, int64_t token, const int32_t* markers,
                              const int32_t* offsets, int marker_count, int tail_capacity) {
    ++s[1];
    int n = int(s[3]);
    // A valid pending tail is always a proper prefix before a new token.
    if (n >= tail_capacity) { s[0] = -2; return; }
    s[5 + n++] = token;
    while (n) {
        bool prefix = false;
        for (int m = 1; m < marker_count; ++m)
            prefix |= equal_marker(s, markers, offsets, m, n);
        if (prefix) break;
        for (int j = 1; j < n; ++j) s[4 + j] = s[5 + j];
        --n;
    }
    s[3] = n;
    for (int m = 1; n && m < marker_count; ++m) {
        if (n == offsets[m + 1] - offsets[m] && equal_marker(s, markers, offsets, m, n)) {
            s[0] = 0; s[1] = 0; s[3] = 0;
            break;
        }
    }
}

__device__ bool accept_token(int64_t* s, int64_t token, const int32_t* markers,
                             const int32_t* offsets, int marker_count, int tail_capacity) {
    if (s[0] == -1) return true;
    if (s[0] < -1) return false;
    bool eos = false;
    for (int j = offsets[marker_count]; j < offsets[marker_count + 1]; ++j)
        eos |= token == markers[j];
    if (s[0] == 3) {
        if (!eos) s[0] = -2;
        return eos;
    }
    const int64_t forced = forced_token(s, markers, offsets);
    if (forced >= 0 && token != forced) { s[0] = -2; return false; }
    const int start_len = offsets[1] - offsets[0];
    if (s[0] == 0) {
        int n = int(s[3]);
        if (n == start_len) {
            for (int j = 1; j < n; ++j) s[4 + j] = s[5 + j];
            --n;
        }
        s[5 + n++] = token; s[3] = n;
        if (n == start_len && equal_marker(s, markers, offsets, 0, n)) {
            s[0] = 2; s[1] = 0; s[3] = 0;
        }
    } else if (s[0] == 1) {
        const int n = int(s[3]);
        if (n < start_len && token == markers[offsets[0] + n]) {
            s[5 + n] = token; s[3] = n + 1;
            if (n + 1 == start_len) { s[0] = 2; s[3] = 0; }
        } else {
            s[0] = 2; s[3] = 0;
            // An optional-start mismatch is flushed through the thought parser
            // once. A boundary found in that flush does not reopen a thought.
            for (int j = 0; j < n && s[0] == 2; ++j)
                thought_token(s, markers[offsets[0] + j], markers, offsets,
                              marker_count, tail_capacity);
            if (s[0] == 2)
                thought_token(s, token, markers, offsets, marker_count, tail_capacity);
        }
    } else {
        thought_token(s, token, markers, offsets, marker_count, tail_capacity);
    }
    if (s[4] && eos) { s[0] = 3; s[1] = 0; s[3] = 0; }
    return s[0] >= 0;
}

extern "C" __global__ void reasoning_budget(
    int64_t* committed, int64_t* scratch, const int32_t* ctx,
    const int64_t* proposals, const int64_t* decisions, uint32_t* masks, bool* activity,
    const int32_t* markers, const int32_t* offsets,
    int rows, int owners, int context_stride, int depth, int head_step,
    int state_width, int marker_count, int mask_words, int vocabulary, int commit_mode) {
    const int row = blockIdx.x;
    if (commit_mode) {
        if (threadIdx.x || row >= owners || row >= ctx[0]) return;
        int64_t* state = committed + (size_t)row * state_width;
        const int64_t accepted = decisions[2 * row], bonus = decisions[2 * row + 1];
        if (accepted < 0 || accepted > depth || bonus < 0 || bonus >= vocabulary) {
            state[0] = -2; return;
        }
        for (int j = 0; j <= accepted; ++j) {
            const int64_t token = j < accepted ? ctx[1 + row * context_stride + 2 + j] : bonus;
            if (token < 0 || token >= vocabulary || !accept_token(
                    state, token, markers, offsets, marker_count, state_width - 5)) {
                state[0] = -2; break;
            }
        }
        return;
    }
    const bool review = head_step < 0;
    const int owner = review ? row / (depth + 1) : row;
    const int prefix = review ? row % (depth + 1) : head_step;
    const bool owned = owner < owners && owner < ctx[0];
    int64_t* state = scratch + (size_t)row * state_width;
    if (threadIdx.x == 0) {
        for (int j = 0; j < state_width; ++j)
            state[j] = owned ? committed[(size_t)owner * state_width + j] : 0;
        if (!owned) state[0] = -1;
        if (owned) for (int j = 0; j < prefix; ++j) {
            const int64_t token = review ? ctx[1 + owner * context_stride + 2 + j]
                                          : proposals[j * owners + owner];
            if (token < 0 || token >= vocabulary || !accept_token(
                    state, token, markers, offsets, marker_count, state_width - 5)) {
                state[0] = -2; break;
            }
        }
        activity[row] = owned && state[0] != -1;
    }
    __syncthreads();
    const int64_t forced = forced_token(state, markers, offsets);
    for (int word = threadIdx.x; word < mask_words; word += blockDim.x) {
        uint32_t bits = state[0] == -2 ? 0u : forced < 0 ? ~0u :
            (word == forced / 32 ? 1u << (forced % 32) : 0u);
        if (state[0] == 3) {
            bits = 0;
            for (int j = offsets[marker_count]; j < offsets[marker_count + 1]; ++j)
                if (word == markers[j] / 32) bits |= 1u << (markers[j] % 32);
        }
        masks[(size_t)row * mask_words + word] = bits;
    }
}

