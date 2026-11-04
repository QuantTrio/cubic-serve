// SPDX-License-Identifier: MIT
#include "q8_storage.cuh"
extern "C" __global__ void ggml_pack_q8(
    const unsigned char* raw, unsigned char* output, int n, int chunks, int experts) {
    const int nt = (n + 15) / 16;
    const int64_t per_expert = int64_t(chunks) * nt * 128;
    const int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= int64_t(experts) * per_expert) return;
    const int expert = index / per_expert;
    const int64_t local = index % per_expert;
    const int word = local % 8;
    const int row = (local / 8) % (nt * 16);
    const int kb = local / (nt * 128);
    const auto* source = raw + (int64_t(expert) * n * chunks + int64_t(row) * chunks + kb) * 34;
    const unsigned int value = row < n ?
        unsigned(source[2 + 4 * word]) | (unsigned(source[3 + 4 * word]) << 8) |
        (unsigned(source[4 + 4 * word]) << 16) | (unsigned(source[5 + 4 * word]) << 24) : 0;
    auto* target = output + int64_t(expert) * chunks * nt * 544;
    reinterpret_cast<unsigned int*>(target)[(int64_t(row) * chunks + kb) * 8 + word] = value;
    if (word == 0) {
        const unsigned short scale = row < n ? unsigned(source[0]) | (unsigned(source[1]) << 8) : 0;
        reinterpret_cast<unsigned short*>(target + int64_t(chunks) * nt * 512)[
            int64_t(row) * chunks + kb] = scale;
    }
}
