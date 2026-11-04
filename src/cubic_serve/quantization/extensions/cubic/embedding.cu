#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#if OUTPUT_BF16
using Output = __nv_bfloat16;
__device__ Output round_output(float x) { return __float2bfloat16_rn(x); }
__device__ float output_float(Output x) { return __bfloat162float(x); }
#else
using Output = __half;
__device__ Output round_output(float x) { return __float2half_rn(x); }
__device__ float output_float(Output x) { return __half2float(x); }
#endif

// Metadata may follow an odd byte count for packed W3/W5/W7 rows.
__device__ unsigned short load_u16(const unsigned char* p) {
    return p[0] | (static_cast<unsigned short>(p[1]) << 8);
}

extern "C" __global__ void embedding_lookup(
    const long long* ids, const unsigned char* weight, const unsigned short* metadata,
    const __half* curve_a, const __half* curve_b, const int* partitions,
    const unsigned long long* host, Output* output,
    int width, int start, int stop, int weight_bytes, int metadata_bytes, int row_bytes) {
    constexpr int bits = MIRROR_BITS, group_in = MIRROR_GI, group_out = MIRROR_GO;
    constexpr int levels_count = 1 << (bits - 1);
    extern __shared__ unsigned char packet[];
    __shared__ float levels[levels_count];
    for (int q = threadIdx.x; q < levels_count; q += blockDim.x) {
        levels[q] = __fdiv_rn(float(q), float(levels_count - 1));
    }
    int row = blockIdx.x, tid = threadIdx.x;
    long long token = ids[row];
    if (token >= start && token < stop) {
        int local_row = int(token) - start;
        int group = local_row / group_out, partition = partitions[group];
        for (int v = tid; v < row_bytes / 16; v += blockDim.x) {
            int offset = v * 16;
            uint4 value;
            if (weight_bytes % 16 == 0 && offset + 16 <= weight_bytes) {
                value = reinterpret_cast<const uint4*>(
                    weight + size_t(local_row) * weight_bytes)[v];
            } else {
                union { uint4 vector; unsigned char bytes[16]; } u;
                for (int j = 0; j < 16; ++j) {
                    int x = offset + j;
                    if (x < weight_bytes) {
                        u.bytes[j] = weight[size_t(local_row) * weight_bytes + x];
                    } else if (x < weight_bytes + metadata_bytes) {
                        u.bytes[j] = reinterpret_cast<const unsigned char*>(metadata)[
                            size_t(group) * metadata_bytes + x - weight_bytes];
                    } else if (x < weight_bytes + metadata_bytes + 8) {
                        u.bytes[j] = reinterpret_cast<const unsigned char*>(curve_a)[
                            partition * 8 + x - weight_bytes - metadata_bytes];
                    } else if (x < weight_bytes + metadata_bytes + 16) {
                        u.bytes[j] = reinterpret_cast<const unsigned char*>(curve_b)[
                            partition * 8 + x - weight_bytes - metadata_bytes - 8];
                    } else {
                        u.bytes[j] = 0;
                    }
                }
                value = u.vector;
            }
            reinterpret_cast<uint4*>(packet)[v] = value;
        }
    } else {
        const uint4* source = reinterpret_cast<const uint4*>(
            host[0] + size_t(token) * row_bytes);
        for (int v = tid; v < row_bytes / 16; v += blockDim.x) {
            reinterpret_cast<uint4*>(packet)[v] = source[v];
        }
    }
    __syncthreads();
    for (int column = tid; column < width; column += blockDim.x) {
        int bit = column * bits, byte = bit / 8, raw = packet[byte];
        if (byte + 1 < weight_bytes) raw |= int(packet[byte + 1]) << 8;
        raw = (raw >> (bit % 8)) & ((1 << bits) - 1);
        int code = raw >= levels_count ? raw - (1 << bits) : raw;
        if (code == -levels_count) code = 0;
        unsigned short meta = load_u16(packet + weight_bytes + 2 * (column / group_in));
        union { unsigned short u; __half h; } scale_bits, a_bits, b_bits;
        scale_bits.u = (meta & 0x3fff) << 1;
        int curve_offset = weight_bytes + metadata_bytes + 2 * (meta >> 14);
        a_bits.u = load_u16(packet + curve_offset);
        b_bits.u = load_u16(packet + curve_offset + 8);
        float scale = __half2float(scale_bits.h);
        float a = __half2float(a_bits.h), b = __half2float(b_bits.h);
        float t = levels[code < 0 ? -code : code];
        // Preserve Cubic's separate FP32 rounding steps; no FMA contraction.
        float n = __fmul_rn(t, __fadd_rn(a, __fmul_rn(t,
            __fadd_rn(b, __fmul_rn(t, __fsub_rn(__fsub_rn(1.f, a), b))))));
        float value = __fmul_rn(code < 0 ? -n : n, scale);
        Output out = round_output(value);
        // The original ordered rank sum starts from +0 in FP32.
        if (output_float(out) == 0.f) out = round_output(0.f);
        output[size_t(row) * width + column] = out;
    }
}
