// Packed dots copied from llama.cpp, see upstream/manifest.json and LICENSE.
#include "upstream/vecdotq.cuh"
#if PLANAR_Q8
#include "q8_storage.cuh"
#endif

#if IS_BF16
using scalar = __nv_bfloat16;
#else
using scalar = __half;
#endif

#if QUANTIZE
// quantize.cu::quantize_q8_1: same 32-value scale, roundf and FP16 ds.
// Only source dtype/row addressing and explicit Cubic row activity differ.
extern "C" __global__ void llama_quantize(
    const scalar* x, block_q8_1* y, const int* activity, int m, int k) {
    const int kk = int(blockIdx.x)*blockDim.x + threadIdx.x;
    const int row = int(blockIdx.y), lane = threadIdx.x%32;
    if (kk >= k || row >= m) return;
    const float xi = activity && activity[row] <= 0 ? 0.0f : float(x[int64_t(row)*k+kk]);
    float amax = fabsf(xi), sum = xi;
    #pragma unroll
    for (int mask=16; mask; mask>>=1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, mask));
        sum += __shfl_xor_sync(0xffffffff, sum, mask);
    }
    const float d = amax/127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi/d);
    block_q8_1* dst = y + int64_t(row)*(k/32) + kk/32;
    dst->qs[lane] = q;
    if (lane == 0) dst->ds = make_half2(d, sum);
}
#else
#if WEIGHT_TYPE == 8
constexpr int qk=QK8_0, qi=QI8_0, vdr=VDR_Q8_0_Q8_1_MMVQ;
#define DOT vec_dot_q8_0_q8_1
#elif WEIGHT_TYPE == 14
constexpr int qk=QK_K, qi=QI6_K, vdr=VDR_Q6_K_Q8_1_MMVQ;
#define DOT vec_dot_q6_K_q8_1
#elif WEIGHT_TYPE == 20
constexpr int qk=QK4_NL, qi=QI4_NL, vdr=VDR_IQ4_NL_Q8_1_MMVQ;
#define DOT vec_dot_iq4_nl_q8_1
#elif WEIGHT_TYPE == 21
constexpr int qk=QK_K, qi=QI3_S, vdr=VDR_IQ3_S_Q8_1_MMVQ;
#define DOT vec_dot_iq3_s_q8_1
#elif WEIGHT_TYPE == 23
constexpr int qk=QK_K, qi=QI4_XS, vdr=VDR_IQ4_XS_Q8_1_MMVQ;
#define DOT vec_dot_iq4_xs_q8_1
#endif
// mmvq.cu::mul_mat_vec_q_moe contraction, generalized to Cubic's existing
// compact route-row metadata. Each warp reuses activation blocks across its
// original expert's output rows, with an independent unchanged sum per row.
// No expert remapping, host routing readback, or activation/weight expansion.
extern "C" __global__ void llama_matmul(
    const unsigned char* weights, const block_q8_1* x, scalar* output,
    const int* activity, const int* counts, const int* starts, const int* row_map,
    int m, int n, int k, int row_bytes, int experts, int top_k, int64_t output_stride) {
    const int column = ROWS_PER_WARP*(int(blockIdx.x)*blockDim.y + threadIdx.y);
    const int slot = int(blockIdx.y);
    if (column >= n) return;
    int expert = 0;
    if (counts) {
        if (slot >= starts[experts-1] + counts[experts-1]) return;
        int lo = 0, hi = experts;
        while (lo + 1 < hi) {
            const int mid = (lo + hi)/2;
            if (starts[mid] <= slot) lo = mid; else hi = mid;
        }
        expert = lo;
    }
    const int route = counts ? row_map[slot] : slot;
    const int row = route/top_k;
    if (row >= m || (activity && activity[row] <= 0)) return;
    const void* vx = weights + (int64_t(expert)*n+column)*row_bytes;
#if PLANAR_Q8
    vx = q8_packed_expert(weights,n,k/32,expert);
#endif
    const block_q8_1* y = x + int64_t(row)*(k/32);
    constexpr int blocks_per_iter = vdr*32/qi;
    float tmp[ROWS_PER_WARP] = {};
    for (int kbx = threadIdx.x/(qi/vdr); kbx < k/qk; kbx += blocks_per_iter) {
        const int kby = kbx*(qk/QK8_1);
        const int kqs = vdr*(threadIdx.x%(qi/vdr));
        #pragma unroll
        for (int i=0; i<ROWS_PER_WARP; ++i)
            if (column+i < n) {
#if PLANAR_Q8
                int v[vdr],u[vdr];
                #pragma unroll
                for(int z=0;z<vdr;z++) {
                    v[z]=q8_packed_word(static_cast<const unsigned char*>(vx),n,k/32,column+i,kbx,kqs+z);
                    u[z]=get_int_b4(y[kby].qs,kqs+z);
                }
                tmp[i] += vec_dot_q8_0_q8_1_impl<float,vdr>(v,u,
                    float(q8_packed_scale(static_cast<const unsigned char*>(vx),n,k/32,column+i,kbx)),
                    __low2float(y[kby].ds));
#else
                tmp[i] += DOT(static_cast<const unsigned char*>(vx)+int64_t(i)*row_bytes,
                              y+kby, kbx, kqs);
#endif
            }
    }
    #pragma unroll
    for (int i=0; i<ROWS_PER_WARP; ++i) {
        #pragma unroll
        for (int mask=16; mask; mask>>=1)
            tmp[i] += __shfl_xor_sync(0xffffffff, tmp[i], mask);
        if (threadIdx.x == 0 && column+i < n)
            output[int64_t(route)*output_stride+column+i] = scalar(tmp[i]);
    }
}
#endif
