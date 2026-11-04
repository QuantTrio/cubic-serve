#include <cuda_runtime.h>
// FP16/BF16 lookup preserves storage bits, with rank-sum signed-zero semantics.
extern "C" __global__ void embedding_lookup(const long long* ids,
    const unsigned short* local, const unsigned long long* host,
    unsigned short* out, int width, int start, int stop, int stride) {
    int row=blockIdx.x;
    long long token=ids[row];
    const unsigned short* source = token>=start && token<stop
        ? local+(token-start)*width
        : reinterpret_cast<const unsigned short*>(host[0]+token*stride);
    for(int col=threadIdx.x;col<width;col+=blockDim.x){
        unsigned short value=source[col];
        out[size_t(row)*width+col]=(value&0x7fff)==0 ? 0 : value;
    }
}
