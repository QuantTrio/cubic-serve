// Select original encoded rows first; never expand an entire embedding table.
#include "decode.cuh"
#include "q8_storage.cuh"

extern "C" __global__ void ggml_embedding(
    const unsigned char* ids, const int* activity,
    const long long* starts, const long long* ends, const long long* pointers,
    const int* row_bytes, const int* types, const int* offsets, const int* layouts,
    const unsigned char* grid, const unsigned char* scale, scalar* output,
    int lookups, int heads, int width, int shards, int ids_bytes, int scale_type) {
    const int lookup = blockIdx.x;
    if (lookup >= lookups || (activity && activity[lookup / heads] == 0)) return;
    const long long token = ids_bytes == 8
        ? reinterpret_cast<const long long*>(ids)[lookup]
        : reinterpret_cast<const int*>(ids)[lookup];
    int lo = 0, hi = shards;
    while (lo < hi) {
        const int middle = (lo + hi) / 2;
        if (starts[middle] <= token) lo = middle + 1; else hi = middle;
    }
    const int owner = lo - 1;
    const bool owned = owner >= 0 && token < ends[owner];
    float direct_scale = 1.0f;
    if (scale) {
        direct_scale = scale_type == 0 ? reinterpret_cast<const float*>(scale)[0]
            : scale_type == 1 ? half_value(scale)
            : __bfloat162float(__ushort_as_bfloat16(u16(scale)));
    }
    for (int column = blockIdx.y * blockDim.x + threadIdx.x;
         column < width && column < (blockIdx.y + 1) * blockDim.x; column += blockDim.x) {
        float value = 0.0f;
        if (owned) {
            const auto* base = reinterpret_cast<const unsigned char*>(pointers[owner]);
            if (layouts[owner]) {
                const int n=ends[owner]-starts[owner],r=token-starts[owner],kb=column/32;
                const int word=q8_packed_word(base,n,width/32,r,kb,(column%32)/4);
                const int8_t code=static_cast<int8_t>(static_cast<unsigned>(word) >> (8*(column%4)));
                value=__fmul_rn(__half2float(q8_packed_scale(base,n,width/32,r,kb)),float(code));
            } else {
                const auto* row=base+(token-starts[owner])*row_bytes[owner];
                value = original_weight(row, column + offsets[owner], types[owner], grid);
            }
        }
        output[std::int64_t(lookup) * width + column] = rounded(__fmul_rn(value, direct_scale));
    }
}
