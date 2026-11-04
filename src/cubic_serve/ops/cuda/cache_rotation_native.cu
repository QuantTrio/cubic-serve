// Out-of-place, native-precision per-head rotation. All geometry is runtime data.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>
#if NATIVE_BF16
using scalar=__nv_bfloat16;
#else
using scalar=__half;
#endif

extern "C" __global__ void cache_rotation_native(
    const scalar* __restrict__ first,const scalar* __restrict__ second,
    const scalar* __restrict__ third,scalar* __restrict__ first_output,
    scalar* __restrict__ second_output,scalar* __restrict__ third_output,
    int first_vectors,int second_vectors,int third_vectors,int dim,int group,float normalization) {
    int vector=blockIdx.x;
    const scalar* input;
    scalar* output;
    if(vector<first_vectors) {input=first;output=first_output;}
    else if((vector-=first_vectors)<second_vectors) {input=second;output=second_output;}
    else {vector-=second_vectors;if(vector>=third_vectors)return;input=third;output=third_output;}
    extern __shared__ float values[];
    const int64_t offset=int64_t(vector)*dim;
    for(int d=threadIdx.x;d<dim;d+=blockDim.x)values[d]=float(input[offset+d]);
    __syncthreads();
    for(int step=1;step<group;step*=2) {
        // Each pair owns both addresses for the stage, then all pairs advance.
        for(int pair=threadIdx.x;pair<dim/2;pair+=blockDim.x) {
            const int block=pair/(group/2),local=pair%(group/2);
            const int first=block*group+(local/step)*(2*step)+local%step;
            const float a=values[first],b=values[first+step];
            values[first]=__fadd_rn(a,b);values[first+step]=__fsub_rn(a,b);
        }
        __syncthreads();
    }
    for(int d=threadIdx.x;d<dim;d+=blockDim.x)
        output[offset+d]=scalar(__fmul_rn(values[d],normalization));
}
