// Original dense A8 fragment layout; strict ascending FP32 fold and one final store.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
extern "C" __global__ void reduce_partials(
 const float* partials, void* output, int m, int n, int bm_shift, int bn_shift, int n_tiles, int stride, int parts, int dtype) {
 int global_r=blockIdx.y*4+threadIdx.x/64;
 int global_c=blockIdx.x*64+threadIdx.x%64;
 if(global_r>=m||global_c>=n)return;
 long long offset=static_cast<long long>(global_r)*n+global_c;
 int tile=(global_r>>bm_shift)*n_tiles+(global_c>>bn_shift);
 int r=global_r&((1<<bm_shift)-1),c=global_c&((1<<bn_shift)-1);
 int bn=1<<bn_shift;
 int lane=32*(c/32)+4*(r%8)+(c%8)/2;
 int frag=16*(r/16)+8*((c%32)/16)+4*((c%16)/8)+2*((r%16)/8)+c%2;
 int local=(bn*(frag/4)+lane)*4+frag%4;
 const float* p=partials+static_cast<long long>(tile)*parts*stride+local;
 float sum=0.f;
 for(int j=0;j<parts;j+=4){
  float v0=p[static_cast<long long>(j)*stride];
  float v1=j+1<parts?p[static_cast<long long>(j+1)*stride]:0.f;
  float v2=j+2<parts?p[static_cast<long long>(j+2)*stride]:0.f;
  float v3=j+3<parts?p[static_cast<long long>(j+3)*stride]:0.f;
  sum=__fadd_rn(sum,v0);
  if(j+1<parts)sum=__fadd_rn(sum,v1);
  if(j+2<parts)sum=__fadd_rn(sum,v2);
  if(j+3<parts)sum=__fadd_rn(sum,v3);
 }
 if(dtype==0)static_cast<__nv_bfloat16*>(output)[offset]=__float2bfloat16_rn(sum);
 else if(dtype==1)static_cast<half*>(output)[offset]=__float2half_rn(sum);
 else static_cast<float*>(output)[offset]=sum;
}
