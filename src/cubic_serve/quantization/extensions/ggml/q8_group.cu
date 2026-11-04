// SPDX-License-Identifier: MIT
// Q8_0 integer MMA with the original MMVQ floating reduction.

#include "upstream/vecdotq.cuh"
#if PLANAR_Q8
#include "q8_storage.cuh"
using weight_block=unsigned char;
#else
using weight_block=block_q8_0;
#endif
#if IS_BF16
using scalar=__nv_bfloat16;
#else
using scalar=__half;
#endif
extern "C" __global__ void ggml_q8_group(const weight_block* w0,const weight_block* w1,const block_q8_1* x,scalar* out0,scalar* out1,const int* activity,int m,int n0,int n1,int k,int64_t stride0,int64_t stride1) {
 const int member=int(blockIdx.x)>=(n0+15)/16; const int tile_id=int(blockIdx.x)-(member ? (n0+15)/16:0);
 const weight_block* w=member ? w1:w0;scalar* output=member ? out1:out0;int n=member ? n1:n0;int64_t stride=member ? stride1:stride0;
 int lane=threadIdx.x,group=lane/4,quad=lane%4;
 int nr=tile_id*16,mr=int(blockIdx.y)*4,residue=threadIdx.y;
 __shared__ float partial[8*4*16*4]; int chunks=k/32;float s[2][4]={};
 // Each warp keeps one of the original eight K-block residues. MMA columns
 // carry the original four independent eight-byte integer dots per token.
 for(int kb=residue;kb<chunks;kb+=8) {
  int a0=0,a1=0,a2=0,a3=0;
#if PLANAR_Q8
  a0=q8_packed_word(w,n,chunks,nr+group,kb,quad);a2=q8_packed_word(w,n,chunks,nr+group,kb,quad+4);
  a1=q8_packed_word(w,n,chunks,nr+group+8,kb,quad);a3=q8_packed_word(w,n,chunks,nr+group+8,kb,quad+4);
  float d0=float(q8_packed_scale(w,n,chunks,nr+group,kb));
  float d1=float(q8_packed_scale(w,n,chunks,nr+group+8,kb));
#else
  if(nr+group<n){const auto* a=w+int64_t(nr+group)*chunks+kb;a0=get_int_b2(a->qs,quad);a2=get_int_b2(a->qs,quad+4);}
  if(nr+group+8<n){const auto* a=w+int64_t(nr+group+8)*chunks+kb;a1=get_int_b2(a->qs,quad);a3=get_int_b2(a->qs,quad+4);}
  float d0=nr+group<n ? float(w[int64_t(nr+group)*chunks+kb].d):0.0f;
  float d1=nr+group+8<n ? float(w[int64_t(nr+group+8)*chunks+kb].d):0.0f;
#endif
  #pragma unroll
  for(int h=0;h<2;h++) {
   int j=8*h+group,row=mr+j/4,quarter=j%4,b0=0,b1=0;
   if(row<m && (!activity || activity[row]>0)) {
    const auto* b=x+int64_t(row)*chunks+kb;
    if(quad/2==quarter)b0=get_int_b4(b->qs,quad);
    if(2+quad/2==quarter)b1=get_int_b4(b->qs,quad+4);
   }
   int c0=0,c1=0,c2=0,c3=0;
   asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};" : "+r"(c0),"+r"(c1),"+r"(c2),"+r"(c3):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
   int rowc=mr+(8*h+2*quad)/4;
   float dy=rowc<m && (!activity || activity[rowc]>0) ? __low2float(x[int64_t(rowc)*chunks+kb].ds):0.0f;
   s[h][0]=__fmaf_rn(d0*dy,float(c0),s[h][0]);s[h][1]=__fmaf_rn(d0*dy,float(c1),s[h][1]);
   s[h][2]=__fmaf_rn(d1*dy,float(c2),s[h][2]);s[h][3]=__fmaf_rn(d1*dy,float(c3),s[h][3]);
  }
 }
 #pragma unroll
 for(int h=0;h<2;h++) {
  int j=8*h+2*quad,row=mr+j/4,q=j%4;
  if(row<m && nr+group<n){int64_t off=((int64_t(residue)*4+row-mr)*16+group)*4;partial[off+q]=s[h][0];partial[off+q+1]=s[h][1];}
  if(row<m && nr+group+8<n){int64_t off=((int64_t(residue)*4+row-mr)*16+group+8)*4;partial[off+q]=s[h][2];partial[off+q+1]=s[h][3];}
 }

 // Reproduce XOR masks 16, 8, 4, 2, 1, then store once in the activation dtype.
 __syncthreads();
 int tid=threadIdx.y*32+lane;
 if(tid<64) {
  int row=mr+tid/16,col=nr+tid%16;
  if(row<m && col<n && (!activity || activity[row]>0)) {
   float v[4];
   #pragma unroll
   for(int q=0;q<4;q++) {
    const float* a=partial+(tid/16*16+tid%16)*4+q;constexpr int jump=4*16*4;
    float v0=__fadd_rn(a[0*jump],a[4*jump]),v1=__fadd_rn(a[1*jump],a[5*jump]),v2=__fadd_rn(a[2*jump],a[6*jump]),v3=__fadd_rn(a[3*jump],a[7*jump]);
    v[q]=__fadd_rn(__fadd_rn(v0,v2),__fadd_rn(v1,v3));
   }
   output[int64_t(row)*stride+col]=scalar(__fadd_rn(__fadd_rn(v[0],v[2]),__fadd_rn(v[1],v[3])));
  }
 }
}
