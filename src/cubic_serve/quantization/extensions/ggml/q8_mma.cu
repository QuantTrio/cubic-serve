// SPDX-License-Identifier: MIT
// Full signed 32-value integer dots; eight FP32 K residues then a fixed tree.
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

#if PIPELINED
// Original raw bytes in two transient stages; no resident encoding change.
static __device__ __forceinline__ void copy16(unsigned char* dst,const unsigned char* src,int bytes){
 if((reinterpret_cast<uintptr_t>(src)&15)==0){
  unsigned addr=__cvta_generic_to_shared(dst);
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;"::"r"(addr),"l"(src),"r"(bytes):"memory");
 }else{
  #pragma unroll
  for(int i=0;i<8;i++)reinterpret_cast<uint16_t*>(dst)[i]=i*2<bytes?reinterpret_cast<const uint16_t*>(src)[i]:0;
 }
}

constexpr int KB=16,WS=KB*34+16,XS=KB*36+16,STAGE=16*WS+8*XS;
static __device__ __forceinline__ void stage(unsigned char* dst,const weight_block* w,const block_q8_1* x,const int* activity,int m,int n,int chunks,int nr,int mr,int tile){
 const int tid=threadIdx.y*32+threadIdx.x,base=tile*KB;
 const int wb=min(KB,chunks-base)*34,xb=min(KB,chunks-base)*36;
#if PLANAR_Q8
 const int cb=min(KB,chunks-base)*32,db=min(KB,chunks-base)*2;
 for(int t=tid;t<16*(KB*32/16);t+=256){
  const int row=t/(KB*32/16),off=t%(KB*32/16)*16;
  const int bytes=nr+row<n?min(16,max(0,cb-off)):0;
  copy16(dst+row*WS+off,bytes?w+(int64_t(nr+row)*chunks+base)*32+off:w,bytes);
 }
 const auto* scales=w+int64_t((n+15)/16)*16*chunks*32;
 for(int t=tid;t<16*(KB*2/16);t+=256){
  const int row=t/(KB*2/16),off=t%(KB*2/16)*16;
  const int bytes=nr+row<n?min(16,max(0,db-off)):0;
  copy16(dst+row*WS+KB*32+off,bytes?scales+(int64_t(nr+row)*chunks+base)*2+off:scales,bytes);
 }
#else
 for(int t=tid;t<16*(KB*34/16);t+=256){
  int row=t/(KB*34/16),off=t%(KB*34/16)*16;
  int bytes=nr+row<n?min(16,max(0,wb-off)):0;
  const unsigned char* src=bytes?reinterpret_cast<const unsigned char*>(w+int64_t(nr+row)*chunks+base)+off:reinterpret_cast<const unsigned char*>(w);
  copy16(dst+row*WS+off,src,bytes);
 }
#endif
 for(int t=tid;t<8*(KB*36/16);t+=256){
  int row=t/(KB*36/16),off=t%(KB*36/16)*16;
  int bytes=mr+row<m&&(!activity||activity[mr+row]>0)?min(16,max(0,xb-off)):0;
  const unsigned char* src=bytes?reinterpret_cast<const unsigned char*>(x+int64_t(mr+row)*chunks+base)+off:reinterpret_cast<const unsigned char*>(x);
  copy16(dst+16*WS+row*XS+off,src,bytes);
 }
 asm volatile("cp.async.commit_group;":::"memory");
}
#endif
extern "C" __global__ void ggml_q8_mma(const weight_block* w0,const weight_block* w1,const block_q8_1* x,scalar* o0,scalar* o1,const int* activity,int m,int n0,int n1,int k,int64_t stride0,int64_t stride1) {
 const int member=int(blockIdx.x)>=(n0+15)/16,tile=int(blockIdx.x)-(member?(n0+15)/16:0);
 const auto* w=member?w1:w0;auto* out=member?o1:o0;int n=member?n1:n0;int64_t stride=member?stride1:stride0;
 int lane=threadIdx.x,group=lane/4,quad=lane%4,residue=threadIdx.y,nr=tile*16,mr=int(blockIdx.y)*8,chunks=k/32;
 float s0=0,s1=0,s2=0,s3=0;__shared__ float partial[8*8*16];
#if PIPELINED
 extern __shared__ __align__(16) unsigned char data[];
 const int tiles=(chunks+KB-1)/KB;stage(data,w,x,activity,m,n,chunks,nr,mr,0);
 for(int t=0;t<tiles;t++){
  if(t+1<tiles){stage(data+((t+1)&1)*STAGE,w,x,activity,m,n,chunks,nr,mr,t+1);asm volatile("cp.async.wait_group 1;":::"memory");}
  else asm volatile("cp.async.wait_group 0;":::"memory");
  __syncthreads();const unsigned char* cur=data+(t&1)*STAGE;
  #pragma unroll
  for(int z=0;z<KB/8;z++){int kb=t*KB+residue+z*8;if(kb<chunks){
  int a0=0,a1=0,a2=0,a3=0,b0=0,b1=0;

#if PLANAR_Q8
  const auto* aa=cur+group*WS+(kb%KB)*32;
  const auto* az=cur+(group+8)*WS+(kb%KB)*32;
  a0=get_int_b4(aa,quad);a2=get_int_b4(aa,quad+4);
  a1=get_int_b4(az,quad);a3=get_int_b4(az,quad+4);
#else
  const auto* aa=reinterpret_cast<const block_q8_0*>(cur+group*WS+(kb%KB)*34);
  const auto* az=reinterpret_cast<const block_q8_0*>(cur+(group+8)*WS+(kb%KB)*34);
  a0=get_int_b2(aa->qs,quad);a2=get_int_b2(aa->qs,quad+4);a1=get_int_b2(az->qs,quad);a3=get_int_b2(az->qs,quad+4);
#endif
  const auto* bx=reinterpret_cast<const block_q8_1*>(cur+16*WS+group*XS+(kb%KB)*36);
  b0=get_int_b4(bx->qs,quad);b1=get_int_b4(bx->qs,quad+4);
  int c0=0,c1=0,c2=0,c3=0;
  asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};" : "+r"(c0),"+r"(c1),"+r"(c2),"+r"(c3):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));

#if PLANAR_Q8
  float d0=__half2float(*reinterpret_cast<const half*>(cur+group*WS+KB*32+(kb%KB)*2));
  float d1=__half2float(*reinterpret_cast<const half*>(cur+(group+8)*WS+KB*32+(kb%KB)*2));
#else
  float d0=float(aa->d),d1=float(az->d);
#endif
  int r0=mr+2*quad,r1=r0+1;
  float y0=__low2float(reinterpret_cast<const block_q8_1*>(cur+16*WS+2*quad*XS+(kb%KB)*36)->ds);
  float y1=__low2float(reinterpret_cast<const block_q8_1*>(cur+16*WS+(2*quad+1)*XS+(kb%KB)*36)->ds);
  s0=__fmaf_rn(d0*y0,float(c0),s0);s1=__fmaf_rn(d0*y1,float(c1),s1);s2=__fmaf_rn(d1*y0,float(c2),s2);s3=__fmaf_rn(d1*y1,float(c3),s3);
  }}
  __syncthreads();
 }
#else
 for(int kb=residue;kb<chunks;kb+=8){
  int a0=0,a1=0,a2=0,a3=0,b0=0,b1=0;
#if PLANAR_Q8
  a0=q8_packed_word(w,n,chunks,nr+group,kb,quad);a2=q8_packed_word(w,n,chunks,nr+group,kb,quad+4);
  a1=q8_packed_word(w,n,chunks,nr+group+8,kb,quad);a3=q8_packed_word(w,n,chunks,nr+group+8,kb,quad+4);
#else
  if(nr+group<n){const auto* a=w+int64_t(nr+group)*chunks+kb;a0=get_int_b2(a->qs,quad);a2=get_int_b2(a->qs,quad+4);}
  if(nr+group+8<n){const auto* a=w+int64_t(nr+group+8)*chunks+kb;a1=get_int_b2(a->qs,quad);a3=get_int_b2(a->qs,quad+4);}
#endif
  if(mr+group<m && (!activity||activity[mr+group]>0)){const auto* b=x+int64_t(mr+group)*chunks+kb;b0=get_int_b4(b->qs,quad);b1=get_int_b4(b->qs,quad+4);}
  int c0=0,c1=0,c2=0,c3=0;
  asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};" : "+r"(c0),"+r"(c1),"+r"(c2),"+r"(c3):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
#if PLANAR_Q8
  float d0=float(q8_packed_scale(w,n,chunks,nr+group,kb)),d1=float(q8_packed_scale(w,n,chunks,nr+group+8,kb));
#else
  float d0=nr+group<n?float(w[int64_t(nr+group)*chunks+kb].d):0.0f,d1=nr+group+8<n?float(w[int64_t(nr+group+8)*chunks+kb].d):0.0f;
#endif
  int r0=mr+2*quad,r1=r0+1;
  float y0=r0<m&&(!activity||activity[r0]>0)?__low2float(x[int64_t(r0)*chunks+kb].ds):0.0f,y1=r1<m&&(!activity||activity[r1]>0)?__low2float(x[int64_t(r1)*chunks+kb].ds):0.0f;
  s0=__fmaf_rn(d0*y0,float(c0),s0);s1=__fmaf_rn(d0*y1,float(c1),s1);s2=__fmaf_rn(d1*y0,float(c2),s2);s3=__fmaf_rn(d1*y1,float(c3),s3);
 }
#endif
 int r0=mr+2*quad,r1=r0+1;
 if(r0<m&&nr+group<n)partial[(residue*8+r0-mr)*16+group]=s0;
 if(r1<m&&nr+group<n)partial[(residue*8+r1-mr)*16+group]=s1;
 if(r0<m&&nr+group+8<n)partial[(residue*8+r0-mr)*16+group+8]=s2;
 if(r1<m&&nr+group+8<n)partial[(residue*8+r1-mr)*16+group+8]=s3;
 __syncthreads();int tid=residue*32+lane;
 if(tid<128){int row=mr+tid/16,col=nr+tid%16;if(row<m&&col<n&&(!activity||activity[row]>0)){const float* a=partial+tid;float t0=__fadd_rn(a[0],a[4*128]),t1=__fadd_rn(a[128],a[5*128]),t2=__fadd_rn(a[2*128],a[6*128]),t3=__fadd_rn(a[3*128],a[7*128]);out[int64_t(row)*stride+col]=scalar(__fadd_rn(__fadd_rn(t0,t2),__fadd_rn(t1,t3)));}}
}
