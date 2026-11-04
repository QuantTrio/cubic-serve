// SPDX-License-Identifier: MIT
// Original GGML block decode/MMA with Cubic explicit logical row ownership.
#include "upstream/mmq-tiles.cuh"
#if PLANAR_Q8
#include "q8_storage.cuh"
#endif
#if IS_BF16
using scalar=__nv_bfloat16;
#else
using scalar=__half;
#endif
#if WEIGHT_TYPE==8
#define LOAD ggml_cuda_mmq_load_tiles_q8_0
constexpr int qk=QK8_0;
#elif WEIGHT_TYPE==14
#define LOAD ggml_cuda_mmq_load_tiles_q6_K
constexpr int qk=QK_K;
#elif WEIGHT_TYPE==20
#define LOAD ggml_cuda_mmq_load_tiles_iq4_nl
constexpr int qk=QK4_NL;
#elif WEIGHT_TYPE==21
#define LOAD ggml_cuda_mmq_load_tiles_iq3_s
constexpr int qk=QK_K;
#elif WEIGHT_TYPE==23
#define LOAD ggml_cuda_mmq_load_tiles_iq4_xs
constexpr int qk=QK_K;
#endif
#if WEIGHT_TYPE==20 || (WEIGHT_TYPE==8 && REGISTER_TRANSPORT)
// Both encodings have one original FP16 scale per 32 signed integer values.
// Their byte alignment and code arithmetic remain format-owned. Register
// transport preserves MMQ's integer dots and explicit C*dA then FMA-by-dB.
#if WEIGHT_TYPE==20
using register_block=block_iq4_nl;
constexpr int REGISTER_PREFETCH=1;
static __device__ __forceinline__ int2 load_register_codes(const register_block* q,int quad) {
 return get_int_from_table_16(get_int_b2(q->qs,quad),kvalues_iq4nl);
}
#else
using register_block=block_q8_0;
constexpr int REGISTER_PREFETCH=4;
static __device__ __forceinline__ int2 load_register_codes(const register_block* q,int quad) {
 // Q8_0's 34-byte blocks guarantee two-byte, rather than four-byte alignment.
 return make_int2(get_int_b2(q->qs,quad),get_int_b2(q->qs,quad+4));
}
#endif
extern "C" __global__ __launch_bounds__(128,4) void llama_mmq(
 const unsigned char* weights,const block_q8_1* x,scalar* output,
 const int* activity,const int* counts,const int* starts,const int* block_starts,const int* total,const int* row_map,
 int m,int n,int k,int row_bytes,int experts,int top_k,int64_t output_stride) {
 constexpr int J=ROW_TILE;
 const int lane=threadIdx.x,warp=threadIdx.y,group=lane/4,quad=lane%4;
 const int nr=int(blockIdx.x)*64+warp*16,chunks=k/32;
 int expert=0,local=int(blockIdx.y)*J;
 if(counts){
  if(int(blockIdx.y)>=*total)return;
  int lo=0,hi=experts;
  while(lo+1<hi){int mid=(lo+hi)/2;if(block_starts[mid]<=int(blockIdx.y))lo=mid;else hi=mid;}
  expert=lo;local=(int(blockIdx.y)-block_starts[expert])*J;
 }
 // Each MMA lane consumes one B row and two output/scale rows per fragment.
 // Keep only those scalar identities; dynamic indexing of own[J] spills the
 // entire map to per-thread local memory and repeats its divisions in K.
 int source[J/8],dest0[J/8],dest1[J/8],scale0[J/8],scale1[J/8];
 const int limit=counts?counts[expert]:m,base=counts?starts[expert]:0;
 #pragma unroll
 for(int t=0;t<J/8;t++){
  const int ib=local+t*8+group,ic0=local+t*8+2*quad,ic1=ic0+1;
  int ob=ib<limit?(counts?row_map[base+ib]:ib):-1;
  int o0=ic0<limit?(counts?row_map[base+ic0]:ic0):-1;
  int o1=ic1<limit?(counts?row_map[base+ic1]:ic1):-1;
  int rb=ob<0?-1:ob/top_k,r0=o0<0?-1:o0/top_k,r1=o1<0?-1:o1/top_k;
  if(rb<0 || rb>=m || (activity&&activity[rb]<=0))rb=-1;
  if(r0<0 || r0>=m || (activity&&activity[r0]<=0)){o0=-1;r0=-1;}
  if(r1<0 || r1>=m || (activity&&activity[r1]<=0)){o1=-1;r1=-1;}
  source[t]=rb;dest0[t]=o0;dest1[t]=o1;scale0[t]=r0;scale1[t]=r1;
 }
 float sums[J/8][4]={};
 for(int kb0=0;kb0<chunks;kb0+=REGISTER_PREFETCH){
  int aa[REGISTER_PREFETCH][4];float dd[REGISTER_PREFETCH][2];
  #pragma unroll
  for(int z=0;z<REGISTER_PREFETCH;z++){
   const int kb=kb0+z;
  int a0=0,a1=0,a2=0,a3=0;float d0=0.0f,d1=0.0f;
  if(kb<chunks && nr+group<n){
#if PLANAR_Q8
   const auto* w=q8_packed_expert(weights,n,chunks,expert);
   a0=q8_packed_word(w,n,chunks,nr+group,kb,quad);
   a2=q8_packed_word(w,n,chunks,nr+group,kb,quad+4);
   d0=__half2float(q8_packed_scale(w,n,chunks,nr+group,kb));
#else
   const register_block* q=reinterpret_cast<const register_block*>(weights)+(int64_t(expert)*n+nr+group)*chunks+kb;
   int2 v=load_register_codes(q,quad);a0=v.x;a2=v.y;d0=__half2float(q->d);
#endif
  }
  if(kb<chunks && nr+group+8<n){
#if PLANAR_Q8
   const auto* w=q8_packed_expert(weights,n,chunks,expert);
   a1=q8_packed_word(w,n,chunks,nr+group+8,kb,quad);
   a3=q8_packed_word(w,n,chunks,nr+group+8,kb,quad+4);
   d1=__half2float(q8_packed_scale(w,n,chunks,nr+group+8,kb));
#else
   const register_block* q=reinterpret_cast<const register_block*>(weights)+(int64_t(expert)*n+nr+group+8)*chunks+kb;
   int2 v=load_register_codes(q,quad);a1=v.x;a3=v.y;d1=__half2float(q->d);
#endif
  }
   aa[z][0]=a0;aa[z][1]=a1;aa[z][2]=a2;aa[z][3]=a3;dd[z][0]=d0;dd[z][1]=d1;
  }
  #pragma unroll
  for(int z=0;z<REGISTER_PREFETCH;z++){
   const int kb=kb0+z;if(kb>=chunks)continue;
   const int a0=aa[z][0],a1=aa[z][1],a2=aa[z][2],a3=aa[z][3];const float d0=dd[z][0],d1=dd[z][1];
  #pragma unroll
  for(int j0=0;j0<J;j0+=8){
   int b0=0,b1=0;const int r=source[j0/8];
   if(r>=0){
    const block_q8_1* q=x+int64_t(r)*chunks+kb;b0=get_int_b4(q->qs,quad);b1=get_int_b4(q->qs,quad+4);
   }
   int c0=0,c1=0,c2=0,c3=0;
   asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};":
       "+r"(c0),"+r"(c1),"+r"(c2),"+r"(c3):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
   const int r0=scale0[j0/8],r1=scale1[j0/8];
   const float y0=r0>=0?__low2float(x[int64_t(r0)*chunks+kb].ds):0.0f;
   const float y1=r1>=0?__low2float(x[int64_t(r1)*chunks+kb].ds):0.0f;
   sums[j0/8][0]=__fmaf_rn(__fmul_rn(float(c0),d0),y0,sums[j0/8][0]);sums[j0/8][1]=__fmaf_rn(__fmul_rn(float(c1),d0),y1,sums[j0/8][1]);
   sums[j0/8][2]=__fmaf_rn(__fmul_rn(float(c2),d1),y0,sums[j0/8][2]);sums[j0/8][3]=__fmaf_rn(__fmul_rn(float(c3),d1),y1,sums[j0/8][3]);
  }
  }
 }
 #pragma unroll
 for(int j0=0;j0<J;j0+=8){
  const int r0=dest0[j0/8],r1=dest1[j0/8];
  if(nr+group<n){
   if(r0>=0)output[int64_t(r0)*output_stride+nr+group]=scalar(sums[j0/8][0]);
   if(r1>=0)output[int64_t(r1)*output_stride+nr+group]=scalar(sums[j0/8][1]);
  }
  if(nr+group+8<n){
   if(r0>=0)output[int64_t(r0)*output_stride+nr+group+8]=scalar(sums[j0/8][2]);
   if(r1>=0)output[int64_t(r1)*output_stride+nr+group+8]=scalar(sums[j0/8][3]);
  }
 }
}
#else
extern "C" __global__ void llama_mmq(
 const unsigned char* weights,const block_q8_1* x,scalar* output,
 const int* activity,const int* counts,const int* starts,const int* block_starts,const int* total,const int* row_map,
 int m,int n,int k,int row_bytes,int experts,int top_k,int64_t output_stride) {
 constexpr int I=64,J=ROW_TILE,S=76;
 extern __shared__ __align__(16) int data[];
 int* ty=data;int* tx=ty+J*MMQ_TILE_Y_K;
 const int ni=int(blockIdx.x)*I;
 int expert=0,local=int(blockIdx.y)*J;
 if(counts) {
  if(int(blockIdx.y)>=*total)return;
  int lo=0,hi=experts;
  while(lo+1<hi) {int mid=(lo+hi)/2;if(block_starts[mid]<=int(blockIdx.y))lo=mid;else hi=mid;}
  expert=lo;local=(int(blockIdx.y)-block_starts[expert])*J;
 }
#if QUERY_ROW_OWNERS
 // Wide tiles query immutable metadata instead of spilling a dynamic array.
 auto owner_at=[&](int j){return counts ? (local+j<counts[expert] ? row_map[starts[expert]+local+j] : -1) : (local+j<m ? local+j : -1);};
#else
 int own[J];
 for(int j=0;j<J;j++)own[j]=counts ? (local+j<counts[expert] ? row_map[starts[expert]+local+j] : -1) : (local+j<m ? local+j : -1);
#define owner_at(j) own[j]
#endif
 const int tid=threadIdx.y*32+threadIdx.x;
 constexpr int YLOADS=(J*MMQ_TILE_Y_K+127)/128;
 int source_row[YLOADS];
 #pragma unroll
 for(int t=0;t<YLOADS;t++) {
  const int l=tid+t*128;
  const int owner=l<J*MMQ_TILE_Y_K?owner_at(l/MMQ_TILE_Y_K):-1;
  int row=owner<0?-1:owner/top_k;
  if(row<0 || row>=m || (activity&&activity[row]<=0))row=-1;
  source_row[t]=row;
 }
 float sum[J*I/128]={0.0f};
 for(int kk=0;kk<k;kk+=256) {
#if PLANAR_Q8
  // Preserve the upstream Q8 MMQ shared tile and its arithmetic. Only the
  // global code/scale transport changes; incomplete K blocks are zero.
  const auto* wp=q8_packed_expert(weights,n,k/32,expert);
  const int lane=threadIdx.x,warp=threadIdx.y,chunks=k/32,kb0=kk/32;
  #pragma unroll
  for(int i0=0;i0<I;i0+=4){
   const int i=i0+warp,r=ni+min(i,n-1-ni),kb=kb0+lane/8,word=lane%8;
   tx[i*S+lane]=kb<chunks?q8_packed_word(wp,n,chunks,r,kb,word):0;
   tx[i*S+MMQ_TILE_NE_K+lane]=kb+4<chunks?q8_packed_word(wp,n,chunks,r,kb+4,word):0;
  }
  float* df=reinterpret_cast<float*>(tx+2*MMQ_TILE_NE_K);
  #pragma unroll
  for(int i0=0;i0<I;i0+=16){
   const int i=i0+warp*4+lane/8,r=ni+min(i,n-1-ni),kb=kb0+lane%8;
   df[i*S+lane%8]=kb<chunks?__half2float(q8_packed_scale(wp,n,chunks,r,kb)):0.0f;
  }
#else
  const char* wp=reinterpret_cast<const char*>(weights)+(int64_t(expert)*n+ni)*row_bytes;
  const int valid_columns=min(I,n-ni);
  if(kk+256>k) {
   constexpr int qb=WEIGHT_TYPE==8 ? sizeof(block_q8_0) : sizeof(block_iq4_nl);
   constexpr int staged_row_bytes=256/qk*qb;
   char* staged=reinterpret_cast<char*>(tx+I*S);
   for(int t=tid;t<I*staged_row_bytes;t+=128) {
    int i=t/staged_row_bytes,b=t%staged_row_bytes;
    staged[t]=i<valid_columns && b<(k-kk)/qk*qb ? wp[int64_t(i)*row_bytes+kk/qk*qb+b] : 0;
   }
   __syncthreads();
   LOAD<WEIGHT_TYPE,J,true>(staged,tx,0,min(I-1,n-1-ni),256/qk);
  } else {
   LOAD<WEIGHT_TYPE,J,true>(wp,tx,kk/qk,min(I-1,n-1-ni),k/qk);
  }
#endif
  for(int h=0;h<2;h++) {
   #pragma unroll
   for(int t=0;t<YLOADS;t++) {
    const int l=tid+t*128;
    if(l<J*MMQ_TILE_Y_K) {
     const int p=l%MMQ_TILE_Y_K,r=source_row[t];
     int value=0;
     if(r>=0 && kk+h*128+(p<4 ? p*32 : ((p-4)/8)*32)<k) {
      const block_q8_1* q=x+int64_t(r)*(k/32)+(kk+h*128)/32;
      if(p<4)value=__float_as_int(__low2float(q[p].ds));
      else {int qi=(p-4)/8,byte=((p-4)%8)*4;value=*reinterpret_cast<const int*>(q[qi].qs+byte);}
     }
     ty[l]=value;
    }
   }
   __syncthreads();
#if WEIGHT_TYPE==14
   ggml_cuda_mmq_vec_dot_q6_K_q8_1_mma<WEIGHT_TYPE,J,true>(tx,ty,sum,h*MMQ_TILE_NE_K);
#else
   ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma<WEIGHT_TYPE,J,true,MMQ_Q8_1_DS_LAYOUT_D4>(tx,ty,sum,h*MMQ_TILE_NE_K);
#endif
   __syncthreads();
  }
 }
 using tile_C=tile<16,8,int>;
 int i0=int(threadIdx.y)*16;
 #pragma unroll
 for(int j0=0;j0<J;j0+=8) {
  #pragma unroll
  for(int l=0;l<tile_C::ne;l++) {
   int j=owner_at(j0+tile_C::get_j(l)),i=ni+i0+tile_C::get_i(l);
   if(i<n && j>=0 && j<m*top_k && (!activity || activity[j/top_k]>0))output[int64_t(j)*output_stride+i]=scalar(sum[j0/8*tile_C::ne+l]);
  }
 }
}
#undef owner_at
#endif
