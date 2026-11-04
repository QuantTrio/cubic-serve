// Portable CUDA 13 greedy sampling: exact max, minimum token-id ties.
// Ownerless rows never read logits; malformed active shards are rejected.
#if PHASE==0
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math.h>
#include <stdint.h>
#include <limits.h>
#if SCORE_HALF
using score_t=__half;
#else
using score_t=__nv_bfloat16;
#endif
extern "C" __global__ void greedy_partial(const score_t* logits,const int* metadata,float* scores,int* ids,int* finite
#if HAS_GRAMMAR
 , const uint32_t* token_mask
#endif
 ) {
 const int row=blockIdx.x,tile=blockIdx.y,lane=threadIdx.x&31,warp=threadIdx.x>>5;
 const int begin=tile*TILE,end=min(begin+TILE,WIDTH);
 float best=-INFINITY;int best_id=INT_MAX,good=1;
 if(row<metadata[0]) for(int j=begin+threadIdx.x;j<end;j+=blockDim.x){
  float v=float(logits[(size_t)row*STRIDE+j]);good &= isfinite(v);
#if HAS_GRAMMAR
  const int64_t token=VOCAB_START+j;
  if(!((token_mask[(size_t)row*MASK_WORDS+token/32]>>(token%32))&1u))continue;
#endif
  if(v>best || (v==best && j<best_id)){best=v;best_id=j;}
 }
 for(int d=16;d;d>>=1){
  float v=__shfl_down_sync(0xffffffff,best,d);int id=__shfl_down_sync(0xffffffff,best_id,d);
  int ok=__shfl_down_sync(0xffffffff,good,d);good &=ok;
  if(v>best || (v==best && id<best_id)){best=v;best_id=id;}
 }
 __shared__ float s[8];__shared__ int ix[8],f[8];
 if(lane==0){s[warp]=best;ix[warp]=best_id;f[warp]=good;}
 __syncthreads();
 if(warp==0){
  best=lane<8?s[lane]:-INFINITY;best_id=lane<8?ix[lane]:INT_MAX;good=lane<8?f[lane]:1;
  for(int d=16;d;d>>=1){float v=__shfl_down_sync(0xffffffff,best,d);int id=__shfl_down_sync(0xffffffff,best_id,d);good &=__shfl_down_sync(0xffffffff,good,d);if(v>best||(v==best&&id<best_id)){best=v;best_id=id;}}
  if(!lane){int offset=row*TILES+tile;scores[offset]=best;ids[offset]=best_id;finite[offset]=good;}
 }
}
#elif PHASE==1
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math.h>
#include <stdint.h>
#include <limits.h>
#if SCORE_HALF
using score_t=__half;
#else
using score_t=__nv_bfloat16;
#endif
extern "C" __global__ void greedy_merge(const float* scores,const int* ids,const int* finite,const int* metadata,score_t* output_scores,int64_t* output_ids){
 const int row=blockIdx.x,lane=threadIdx.x&31,warp=threadIdx.x>>5;
 float best=-INFINITY;int best_id=INT_MAX,good=1;
 for(int j=threadIdx.x;j<TILES;j+=blockDim.x){int pos=row*TILES+j;float v=scores[pos];int id=ids[pos];good &=finite[pos];if(v>best||(v==best&&id<best_id)){best=v;best_id=id;}}
 for(int d=16;d;d>>=1){float v=__shfl_down_sync(0xffffffff,best,d);int id=__shfl_down_sync(0xffffffff,best_id,d);good &=__shfl_down_sync(0xffffffff,good,d);if(v>best||(v==best&&id<best_id)){best=v;best_id=id;}}
 __shared__ float s[8];__shared__ int ix[8],f[8];
 if(!lane){s[warp]=best;ix[warp]=best_id;f[warp]=good;}
 __syncthreads();
 if(warp==0){
  best=lane<8?s[lane]:-INFINITY;best_id=lane<8?ix[lane]:INT_MAX;good=lane<8?f[lane]:1;
  for(int d=16;d;d>>=1){float v=__shfl_down_sync(0xffffffff,best,d);int id=__shfl_down_sync(0xffffffff,best_id,d);good &=__shfl_down_sync(0xffffffff,good,d);if(v>best||(v==best&&id<best_id)){best=v;best_id=id;}}
  if(!lane){s[0]=best;ix[0]=best_id;f[0]=good;}
 }
 __syncthreads();
 for(int j=threadIdx.x;j<CANDIDATES;j+=blockDim.x){
  bool valid=j==0 && row<metadata[0] && f[0] && WIDTH>0;
#if HAS_GRAMMAR
  valid &= isfinite(s[0]) && ix[0]!=INT_MAX;
#endif
  bool invalid=j==0 && row<metadata[0] && !f[0] && WIDTH>0;
  output_scores[row*CANDIDATES+j]=score_t(valid?s[0]:NAN);
  output_ids[row*CANDIDATES+j]=valid?(int64_t)ix[0]+VOCAB_START:(invalid?-2:-1);
 }
}
#elif PHASE==2
// Original Cubic greedy candidate ordering and rejecting predicates.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <limits.h>
#include <math.h>
#if SCORE_HALF
using score_t=__half;
#else
using score_t=__nv_bfloat16;
#endif
extern "C" __global__ void greedy_root(const score_t* scores,const int64_t* ids,const float* temperatures,const float* top_ps,const int64_t* top_ks,const int64_t* seeds,const int64_t* ordinals,const int* metadata,int64_t* tokens,bool* valid){
 int row=blockIdx.x,lane=threadIdx.x&31,warp=threadIdx.x>>5,active=metadata[0];
 float best=-INFINITY;int64_t best_id=LLONG_MAX;int good=1,real_count=0;
 for(int j=threadIdx.x;j<TP*K;j+=blockDim.x){
  int shard=j/K,col=j%K;int pos=(shard*ROWS+row)*K+col;float s=float(scores[pos]);int64_t id=ids[pos];bool real=isfinite(s)&&id>=0&&id<VOCAB_SIZE;bool sentinel=isnan(s)&&id==-1;
  good &=real||sentinel;real_count+=real;
  if(real&&(s>best||(s==best&&id<best_id))){best=s;best_id=id;}
 }
 for(int d=16;d;d>>=1){float s=__shfl_down_sync(0xffffffff,best,d);int64_t id=__shfl_down_sync(0xffffffff,best_id,d);good &=__shfl_down_sync(0xffffffff,good,d);real_count+=__shfl_down_sync(0xffffffff,real_count,d);if(s>best||(s==best&&id<best_id)){best=s;best_id=id;}}
 __shared__ float ws[4];__shared__ int64_t wi[4];__shared__ int wg[4],wc[4];
 if(!lane){ws[warp]=best;wi[warp]=best_id;wg[warp]=good;wc[warp]=real_count;}__syncthreads();
 if(!warp){
  best=lane<4?ws[lane]:-INFINITY;best_id=lane<4?wi[lane]:LLONG_MAX;good=lane<4?wg[lane]:1;real_count=lane<4?wc[lane]:0;
  for(int d=16;d;d>>=1){float s=__shfl_down_sync(0xffffffff,best,d);int64_t id=__shfl_down_sync(0xffffffff,best_id,d);good &=__shfl_down_sync(0xffffffff,good,d);real_count+=__shfl_down_sync(0xffffffff,real_count,d);if(s>best||(s==best&&id<best_id)){best=s;best_id=id;}}
  if(!lane){
   float t=temperatures[row],p=top_ps[row];int64_t k=top_ks[row];
   bool ok=good&&real_count>=1&&t==0.f&&isfinite(p)&&p>0.f&&p<=1.f&&k>=1&&k<=K&&seeds[row]>=0&&ordinals[row]>=0;
   if(row<active){tokens[row]=ok?best_id:-1;valid[row]=ok;}
   else{tokens[row]=(metadata[1]&&row==active)?-2:-1;valid[row]=false;}
  }
 }
}
#else
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>
extern "C" __global__ void sampling_final_guard(const int64_t* tokens,const bool* root_valid,const int* metadata,bool* valid,
 const float* temperatures,const float* top_ps,const int64_t* top_ks,const int64_t* seeds,const int64_t* ordinals){
 int active=metadata[0],good=active>0&&active<=CAPACITY;
 for(int i=threadIdx.x;i<CAPACITY;i+=blockDim.x){
  int64_t token=tokens[i];
  if(i<active){
   good &=token>=0&&token<VOCAB_SIZE;if constexpr(ROOT)good &=root_valid[i];
   const float t=temperatures[i],p=top_ps[i];
   good &=isfinite(t)&&t>=0.f&&t<=2.f&&isfinite(p)&&p>0.f&&p<=1.f;
   good &=top_ks[i]>=1&&top_ks[i]<=K&&seeds[i]>=0&&ordinals[i]>=0;
  }
  else good &=(i==active)?(token==-1||token==-2):(token==-1);
 }

 __shared__ int warp_good[4];int lane=threadIdx.x&31,warp=threadIdx.x>>5;
 for(int d=16;d;d>>=1)good &= __shfl_down_sync(0xffffffff,good,d);
 if(!lane)warp_good[warp]=good;__syncthreads();
 if(warp==0){good=lane<4?warp_good[lane]:1;for(int d=16;d;d>>=1)good &= __shfl_down_sync(0xffffffff,good,d);if(!lane)valid[0]=good;}
}


#endif
