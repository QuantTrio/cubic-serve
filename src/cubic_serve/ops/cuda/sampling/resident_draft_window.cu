#include <cuda_runtime.h>

// The journal header and stop word follow SharedResidentJournal. Each round
// owns a distinct record until the topology-wide lease has been consumed.
extern "C" __global__ void window_step(
 int* meta, const int* ctx, const long long* result, const long long* ordinals,
 const int* metadata, const bool* ready, long long* remaining,
 const int* eos_offsets, const long long* eos_values,
 unsigned long long* clocks, unsigned long long* policy,
 const unsigned long long* host_address, int capacity, int depth,
 int context_stride, int metadata_stride, int axes_column, int vocabulary,
 int root, unsigned long long handle, int phase) {
 const int index=meta[2];
 const int payload=capacity*(2*depth+8);
 const int stride=payload+6;
 long long* host=reinterpret_cast<long long*>(host_address[0]);
 long long* record=host+64+index*stride;
 if(phase<3) {
  if(threadIdx.x==0) {
   unsigned long long tick;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(tick));
   clocks[index*4+phase]=tick;
   if(phase==0 && index==0)policy[1]=tick;
  }
  return;
 }
 if(phase==6) {
  if(threadIdx.x==0)cudaGraphSetConditional(handle,meta[4]==0);
  return;
 }
 if(phase==5) {
  if(threadIdx.x==0) {
   const bool complete=meta[4]!=0;
   if(meta[4]>1)meta[3]=1;
   for(int k=0;k<4;++k)record[payload+k]=clocks[index*4+k];
   record[payload+4]=complete;record[payload+5]=meta[3];
   meta[2]=index+1;cudaGraphSetConditional(handle,!complete);
   __threadfence_system();
   *reinterpret_cast<volatile int*>(host)=index+1;
   __threadfence_system();
  }
  return;
 }
 __shared__ int stop, pending_decision;
 if(threadIdx.x==0){stop=meta[4];pending_decision=0;}
 __syncthreads();
 for(int r=threadIdx.x;r<meta[0];r+=blockDim.x) {
  if(phase==4) {
   if(!ready[r])atomicMax(&stop,1);
   continue;
  }
  const long long* value=result+r*(depth+3);
  const long long raw_accepted=value[0],bonus=value[1];
  const int accepted=(raw_accepted>=0 && raw_accepted<=depth)?int(raw_accepted):0;
  bool good=value[depth+2] && raw_accepted>=0 && raw_accepted<=depth
            && bonus>=0 && bonus<vocabulary && remaining[3*r]>=accepted+1;
  good=good && ordinals[r]>=0 && ordinals[r]<=0x7fffffffffffffffLL-accepted;
  long long* out=record+r*(2*depth+8);
  for(int j=0;j<depth;++j)good=good && value[j+2]>=0 && value[j+2]<vocabulary;
  out[0]=good?accepted+1:0;
  for(int j=0;j<=depth;++j) {
   const long long token=j<accepted?ctx[1+r*context_stride+2+j]:bonus;
   out[j+1]=j<=accepted?token:-1;
   if(j<=accepted) {
    good=good && token>=0 && token<vocabulary;
    for(int e=eos_offsets[r];e<eos_offsets[r+1];++e)
     if(token==eos_values[e])atomicMax(&stop,1);
   }
  }
  for(int j=0;j<depth;++j)out[depth+2+j]=value[j+2];
  out[2*depth+2]=static_cast<long long>(ctx[1+r*context_stride])+accepted+1;
  out[2*depth+3]=good?ordinals[r]+accepted:0;
  const int first=r*(depth+1)*metadata_stride;
  for(int axis=0;axis<3;++axis) {
   const long long x0=metadata[first+axes_column+axis];
   const long long x1=metadata[first+metadata_stride+axes_column+axis];
   out[2*depth+4+axis]=x0+(accepted+1)*(x1-x0);
  }
  out[2*depth+7]=good;
  if(!good) {atomicMax(meta+3,1);atomicMax(&stop,2);}
  else {
   remaining[3*r]-=accepted+1;
   if(remaining[3*r]<=0)atomicMax(&stop,1);
   if(remaining[3*r+1]>=0) {
    remaining[3*r+1]=max(0LL,remaining[3*r+1]-accepted-1);
    remaining[3*r+2]=max(0LL,remaining[3*r+2]-1);
    if(remaining[3*r+1] || remaining[3*r+2])atomicExch(&pending_decision,1);
   } else atomicExch(&pending_decision,1);
  }
 }
 __threadfence_system();
 __syncthreads();
 if(threadIdx.x==0) {
  if(phase==4) {
   meta[4]=max(stop,meta[3]?2:0);
   // The caller performs the topology MAX before setting the condition.
   return;
  }
  unsigned long long tick;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(tick));
  clocks[index*4+3]=tick;
  const bool revoked=root && *reinterpret_cast<volatile int*>(host+1);
  const bool expired=root && policy[0] && tick-policy[1]>=policy[0];
  meta[4]=max(stop,int(!pending_decision || revoked || expired || index+1>=meta[1]));
 }
}
