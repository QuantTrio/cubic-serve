#include <cuda_runtime.h>
__device__ __forceinline__ void apply_page_program_row(
 int row, const int* lengths, const int* segments, const int* owners,
 const int* activity, const int* program, int step,
 int* sources, int* destinations, int* tables, int* offsets,
 int capacity, int table_pages, int page_size) {
 if(activity[row]<=0)return;
 const int first=owners[row];
 const int width=segments[first];
 const int* plan=program+(step*capacity+row)*4;
 sources[row]=plan[0];destinations[row]=plan[1];
 offsets[row]=(lengths[row]-1)%page_size;
 const int* first_plan=program+(step*capacity+first)*4;
 const int* last_plan=program+(step*capacity+first+width-1)*4;
 const int first_position=lengths[first]-1;
 for(int page=first_plan[2];page<=last_plan[2];++page) {
   const int token=first+max(0,page*page_size-first_position);
   tables[row*table_pages+page]=program[(step*capacity+token)*4+1];
 }
}

extern "C" __global__ void window_step(int* meta, const long long* tokens,
 const bool* valid, const int* eos_offsets, const long long* eos_values,
 long long* output, unsigned long long* clocks, const unsigned long long* host_address,
 const int* page_schedule, int* selected_updates, unsigned long long* policy, const int* logical_activity, const int* prompt_steps, const int* output_rows, bool* update_activity,
 const int* page_program, const int* owner_starts, const int* lengths, const int* segments,
 int* page_source, int* page_destination, int* page_table, int* token_offsets,
 int page_size, int table_pages,
 int capacity, int vocabulary, int root, unsigned long long handle, unsigned long long page_handle,
 unsigned long long segment_handle, unsigned long long decode_handle, int phase) {
 int index=meta[2];
 if (phase == 3) {
   __shared__ int pages;
   if(threadIdx.x==0)pages=0;
   __syncthreads();
   for(int row=threadIdx.x;row<capacity;row+=blockDim.x) {
     bool update=logical_activity[row]>0 && page_schedule[(index*capacity+row)*3]==row;
     update_activity[row]=update;
     if(update)atomicExch(&pages,1);
     for(int column=0;column<3;column++)selected_updates[row*3+column]=update?page_schedule[(index*capacity+row)*3+column]:-1;
   }
   __syncthreads();
   if(threadIdx.x==0) {
     cudaGraphSetConditional(page_handle,pages);
     if(segment_handle) {
       cudaGraphSetConditional(segment_handle,policy[2]!=0);
       cudaGraphSetConditional(decode_handle,policy[2]==0);
     }
   }
   return;
 }
 long long* host=reinterpret_cast<long long*>(host_address[0]);
 long long* row_output=host+64+index*(capacity+3);
 if (phase == 4) {
  if(threadIdx.x==0) {
   bool complete=meta[4]!=0;
   if(meta[4]==2)meta[3]=1;
   meta[2]=index+1;cudaGraphSetConditional(handle,!complete);
   row_output[capacity]=clocks[index*2+1]-clocks[index*2];
   row_output[capacity+1]=complete;row_output[capacity+2]=meta[3];
   __threadfence_system();
   *reinterpret_cast<volatile int*>(host)=index+1;
   __threadfence_system();
  }
  return;
 }
 if (phase < 2) {
   if(phase==0 && policy[2]) {
     for(int row=threadIdx.x;row<capacity;row+=blockDim.x) {
       apply_page_program_row(row,lengths,segments,owner_starts,logical_activity,
         page_program,index,page_source,page_destination,page_table,token_offsets,
         capacity,table_pages,page_size);
     }
     __syncthreads();
   }
   if (threadIdx.x==0) { unsigned long long tick; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(tick)); clocks[index*2+phase]=tick; if(phase==0 && index==0)policy[1]=tick; }
   return;
 }
 __shared__ int stop;
 if (threadIdx.x==0) stop=0;
 __syncthreads();
 for (int row=threadIdx.x;row<meta[0];row+=blockDim.x) {
   int output_row=output_rows[row];
   if(output_row<0 || output_row>=capacity || logical_activity[output_row]<=0) {
     atomicExch(meta+3,1);atomicExch(&stop,1);continue;
   }
   long long token=tokens[output_row];output[index*capacity+row]=token;row_output[row]=token;
   if (token<0 || token>=vocabulary) { atomicExch(meta+3,1);atomicExch(&stop,1); }
   // Mid-prompt logits have no public sampled-token identity and cannot end
   // a request. The final prompt input produces its first public token.
   if (index + 1 >= prompt_steps[output_row]) {
     for (int j=eos_offsets[row];j<eos_offsets[row+1];j++) if (token==eos_values[j]) atomicExch(&stop,1);
   }
 }
 if (threadIdx.x==0 && !*valid) {atomicExch(meta+3,1);atomicExch(&stop,1);}
 __threadfence_system();
 __syncthreads();
 if (threadIdx.x==0) {
   unsigned long long tick; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(tick));
   bool root_stop=root && ((policy[0] && tick-policy[1]>=policy[0]) || *reinterpret_cast<volatile int*>(host+1));
   meta[4]=meta[3]?2:int(stop || root_stop || index+1>=meta[1]);
 }
}
