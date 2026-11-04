extern "C" __global__ void ordered_host_gather(
    const Element* input, Element* output,
    const unsigned long long* arena_address,
    int rows, int width, int stride, int ranks, int rank, unsigned* epochs
#if COLUMN_GUARDED
    , const int* lengths, const int* active, int threshold
#endif
    ) {
#if COLUMN_GUARDED
    // Every rank holds the same sealed logical row/length metadata. Skipping
    // leaves epochs untouched on ALL peers; no host read or runtime recapture.
    __shared__ int needed;
    if (threadIdx.x == 0) needed = 0;
    __syncthreads();
    for (int row=threadIdx.x;row<rows;row+=blockDim.x)
        if (active[row] && lengths[row]>=threshold) atomicOr(&needed,1);
    __syncthreads();
    if (!needed) return;
#endif
    using T=Element;
    char* arena=reinterpret_cast<char*>(arena_address[0]);
    const int tid=threadIdx.x,bid=blockIdx.x,blocks=gridDim.x;
    volatile unsigned* flags=reinterpret_cast<volatile unsigned*>(arena+size_t(stride)*ranks*2);
    __shared__ unsigned token;
    if(tid==0) token=epochs[bid]+1;
    __syncthreads();
    int slot=token&1;
    T* mine=reinterpret_cast<T*>(arena+size_t(stride)*(slot*ranks+rank));
    int row_vectors=(width+7)/8, vectors=rows*row_vectors;
    if(COLUMN_SOURCE<0 || rank==COLUMN_SOURCE)
    for(int v=bid*blockDim.x+tid;v<vectors;v+=blocks*blockDim.x){
        int row=v/row_vectors,col=(v%row_vectors)*8,at=row*width+col;
        if(width%8==0) *reinterpret_cast<uint4*>(mine+at)=*reinterpret_cast<const uint4*>(input+at);
        else for(int j=0;j<8 && col+j<width;++j) mine[at+j]=input[at+j];
    }
    __threadfence_system();__syncthreads();
    if(tid==0){flags[((slot*ranks+rank)*blocks+bid)*16]=token;__threadfence_system();}
    __syncthreads();
    for(int peer=tid;peer<ranks;peer+=blockDim.x)
        while(flags[((slot*ranks+peer)*blocks+bid)*16]!=token) __nanosleep(64);
    __syncthreads();__threadfence_system();
    if(COLUMN_DESTINATION<0 || rank==COLUMN_DESTINATION)
    for(int v=bid*blockDim.x+tid;v<vectors;v+=blocks*blockDim.x){
        int row=v/row_vectors,col=(v%row_vectors)*8,at=row*width+col;
        for(int peer=0;peer<ranks;++peer){
            if(COLUMN_SOURCE>=0 && peer!=COLUMN_SOURCE) continue;
            const T* source=peer==rank?input:reinterpret_cast<const T*>(arena+size_t(stride)*(slot*ranks+peer));
            int dst=(row*ranks+peer)*width+col;
            if(width%8==0) *reinterpret_cast<uint4*>(output+dst)=*reinterpret_cast<const uint4*>(source+at);
            else for(int j=0;j<8 && col+j<width;++j) output[dst+j]=source[at+j];
        }
    }
    __syncthreads();if(tid==0)epochs[bid]=token;
}
