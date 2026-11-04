// SM80 native-operand/F32-accumulator contract matching the established small-M
// and GDN primitives. Packing and quantization math belong to the producer.
#pragma once
#include <cuda_runtime.h>

__device__ __forceinline__ void cubic_mma16816(float (&c)[4],
    const unsigned (&a)[4],const unsigned (&b)[2]) {
#if CUBIC_MMA_BF16
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
#else
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
#endif
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
        :"=f"(c[0]),"=f"(c[1]),"=f"(c[2]),"=f"(c[3])
        :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]),
         "f"(c[0]),"f"(c[1]),"f"(c[2]),"f"(c[3]));
}
__device__ __forceinline__ void cubic_load_mma_a(unsigned (&a)[4],
    const void* pointer,int stride,int lane) {
    auto* p=static_cast<const unsigned short*>(pointer)+(lane%16)*stride+(lane/16)*8;
    const unsigned address=unsigned(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];"
        :"=r"(a[0]),"=r"(a[1]),"=r"(a[2]),"=r"(a[3]):"r"(address));
}
__device__ __forceinline__ void cubic_load_mma_b_col(unsigned (&b)[2],
    const void* pointer,int stride,int lane) {
    auto* p=static_cast<const unsigned short*>(pointer)+(lane%8)*stride+((lane/8)&1)*8;
    const unsigned address=unsigned(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1},[%2];"
        :"=r"(b[0]),"=r"(b[1]):"r"(address));
}
__device__ __forceinline__ void cubic_load_mma_b_row(unsigned (&b)[2],
    const void* pointer,int stride,int lane) {
    auto* p=static_cast<const unsigned short*>(pointer)+(lane%16)*stride;
    const unsigned address=unsigned(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];"
        :"=r"(b[0]),"=r"(b[1]):"r"(address));
}
