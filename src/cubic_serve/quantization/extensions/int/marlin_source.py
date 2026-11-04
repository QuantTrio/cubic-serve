"""Pure format-owned Marlin source recipes, usable by CPU-only wheel builds."""
from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from cubic_serve.quantization.common.marlin.parameters import MarlinParameters

def _a8_source(
    p: MarlinParameters, dtype: str, bits: int, n: int, k: int, group: int,
    *, fp32_output: bool = False, dense: bool = True, fixed_partitions: int = 0,
    inline_zero: bool = False,
) -> str:
    """Use the shared Marlin integer MMA and original packed W2-W8 transport."""
    fragment = ("kU4" if bits <= 4 else "kU8") if inline_zero else (
        "kU4B8" if bits <= 4 else "kS8"
    )
    partition_define = (
        f"#define CUBIC_MARLIN_FIXED_PARTITIONS {int(bool(fixed_partitions))}\n"
        if fixed_partitions else ""
    )
    return f"""#define MARLIN_NAMESPACE_NAME integer_a8
#define CUBIC_MARLIN_DENSE_LINEAR {int(dense)}
#define CUBIC_MARLIN_STORAGE_BITS {bits}
#define CUBIC_MARLIN_UNWEIGHTED 1
#define CUBIC_MARLIN_INT_A8_FLOAT_SCALE 1
{'#define CUBIC_MARLIN_INT_A8_REGISTER_ZERO 1' if inline_zero else ''}
{partition_define}
{'#define CUBIC_MARLIN_FP32_OUTPUT 1' if fp32_output else ''}
#include "libtorch_stable/moe/marlin_moe_wna16/kernel.h"
#include "marlin_template.h"
namespace integer_a8 {{
template __global__ void Marlin<vllm::kS8.id(),vllm::{fragment}.id(),vllm::{dtype}.id(),
 vllm::kFloat16.id(),{p.marlin_warps * 32},{p.marlin_block_m // 16},
 {p.marlin_block_n // 16},{p.marlin_block_k // 16},false,{p.marlin_stages},
 {group // 16},false>(MARLIN_KERNEL_PARAMS);
}}"""

def _source(
    p: MarlinParameters,
    dtype: str,
    bits: int,
    group: int,
    quantum: int,
    scale_dtype: str,
    dense: bool,
    fixed_partitions: int = 0,
    *,
    n: int,
    k: int,
) -> str:
    # Format/tile source is independent of matrix dimensions and original GI.
    fragment = "kU4B8" if bits <= 4 else "kU8B128"
    halves = 2 if group % quantum else 1
    stored_planes = 1 if scale_dtype == "F16" else 2
    planes = stored_planes * halves
    rounded = {
        "F16": "__half2float(__float2half_rn(v))",
        "BF16": "__bfloat162float(__float2bfloat16_rn(v))",
        "F32": "v",
    }[scale_dtype]
    partition_define = (
        f"#define CUBIC_MARLIN_FIXED_PARTITIONS {int(bool(fixed_partitions))}\n" if fixed_partitions else ""
    )
    # Two metadata layouts, independent of the actual GI/N/K values. When a
    # stored group is exactly one pipeline quantum, its quotient is already q;
    # retain invariant division only for the unequal-group layout.
    same_group = group == quantum
    group_division = "" if same_group else """
   const unsigned gm=__float_as_uint(__ldg(blob+2)), gs=__float_as_uint(__ldg(blob+3));
   auto div_g=[&](unsigned value) { return (__umulhi(value,gm)+value)>>gs; };"""
    group_count = (f"(full_k+{quantum-1})/{quantum}" if same_group else
                   "1+div_g(static_cast<unsigned>(full_k-1))")
    group_index = "q" if same_group else "div_g(static_cast<unsigned>(k+h*8))"
    m8 = str(p.marlin_block_m == 8).lower()
    fragment_define = "#define CUBIC_MARLIN_FP16_GROUP_FRAGMENT 1\n" if scale_dtype == "F16" else ""
    return f"""{fragment_define}#define CUBIC_MARLIN_STORAGE_BITS {bits}
#define MARLIN_NAMESPACE_NAME integer
#define CUBIC_MARLIN_DENSE_LINEAR {int(dense)}
#define CUBIC_MARLIN_RUNTIME_METADATA 1
#define CUBIC_MARLIN_UNWEIGHTED 1
{partition_define}#include "libtorch_stable/moe/marlin_moe_wna16/kernel.h"
namespace integer {{
struct AffineDecoder {{
 static constexpr int metadata_planes={planes};
 template<int planes> __device__ static void fetch(int4* target,int stage,int dest,int index,
     const float* blob,int expert,int row_stride,int global_stride,int n_offset,int n,int full_k) {{
   const unsigned nm=__float_as_uint(__ldg(blob+0)), ns=__float_as_uint(__ldg(blob+1));
   auto div_n=[&](unsigned value) {{ return (__umulhi(value,nm)+value)>>ns; }};
   {group_division}
   const int groups={group_count};
   const float* raw=blob+4+(long long)expert*{stored_planes}*groups*n;
   const int base=index-(stage==row_stride ? threadIdx.x :
                         (threadIdx.x/row_stride)*global_stride+threadIdx.x%row_stride);
   for(int j=threadIdx.x;j<stage*8;j+=blockDim.x) {{
     const int at=(base+j/(row_stride*8)*global_stride)*8+j%(row_stride*8);
     const int q=div_n(static_cast<unsigned>(at));
     const int k=q*{quantum}, col=at-q*n;
     auto out=reinterpret_cast<float*>(target)+j;
     #pragma unroll
     for(int h=0;h<{halves};++h) {{
       const int g={group_index};
       #pragma unroll
       for(int p=0;p<{stored_planes};++p)
         out[(h*{stored_planes}+p)*stage*8]=raw[(p*groups+g)*n+col];
     }}
   }}
 }}
 template<vllm::ScalarTypeId type> __device__ static void apply(
    typename MarlinScalarType<type>::FragB& fragment,
    const float (&metadata)[{planes}][8],int column) {{
   using D=MarlinScalarType<type>;
#ifdef CUBIC_MARLIN_FP16_GROUP_FRAGMENT
   auto input=reinterpret_cast<half2*>(&fragment);
   auto output=reinterpret_cast<typename D::scalar_t2*>(&fragment);
   #pragma unroll
   for(int i=0;i<2;++i) {{
     const int h={halves}==1 ? 0 : i;
     half2 pair=*reinterpret_cast<const half2*>(&metadata[h][column]);
     half2 v=__hmul2_rn(__hsub2_rn(input[i],__high2half2(pair)),__low2half2(pair));
     if constexpr(type==vllm::kFloat16.id()) output[i]=v;
     else {{
       float2 f=__half22float2(v);
       output[i]=__floats2bfloat162_rn(f.x,f.y);
     }}
   }}
#else
   auto values=reinterpret_cast<typename D::scalar_t*>(&fragment);
   #pragma unroll
   for(int i=0;i<4;++i) {{
     float q=D::num2float(values[i]);
     const int h={halves}==1 ? 0 : i/2;
     float v=__fmul_rn(__fsub_rn(q,metadata[h*2+1][column]),metadata[h*2][column]);
     values[i]=D::float2num({rounded});
   }}
#endif
 }}
}};
}}
#define CUBIC_MARLIN_GROUP_DECODER AffineDecoder
#include "marlin_template.h"
namespace integer {{
template __global__ void Marlin<vllm::{dtype}.id(),vllm::{fragment}.id(),
 vllm::{dtype}.id(),vllm::kFloat16.id(),{p.marlin_warps * 32},{max(1, p.marlin_block_m // 16)},
 {p.marlin_block_n // 16},{p.marlin_block_k // 16},{m8},{p.marlin_stages},{quantum // 16},false>
 (MARLIN_KERNEL_PARAMS);
}}"""
