"""Original-width Cubic group metadata; expansion is confined to kernel tiles."""

from __future__ import annotations

import struct
from functools import partial

import torch
from torch import Tensor


def _assemble(
    raw: list[Tensor],
    descriptors: list[list[bytes]],
    variant: int,
    n: int,
    groups: int,
    device: torch.device,
) -> Tensor:
    experts = len(raw)
    desc_offset = 24 + 8 * experts
    records = b"".join(d for expert in descriptors for d in expert)
    raw_offset = (desc_offset + len(records) + 15) // 16 * 16
    header = struct.pack("<6I", variant, n, groups, experts, raw_offset, desc_offset)
    start = 0
    for ds in descriptors:
        header += struct.pack("<2I", start, len(ds))
        start += len(ds)
    header += records
    header += bytes(raw_offset - len(header))
    prefix = torch.tensor(list(header), dtype=torch.uint8, device=device)
    return torch.cat((prefix, *(r.to(device).reshape(-1) for r in raw)))


def prepare_metadata(groups, device: torch.device) -> Tensor:
    base = groups[0][0]
    enc = base.encoding
    n = sum(g[0].shape[0] for g in groups)
    count = base.shape[1] // min(enc.group_in, base.shape[1])
    variant = {
        "int8-scale-int4-ab-global-fp32": 1,
        "e5m9-scale-curve2-tensor-fp16": 2,
        "float32-scale-float16-ab": 0,
    }[enc.metadata]
    raw, descriptors = [], []
    perm = torch.tensor([i + 8 * j for i in range(8) for j in range(8)], device=device)
    for expert in range(len(groups[0])):
        arrays, records, end = [], [], 0
        for g in groups:
            matrix = g[expert]
            p = matrix.payload
            rows = matrix.shape[0]

            def bytes_of(name, size, p=p, rows=rows):
                return p[name].to(device).contiguous().view(torch.uint8).reshape(rows, count, size)

            if variant == 1:
                data = torch.cat((bytes_of("weight_scale", 1), bytes_of("weight_ab", 1)), -1)
            elif variant == 2:
                data = bytes_of("weight_metadata", 2)
            else:
                data = torch.cat(
                    (bytes_of("weight_scale", 4), bytes_of("weight_a", 2), bytes_of("weight_b", 2)),
                    -1,
                )
            arrays.append(data.transpose(0, 1).contiguous())
            for part, size in enumerate(matrix.partitions or (rows,)):
                end += size
                record = struct.pack("<I", end)
                if variant == 1:
                    record += struct.pack(
                        "<3f",
                        *(
                            float(p[name].reshape(-1)[part])
                            for name in (
                                "weight_a_global",
                                "weight_b_global",
                                "weight_scale_global",
                            )
                        ),
                    )
                    record += bytes(16)
                elif variant == 2:
                    record += bytes(12)
                    for name in ("weight_curve_a", "weight_curve_b"):
                        record += bytes(
                            p[name]
                            .reshape(-1, 4)[part]
                            .contiguous()
                            .view(torch.uint8)
                            .cpu()
                            .tolist()
                        )
                else:
                    record += bytes(28)
                records.append(record)
        data = torch.cat(arrays, dim=1)
        raw.append(data.reshape(-1, 64, data.shape[-1])[:, perm].contiguous())
        descriptors.append(records)
    return _assemble(raw, descriptors, variant, n, count, device)


def prepare_fixture(metadata: Tensor, *, variant: int = 0) -> Tensor:
    """Cost fixture uses the plain original-width scale/AB representation."""
    e, planes, g, n = metadata.shape
    if variant:
        if variant == 1:
            row = torch.tensor([1, 0], dtype=torch.uint8, device=metadata.device)
            data = row.expand(e, g, n, 2).contiguous()
            descriptor = struct.pack("<I3f", n, 0.0, 0.0, 1.0) + bytes(16)
        else:
            data = torch.full((e, g, n), 0x1E00, device=metadata.device, dtype=torch.int16).view(
                torch.uint8
            )
            descriptor = (
                struct.pack("<I", n)
                + bytes(12)
                + struct.pack("<8e", 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0)
            )
        return _assemble(
            list(data), [[descriptor] for _ in range(e)], variant, n, g, metadata.device
        )
    if planes == 2:
        ab = metadata[:, 0].contiguous().view(torch.uint8).reshape(e, g, n, 4)
        scale = metadata[:, 1].contiguous().view(torch.uint8).reshape(e, g, n, 4)
    else:
        a = metadata[:, 0].half().contiguous().view(torch.uint8).reshape(e, g, n, 2)
        b = metadata[:, 1].half().contiguous().view(torch.uint8).reshape(e, g, n, 2)
        ab = torch.cat((a, b), -1)
        scale = metadata[:, 2].contiguous().view(torch.uint8).reshape(e, g, n, 4)
    data = torch.cat((scale, ab), -1)
    return _assemble(
        list(data), [[struct.pack("<I", n) + bytes(28)] for _ in range(e)], 0, n, g, metadata.device
    )


FIXTURE_PACKERS = (
    prepare_fixture,
    partial(prepare_fixture, variant=1),
    partial(prepare_fixture, variant=2),
)


def metadata_variant(name: str) -> int:
    return {
        "float32-scale-float16-ab": 0,
        "int8-scale-int4-ab-global-fp32": 1,
        "e5m9-scale-curve2-tensor-fp16": 2,
    }[name]


CUDA_METADATA = r"""
struct PackedCubicMetadata {
  template<int planes>
  __device__ static void fetch(int4* target, int stage, int dest, int index,
                              const float* blob, int expert, int row_stride, int global_stride,
                              int n_offset) {
    const auto bytes = reinterpret_cast<const unsigned char*>(blob);
    const auto header = reinterpret_cast<const unsigned int*>(blob);
    const int variant=__ldg(header+0), n=__ldg(header+1), groups=__ldg(header+2);
    const int stride=variant ? 2:8;
    const auto raw=bytes+__ldg(header+4)+(long long)expert*n*groups*stride;
    const int desc_start=__ldg(header+6+2*expert), desc_count=__ldg(header+7+2*expert);
    const auto desc=bytes+__ldg(header+5)+desc_start*32;
    // A copy-vector owner is not a decode-work owner. Spread group arithmetic
    // over the CTA; small groups may span multiple scale rows in one K tile.
    const int base=index-(stage==row_stride ? threadIdx.x :
                         (threadIdx.x/row_stride)*global_stride+threadIdx.x%row_stride);
    for(int j=threadIdx.x;j<stage*8;j+=blockDim.x) {
      const int at=(base+j/(row_stride*8)*global_stride)*8+j%(row_stride*8);
      const int pos=j%(row_stride*8);
      const int column=n_offset+(pos/64)*64+(pos%8)*8+(pos%64)/8;
      const unsigned char* d=desc;
      for(int part=0;part<desc_count-1 &&
          column>=__ldg(reinterpret_cast<const unsigned int*>(d));++part) d+=32;
      const auto globals=reinterpret_cast<const float*>(d+4);
      float a,b,s;
      if(variant==1) {
        const unsigned int packed=__ldg(reinterpret_cast<const unsigned short*>(raw)+at);
        const int scale=(int)(signed char)(packed&255);
        const unsigned int ab=packed>>8;
        const int ac=(int)(ab&15)-((ab&8)?16:0);
        const int bc=(int)(ab>>4)-((ab&128)?16:0);
        a=__half2float(__float2half_rn(__fadd_rn(1.f,__fmul_rn((float)ac,__ldg(globals+0)))));
        b=__half2float(__float2half_rn(__fmul_rn((float)bc,__ldg(globals+1))));
        s=__fmul_rn((float)scale,__ldg(globals+2));
      } else if(variant==2) {
        const unsigned int packed=reinterpret_cast<const unsigned short*>(raw)[at];
        const auto curves=reinterpret_cast<const __half*>(d+16);
        const int curve=packed>>14;
        a=__half2float(curves[curve]);b=__half2float(curves[4+curve]);
        s=__half2float(__ushort_as_half((packed&0x3fff)<<1));
      } else {
        const auto entry=raw+at*8;
        s=*reinterpret_cast<const float*>(entry);
        const auto ab=reinterpret_cast<const __half*>(entry+4);
        a=__half2float(ab[0]);b=__half2float(ab[1]);
      }
      auto out=reinterpret_cast<float*>(target)+j;
      if constexpr(planes==2) {
        const unsigned int ab=__half_as_ushort(__float2half_rn(a)) |
                            ((unsigned int)__half_as_ushort(__float2half_rn(b))<<16);
        out[0]=__uint_as_float(ab);out[stage*8]=s;
      } else {
        out[0]=a;out[stage*8]=b;out[stage*16]=s;
      }
    }
  }
};
"""
