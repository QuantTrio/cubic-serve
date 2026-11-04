"""Lossless column completion using the existing private mapped-host protocol."""

from pathlib import Path

import torch
import torch.distributed as dist

from cubic_serve.ops.cuda.collective.host_sum import HostSum
from cubic_serve.ops.cuda.native import CudaScalar
from cubic_serve.ops.cuda.native_compiler import compile_native


class ColumnGather:
    def __init__(self, value, output, group, *, arena=None, ranking_guard=None,
                 destination_rank=None, source_rank=None):
        import cubic_serve.ops.cuda.collective.host_sum as host

        if (
            value.ndim != 2
            or output.ndim != 2
            or not value.is_contiguous()
            or not output.is_contiguous()
            or min(value.shape) <= 0
            or value.dtype != output.dtype
            or value.device != output.device
            or value.dtype not in (torch.float16, torch.bfloat16)
            or value.device.type != "cuda"
            or output.shape != (value.shape[0], value.shape[1] * dist.get_world_size(group))
            or value.untyped_storage().data_ptr() == output.untyped_storage().data_ptr()
        ):
            raise ValueError(
                "column completion requires disjoint CUDA half matrices and complete rank columns"
            )
        if arena is not None and (
            arena.group is not group
            or arena.partitioned
            or arena.dtype != value.dtype
            or arena.device != value.device
            or arena.shape != tuple(value.shape)
        ):
            raise ValueError("column completion arena differs from the sealed group and geometry")
        ranks = dist.get_world_size(group)
        for owner in (destination_rank, source_rank):
            if owner is not None and (type(owner) is not int or not 0 <= owner < ranks):
                raise ValueError("column transport owner is outside its process group")
        if destination_rank is not None and source_rank is not None:
            raise ValueError("column transport cannot simultaneously gather and broadcast")
        guard_args = ()
        if ranking_guard is not None:
            lengths, active, threshold = ranking_guard
            if (type(threshold) is not int or threshold <= 0 or any(
                t.device != value.device or t.dtype != torch.int32
                or t.shape != (value.shape[0],) or not t.is_contiguous()
                for t in (lengths, active)
            )):
                raise ValueError("collective guard requires shared row ownership and length metadata")
            guard_args = (lengths, active, CudaScalar("i32", threshold))
        self.arena = a = HostSum.create(value, group) if arena is None else arena
        if a is None:
            raise RuntimeError(
                "column completion requires admitted same-host mapping before capture"
            )
        helpers = (
            Path(host.__file__).with_suffix(".cu").read_text().split("#if !HOST_SUM_PARTITIONED")[0]
        )
        source = (
            f"#define HOST_SUM_FP16 {int(value.dtype == torch.float16)}\n"
            f"#define COLUMN_GUARDED {int(ranking_guard is not None)}\n"
            f"#define COLUMN_DESTINATION {destination_rank if destination_rank is not None else -1}\n"
            f"#define COLUMN_SOURCE {source_rank if source_rank is not None else -1}\n"
            + helpers
            + Path(__file__).with_suffix(".cu").read_text()
        )
        self.module, entry = compile_native(
            (("entry.cu", source.encode()),), value.device, "collective"
        )
        self.call = self.module.bind(
            entry,
            (
                value,
                output,
                a.pointer,
                *(CudaScalar("i32", n) for n in (*value.shape, a.stride, a.ranks, a.rank)),
                a.epochs,
                *guard_args,
            ),
            operator_id="collective.column_gather",
            grid=(a.blocks, 1, 1),
            block=(a.threads, 1, 1),
        )

    def launch(self):
        self.call.launch()
