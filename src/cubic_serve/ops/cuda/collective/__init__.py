"""Synchronous collectives over already validated, fixed-address buffers.

A one-rank group has no peer transport. Its gather is a stream-ordered local
copy; reductions and broadcasts preserve the local value. Multi-rank groups
retain PyTorch's transport and reduction contracts.
"""

import torch
import torch.distributed as dist

from cubic_serve.ops.cuda.collective.fixed_rank_sum import FixedRankBF16SumPrimitive


def all_reduce(tensor, op=dist.ReduceOp.SUM, group=None):
    if dist.get_world_size(group) == 1 and op in (
        dist.ReduceOp.SUM, dist.ReduceOp.MIN, dist.ReduceOp.MAX,
    ):
        return
    dist.all_reduce(tensor, op=op, group=group)


def all_gather_single(output, tensor, group=None):
    if dist.get_world_size(group) == 1:
        if (output.dtype != tensor.dtype or output.device != tensor.device
                or output.numel() != tensor.numel()
                or not output.is_contiguous() or not tensor.is_contiguous()):
            raise ValueError("singleton gather requires matching contiguous buffers")
        output.copy_(tensor.view_as(output))
        return
    dist.all_gather_single(output, tensor, group=group)


def broadcast(tensor, src=None, group=None, *, group_src=None):
    if dist.get_world_size(group) == 1:
        if (src is None) == (group_src is None):
            raise ValueError("broadcast requires exactly one root")
        if group_src is not None and group_src != 0:
            raise ValueError("broadcast root is outside the singleton group")
        if src is not None and src != dist.get_rank():
            raise ValueError("broadcast root is outside the singleton group")
        return
    dist.broadcast(tensor, src=src, group=group, group_src=group_src)


def barrier(group=None):
    if dist.get_world_size(group) == 1:
        # NCCL barrier is host blocking and orders current-device work. Keep
        # that completion boundary without creating a GPU communicator.
        if dist.get_backend(group) == "nccl":
            torch.cuda.current_stream().synchronize()
        return
    dist.barrier(group=group)


__all__ = ("FixedRankBF16SumPrimitive", "all_reduce", "all_gather_single",
           "broadcast", "barrier")
