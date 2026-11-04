"""Singleton transport preserves bytes and the existing multi-rank boundary."""

from datetime import timedelta
from unittest.mock import Mock, patch

import pytest
import torch
import torch.distributed as dist

from cubic_serve.ops.cuda.collective import all_gather_single, all_reduce, barrier, broadcast


def test_singleton_transport_preserves_storage_and_validates_the_root():
    group = object()
    source = torch.tensor([1, -7, 19], dtype=torch.int64)
    output = torch.full((1, 3), -99, dtype=source.dtype)
    with (patch.object(dist, 'get_world_size', return_value=1),
          patch.object(dist, 'get_rank', return_value=7),
          patch.object(dist, 'get_backend', return_value='gloo'),
          patch.object(dist, 'all_reduce', side_effect=AssertionError('peer transport')),
          patch.object(dist, 'all_gather_single', side_effect=AssertionError('peer transport')),
          patch.object(dist, 'broadcast', side_effect=AssertionError('peer transport')),
          patch.object(dist, 'barrier', side_effect=AssertionError('peer transport'))):
        for op in (dist.ReduceOp.SUM, dist.ReduceOp.MIN, dist.ReduceOp.MAX):
            all_reduce(source, op=op, group=group)
        all_gather_single(output, source, group=group)
        broadcast(source, src=7, group=group)
        broadcast(source, group_src=0, group=group)
        barrier(group=group)
        assert source.tolist() == [1, -7, 19]
        assert torch.equal(output[0], source)
        for root in ({'src': 8}, {'group_src': 1}, {}, {'src': 7, 'group_src': 0}):
            with pytest.raises(ValueError, match='root'):
                broadcast(source, group=group, **root)
        with pytest.raises(ValueError, match='matching contiguous'):
            all_gather_single(output.to(torch.float32), source, group=group)
        with pytest.raises(ValueError, match='matching contiguous'):
            all_gather_single(output[:, :2], source, group=group)


def test_singleton_nccl_barrier_retains_host_completion():
    stream = Mock()
    with (patch.object(dist, 'get_world_size', return_value=1),
          patch.object(dist, 'get_backend', return_value='nccl'),
          patch('torch.cuda.current_stream', return_value=stream)):
        barrier()
    stream.synchronize.assert_called_once_with()


def test_multirank_transport_delegates_unchanged():
    group, source, output = object(), object(), object()
    with (patch.object(dist, 'get_world_size', return_value=2),
          patch.object(dist, 'all_reduce') as reduce,
          patch.object(dist, 'all_gather_single') as gather,
          patch.object(dist, 'broadcast') as send,
          patch.object(dist, 'barrier') as wait):
        all_reduce(source, op=dist.ReduceOp.MIN, group=group)
        all_gather_single(output, source, group=group)
        broadcast(source, group_src=1, group=group)
        barrier(group=group)
    reduce.assert_called_once_with(source, op=dist.ReduceOp.MIN, group=group)
    gather.assert_called_once_with(output, source, group=group)
    send.assert_called_once_with(source, src=None, group=group, group_src=1)
    wait.assert_called_once_with(group=group)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required')
def test_singleton_sampling_transport_is_capture_safe_without_a_communicator(tmp_path):
    assert not dist.is_initialized()
    device = torch.device('cuda', 0)
    torch.cuda.set_device(device)
    source = torch.tensor([0., -0., float('nan'), 11.], device=device)
    output = torch.full_like(source, 97.)
    dist.init_process_group('nccl', init_method=f'file://{tmp_path / "group"}',
                            rank=0, world_size=1, timeout=timedelta(seconds=60))
    graph = None
    try:
        group = dist.new_group([0])
        with (patch.object(dist, 'all_reduce', side_effect=AssertionError('peer transport')),
              patch.object(dist, 'all_gather_single', side_effect=AssertionError('peer transport')),
              patch.object(dist, 'broadcast', side_effect=AssertionError('peer transport')),
              patch.object(dist, 'barrier', side_effect=AssertionError('peer transport'))):
            all_gather_single(output, source, group=group)
            barrier()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                all_reduce(source, op=dist.ReduceOp.MAX)
                all_gather_single(output, source, group=group)
                broadcast(output, group_src=0)
            for value in (source.clone(), torch.tensor([3., -0., -5., float('nan')], device=device)):
                source.copy_(value)
                output.fill_(71.)
                graph.replay()
                torch.cuda.synchronize()
                assert torch.equal(output.view(torch.int32), value.view(torch.int32))
    finally:
        if graph is not None:
            graph.reset()
        dist.destroy_process_group()
