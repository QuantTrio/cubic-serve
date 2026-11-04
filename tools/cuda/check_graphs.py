#!/usr/bin/env python3
"""Probe conditional CUDA graphs without loading model weights.

Run in the serving environment:
    python tools/cuda/check_graphs.py --devices 0,1

Compare transports, if the collective case fails:
    python tools/cuda/check_graphs.py --devices 0,1 --disable-p2p

Test Cubic's native device-side loop-control boundary (requires CUDA 13.0 nvcc):
    python tools/cuda/check_graphs.py --devices 0,1 --native-control

This diagnostic does not change serving settings or qualification caches.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import timedelta
import faulthandler
import multiprocessing as mp
from multiprocessing.connection import wait
import os
from pathlib import Path
import signal
import tempfile
import time


def _worker(rank, size, rendezvous, channel, native_control):
    faulthandler.enable()
    import torch
    import torch.distributed as dist
    from cuda.bindings import runtime as rt

    def check(result):
        if int(result[0]):
            raise RuntimeError(str(result[0]))
        return result[1:]

    def describe(graph):
        _, count = check(rt.cudaGraphGetNodes(graph))
        nodes, _ = check(rt.cudaGraphGetNodes(graph, count))
        counts = Counter()
        for node in nodes:
            kind = check(rt.cudaGraphNodeGetType(node))[0]
            counts[kind.name] += 1
            if kind == rt.cudaGraphNodeType.cudaGraphNodeTypeGraph:
                counts.update(describe(check(rt.cudaGraphChildGraphNodeGetGraph(node))[0]))
        return counts

    initialized = False
    try:
        torch.cuda.set_device(rank)
        channel.send(f'rank={rank} GPU={torch.cuda.get_device_name()} '
                     f'driver={check(rt.cudaDriverGetVersion())[0]} torch={torch.__version__}')
        x = torch.zeros(8, dtype=torch.int32, device=f'cuda:{rank}')
        stream = torch.cuda.Stream()
        for collective in (False, True):
            if collective:
                dist.init_process_group('nccl', init_method=rendezvous,
                                        rank=rank, world_size=size, timeout=timedelta(seconds=30))
                initialized = True
                dist.all_reduce(x)
                channel.send(f'rank={rank} NCCL={torch.cuda.nccl.version()}')
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                x.add_(1)
                if collective:
                    dist.all_reduce(x)
            stream.synchronize()
            captured = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(captured, stream=stream):
                x.add_(1)
                if collective:
                    dist.all_reduce(x)
            captured.instantiate()
            body = check(rt.cudaGraphCreate(0))[0]
            check(rt.cudaGraphAddChildGraphNode(body, [], 0, captured.raw_cuda_graph()))
            try:
                for while_loop in (False, True):
                    label = ('collective' if collective else 'kernel') + (' WHILE' if while_loop else ' IF')
                    root = check(rt.cudaGraphCreate(0))[0]
                    executable = None
                    try:
                        # WHILE starts false: test instantiation without an unbounded loop.
                        handle = check(rt.cudaGraphConditionalHandleCreate(root, 0 if while_loop else 1, 1))[0]
                        params = rt.cudaGraphNodeParams()
                        params.type = rt.cudaGraphNodeType.cudaGraphNodeTypeConditional
                        params.conditional.handle = handle
                        params.conditional.type = (rt.cudaGraphConditionalNodeType.cudaGraphCondTypeWhile
                                                   if while_loop else rt.cudaGraphConditionalNodeType.cudaGraphCondTypeIf)
                        params.conditional.size = 1
                        check(rt.cudaGraphAddNode(root, [], None, 0, params))
                        check(rt.cudaGraphAddChildGraphNode(params.conditional.phGraph_out[0], [], 0, body))
                        channel.send(f'rank={rank} {label} instantiating nodes={dict(describe(body))}')
                        executable = check(rt.cudaGraphInstantiateWithFlags(root, 0))[0]
                        for _ in range(3):
                            x.fill_(rank)
                            stream.wait_stream(torch.cuda.current_stream())
                            check(rt.cudaGraphLaunch(executable, stream.cuda_stream))
                            stream.synchronize()
                            expected = (rank if while_loop else
                                        size * (size + 1) // 2 if collective else rank + 1)
                            if x.cpu().tolist() != [expected] * 8:
                                raise RuntimeError(f'{label} output differs from integer oracle')
                        channel.send(f'rank={rank} {label} PASS')
                    finally:
                        if executable is not None:
                            check(rt.cudaGraphExecDestroy(executable))
                        check(rt.cudaGraphDestroy(root))
            finally:
                check(rt.cudaGraphDestroy(body))
                captured.reset()
        if native_control:
            from cubic_serve.ops.cuda.native import CudaScalar
            from cubic_serve.ops.cuda.native_compiler import compile_native

            source = b'''#include <cuda_runtime.h>
extern "C" __global__ void advance(int* counter, unsigned long long handle) {
    if (threadIdx.x == 0) {
        counter[0] += 1;
        cudaGraphSetConditional(handle, counter[0] < 3);
    }
}
'''
            channel.send(f'rank={rank} preparing native control kernel')
            module, entry = compile_native((('entry.cu', source),), x.device, 'graph_control_probe')
            counter = torch.zeros(1, dtype=torch.int32, device=x.device)
            for collective, nested in ((False, False), (False, True), (True, False), (True, True)):
                label = 'native collective WHILE' if collective else 'native kernel WHILE'
                if nested:
                    label += '/IF'
                root = check(rt.cudaGraphCreate(0))[0]
                executable = None
                captured = torch.cuda.CUDAGraph(keep_graph=True)
                try:
                    handle = check(rt.cudaGraphConditionalHandleCreate(root, 1, 1))[0]
                    branch = check(rt.cudaGraphConditionalHandleCreate(root, 1, 1))[0] if nested else None
                    call = module.bind(entry, (counter, CudaScalar('i64', int(handle))),
                                       operator_id='diagnostic.graph_control',
                                       grid=(1, 1, 1), block=(32, 1, 1))
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.graph(captured, stream=stream):
                        x.add_(1)
                        if collective:
                            dist.all_reduce(x)
                        call.launch()
                    # Match retained model/control children before parent composition.
                    channel.send(f'rank={rank} {label} instantiating retained child')
                    captured.instantiate()
                    stream.synchronize()
                    params = rt.cudaGraphNodeParams()
                    params.type = rt.cudaGraphNodeType.cudaGraphNodeTypeConditional
                    params.conditional.handle = handle
                    params.conditional.type = rt.cudaGraphConditionalNodeType.cudaGraphCondTypeWhile
                    params.conditional.size = 1
                    check(rt.cudaGraphAddNode(root, [], None, 0, params))
                    body = params.conditional.phGraph_out[0]
                    if nested:
                        choice = rt.cudaGraphNodeParams()
                        choice.type = rt.cudaGraphNodeType.cudaGraphNodeTypeConditional
                        choice.conditional.handle = branch
                        choice.conditional.type = rt.cudaGraphConditionalNodeType.cudaGraphCondTypeIf
                        choice.conditional.size = 1
                        check(rt.cudaGraphAddNode(body, [], None, 0, choice))
                        body = choice.conditional.phGraph_out[0]
                    check(rt.cudaGraphAddChildGraphNode(body, [], 0, captured.raw_cuda_graph()))
                    channel.send(f'rank={rank} {label} instantiating nodes={dict(describe(captured.raw_cuda_graph()))}')
                    executable = check(rt.cudaGraphInstantiateWithFlags(root, 0))[0]
                    for _ in range(3):
                        counter.zero_()
                        x.fill_(rank)
                        stream.wait_stream(torch.cuda.current_stream())
                        check(rt.cudaGraphLaunch(executable, stream.cuda_stream))
                        stream.synchronize()
                        expected = size * (size + 1) // 2
                        for _ in range(2):
                            expected = size * (expected + 1)
                        expected = expected if collective else rank + 3
                        if counter.item() != 3 or x.cpu().tolist() != [expected] * 8:
                            raise RuntimeError(f'{label} output differs from integer oracle')
                    channel.send(f'rank={rank} {label} PASS')
                finally:
                    if executable is not None:
                        check(rt.cudaGraphExecDestroy(executable))
                    check(rt.cudaGraphDestroy(root))
                    captured.reset()
        if initialized:
            dist.destroy_process_group()
        channel.send(f'rank={rank} COMPLETE')
    except BaseException as error:
        channel.send(f'rank={rank} ERROR {type(error).__name__}: {error}')
        raise
    finally:
        channel.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--devices', default='0,1')
    parser.add_argument('--disable-p2p', action='store_true')
    parser.add_argument('--native-control', action='store_true')
    args = parser.parse_args()
    devices = args.devices.split(',')
    if not devices or any(not d.isdigit() for d in devices) or len(set(devices)) != len(devices):
        parser.error('--devices must list unique GPU ordinals')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.devices
    os.environ.setdefault('NCCL_GRAPH_MIXING_SUPPORT', '0')
    os.environ.setdefault('NCCL_GRAPH_STREAM_ORDERING', '0')
    if args.disable_p2p:
        os.environ['NCCL_P2P_DISABLE'] = '1'
    context = mp.get_context('spawn')
    processes, channels = [], []
    with tempfile.TemporaryDirectory(prefix='cubic-cuda-graph-probe-') as directory:
        rendezvous = Path(directory, 'nccl.store').as_uri()
        try:
            for rank in range(len(devices)):
                reader, writer = context.Pipe(duplex=False)
                process = context.Process(target=_worker, args=(rank, len(devices), rendezvous, writer,
                                                               args.native_control))
                process.start()
                writer.close()
                processes.append(process)
                channels.append(reader)
            # A diagnostic work budget, never a serving or audit timeout.
            deadline = time.monotonic() + 60
            while any(p.is_alive() for p in processes):
                for channel in wait(channels, timeout=0.1):
                    try:
                        print(channel.recv(), flush=True)
                    except EOFError:
                        channels.remove(channel)
                        channel.close()
                if any(p.exitcode not in (None, 0) for p in processes):
                    break
                if time.monotonic() >= deadline:
                    print('Probe exceeded its 60-second diagnostic budget.', flush=True)
                    return 1
            for channel in list(channels):
                while channel.poll():
                    try:
                        print(channel.recv(), flush=True)
                    except EOFError:
                        break
            failed = False
            for rank, process in enumerate(processes):
                code = process.exitcode
                if code != 0:
                    detail = signal.Signals(-code).name if code is not None and code < 0 else code
                    print(f'rank={rank} probe did not complete: {detail}', flush=True)
                    failed = True
            return int(failed)
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                process.join(3)
                if process.is_alive():
                    process.kill()
                    process.join()
            for channel in channels:
                channel.close()


if __name__ == '__main__':
    raise SystemExit(main())
