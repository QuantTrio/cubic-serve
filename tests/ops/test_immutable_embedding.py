"""Explicit distributed original-row and cross-M audit of immutable lookup."""

import os
import socket
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def test_host_registration_query_does_not_create_another_device_context():
    from cubic_serve.ops.cuda.embedding.immutable import driver, host_registration_supported

    with patch("torch.cuda.device") as context, patch.object(
        driver, "cuDeviceGetAttribute", return_value=(driver.CUresult.CUDA_SUCCESS, 1),
    ) as query:
        assert host_registration_supported(5)
        query.assert_called_once_with(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED, 5,
        )
        context.assert_not_called()


def test_mapped_host_capability_uses_the_execution_device_and_propagates_query_errors():
    from cubic_serve.ops.cuda.embedding.immutable import (
        driver,
        mapped_host_memory_supported,
    )

    attribute = driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY
    with patch("torch.cuda.device", return_value=nullcontext()), patch.object(
        driver, "cuDeviceGetAttribute", return_value=(driver.CUresult.CUDA_SUCCESS, 1),
    ) as query:
        assert mapped_host_memory_supported(torch.device("cuda:5"))
        query.assert_called_once_with(attribute, 5)
        assert not mapped_host_memory_supported(torch.device("cpu"))
    with patch("torch.cuda.device", return_value=nullcontext()), patch.object(
        driver, "cuDeviceGetAttribute", return_value=(driver.CUresult.CUDA_SUCCESS, 0),
    ), patch("torch.cuda.current_device", return_value=2):
        assert not mapped_host_memory_supported(torch.device("cuda"))
    with patch("torch.cuda.device", return_value=nullcontext()), patch.object(
        driver, "cuDeviceGetAttribute", side_effect=RuntimeError("query failure"),
    ), unittest.TestCase().assertRaisesRegex(RuntimeError, "query failure"):
        mapped_host_memory_supported(torch.device("cuda:5"))


def _worker(rank, devices, port):
    from datetime import timedelta

    from cubic_serve.executor.runner import configure_reference_math
    from cubic_serve.ops.cuda.embedding.immutable import ImmutableEmbeddingTable
    from cubic_serve.ops.cuda.linear.quantized import PackedLinearOperators
    from cubic_serve.ops.eager.quantization import embedding_matrix
    from cubic_serve.ops.strategies import OperatorConfig
    from tests.ops.test_cubic_wna8 import matrix

    configure_reference_math()
    torch.set_num_threads(1)
    device = torch.device("cuda", devices[rank])
    torch.cuda.set_device(device)
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=len(devices),
        init_method=f"tcp://127.0.0.1:{port}",
        timeout=timedelta(seconds=180),
        device_id=device,
    )
    try:
        for dtype in (torch.bfloat16, torch.float16):
            for bits in (None, 2, 3, 4, 5, 6, 7, 8):
                torch.manual_seed(918 + rank)
                n, k = 64, 257
                span = SimpleNamespace(start=rank * n, stop=(rank + 1) * n)
                operators = PackedLinearOperators(OperatorConfig())
                if bits is None:
                    weight = torch.randn(n, k, device=device, dtype=dtype)
                    weight[0, :2] = torch.tensor([0.0, -0.0], device=device, dtype=dtype)
                else:
                    weight = matrix(
                        bits, metadata="e5m9-scale-curve2-tensor-fp16", k=k, n=n, group_out=2
                    )
                table = ImmutableEmbeddingTable.create(
                    weight, span, dist.group.WORLD, device, dtype, operators=operators
                )
                assert table is not None
                assert table is ImmutableEmbeddingTable.create(
                    weight, span, dist.group.WORLD, device, dtype, operators=operators
                )
                corpus = (torch.arange(7, device=device) * 67) % (n * len(devices))
                previous = None
                for m in (7, 8, 32):
                    ids = torch.zeros(m, device=device, dtype=torch.int64)
                    # Permute the same logical rows between physical capacities.
                    rows = torch.randperm(m, generator=torch.Generator().manual_seed(m))[:7].to(
                        device
                    )
                    ids[rows] = corpus
                    output = torch.empty(m, k, device=device, dtype=dtype)
                    binding = table.bind(ids, output)
                    binding.launch(ids, weight, output)
                    owned = (ids >= span.start) & (ids < span.stop)
                    local = torch.where(owned, ids - span.start, 0)
                    if bits is None:
                        ref = weight[local].cpu()
                    else:
                        ref = embedding_matrix(local.cpu(), weight, dtype=dtype)
                    ref = (ref * owned.cpu()[:, None]).to(device).float()
                    shards = [torch.empty_like(ref) for _ in devices]
                    dist.all_gather(shards, ref)
                    ref = torch.zeros_like(ref)
                    for shard in shards:
                        ref = ref + shard
                    ref = ref.to(dtype)
                    if not torch.equal(output.view(torch.int16), ref.view(torch.int16)):
                        raise AssertionError(
                            (
                                rank,
                                str(dtype),
                                bits,
                                m,
                                ids.tolist(),
                                output[:, :3].tolist(),
                                ref[:, :3].tolist(),
                                float((output.float() - ref.float()).abs().max()),
                            )
                        )
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        binding.launch(ids, weight, output)
                    output.fill_(float("nan"))
                    graph.replay()
                    torch.cuda.synchronize()
                    assert torch.equal(output.view(torch.int16), ref.view(torch.int16))
                    current = output[rows].clone()
                    if previous is not None:
                        assert torch.equal(current.view(torch.int16), previous.view(torch.int16))
                    previous = current
                    del graph, binding
                dist.barrier()
                table.close()
                del table
        if rank == 0:
            print(
                "PASS: plain BF16/FP16 and Cubic W2-W8; TP, original rows, "
                "capture, cross-M and row permutations",
                flush=True,
            )
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(
    os.environ.get("CUBIC_TEST_EMBEDDING_DEVICES"), "explicit device list required"
)
class ImmutableEmbeddingTests(unittest.TestCase):
    def test_distributed_original_rows(self):
        devices = tuple(map(int, os.environ["CUBIC_TEST_EMBEDDING_DEVICES"].split(",")))
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        mp.spawn(_worker, args=(devices, port), nprocs=len(devices), join=True)
