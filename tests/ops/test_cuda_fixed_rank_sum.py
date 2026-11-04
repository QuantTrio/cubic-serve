from __future__ import annotations

import json
import os
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist

from cubic_serve.ops.cuda.collective import FixedRankBF16SumPrimitive
from cubic_serve.ops.cuda.moe.route_completion import RowPartitionedRouteCompletion
from cubic_serve.ops.cuda.normalization import ZeroCenteredRMSNormPrimitive


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_request_time_rank_sum_never_calibrates_a_new_shape():
    template = torch.empty((2360, 1152), dtype=torch.bfloat16, device="cuda:0")
    with (
        patch("torch.distributed.get_world_size", return_value=4),
        patch("cubic_serve.ops.cuda.collective.fixed_rank_sum._calibrate_transport",
              side_effect=AssertionError("request-time CUDA Graph capture")) as calibrate,
    ):
        primitive = FixedRankBF16SumPrimitive(
            template, dist.group.WORLD, calibrate_transport=False)
    calibrate.assert_not_called()
    assert primitive._host is None
    assert primitive._shape == template.shape


@pytest.mark.parametrize("local_denied", (False, True))
def test_prefault_peer_decline_skips_writes_and_restores_only_changed_affinity(local_denied):
    from types import SimpleNamespace

    from cubic_serve.ops.cuda.collective.host_sum import _prefault_rank_payload

    owner = SimpleNamespace(mapping=bytearray(8192), device=torch.device("cpu"))

    def decline(ready, **kwargs):
        ready.zero_()

    with (
        patch("os.sched_getaffinity", return_value={3, 7}),
        patch("os.sched_setaffinity",
              side_effect=PermissionError() if local_denied else None) as affinity,
        patch("cuda.bindings.runtime.cudaDeviceGetPCIBusId", return_value=(0, b"0000:01:00.0\0")),
        patch("pathlib.Path.read_text", return_value="0-3"),
        patch("torch.distributed.all_reduce", side_effect=decline),
        patch("torch.distributed.barrier") as barrier,
        patch("ctypes.memset") as write,
    ):
        assert not _prefault_rank_payload(owner, object())
    write.assert_not_called()
    barrier.assert_not_called()
    assert affinity.call_count == (1 if local_denied else 2)
    assert affinity.call_args.args == (0, {3} if local_denied else {3, 7})


@pytest.mark.parametrize("ranks", (2, 3, 4, 8))
@pytest.mark.parametrize("granularity", (4096, 65536, 2097152))
@pytest.mark.parametrize("cross_node", (False, True))
def test_host_numa_plan_bounds_physical_bytes_and_preserves_plane_ownership(ranks, granularity, cross_node):
    import mmap
    from cubic_serve.ops.cuda.collective.host_sum import _HostArenaPlan

    peers = tuple((rank % 2 if cross_node else 0, granularity) for rank in range(ranks))
    for count in (31, 10240, 327680, 2621440):
        for partitioned in (False, True):
            blocks = 16
            stride = (count * 2 + mmap.PAGESIZE - 1) // mmap.PAGESIZE * mmap.PAGESIZE
            flags = (3 if partitioned else 2) * ranks * blocks * 64
            raw = (2 * ranks * stride + flags + mmap.PAGESIZE - 1) // mmap.PAGESIZE * mmap.PAGESIZE
            plan = _HostArenaPlan.from_capabilities(count, blocks, partitioned, peers)
            if plan is None:
                # Decline is precisely the small-arena or excessive-alignment case.
                required_stride = (count * 2 + granularity - 1) // granularity * granularity
                cross_bytes = 2 * ranks * required_stride + (flags + granularity - 1) // granularity * granularity
                assert raw < granularity or (cross_node and cross_bytes > 2 * raw)
                continue
            assert raw <= plan.size <= 2 * raw
            assert plan.stride >= count * 2
            assert sum(size for _, _, _, size in plan.allocations) == plan.size
            offset = 0
            for _, _, start, size in sorted(plan.allocations, key=lambda a: a[2]):
                assert start == offset
                assert start % granularity == size % granularity == 0
                offset += size
            assert offset == plan.size
            if cross_node:
                for plane in range(2):
                    for rank in range(ranks):
                        assert (rank, peers[rank][0], (plane * ranks + rank) * plan.stride,
                                plan.stride) in plan.allocations
            else:
                assert plan.stride == stride


def test_host_numa_missing_peer_capability_declines_before_allocation():
    from cubic_serve.ops.cuda.collective.host_sum import _HostArenaPlan
    assert _HostArenaPlan.from_capabilities(2621440, 16, False, ((0, 2097152), None)) is None


@pytest.mark.parametrize("unsupported,cross_host", ((False, False), (True, False), (False, True)))
def test_host_numa_capability_query_handles_no_numa_and_cross_host_without_allocating(monkeypatch, unsupported, cross_host):
    from cubic_serve.ops.cuda.collective import host_sum as module
    class Group:
        pass
    group = Group()
    monkeypatch.setattr(module.driver, "cuDeviceGet", lambda *a: (0, 0))
    def attribute(name, dev):
        return (0, -1 if name == module.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID
                else int(not unsupported))
    monkeypatch.setattr(module.driver, "cuDeviceGetAttribute", attribute)
    monkeypatch.setattr(module.driver, "cuMemGetAllocationGranularity", lambda *a: (0, 65536))
    monkeypatch.setattr(module.socket, "gethostname", lambda: "local")
    monkeypatch.setattr(module.dist, "get_world_size", lambda *a: 2)
    def gather(out, value, **kwargs):
        out[:] = [value, (*value[:2], "remote" if cross_host else "local")]
    monkeypatch.setattr(module.dist, "all_gather_object", gather)
    with patch.object(module.driver, "cuMemCreate", side_effect=AssertionError("query allocated")):
        result = module._host_numa_capabilities(torch.device("cuda", 0), group)
    assert result == ((None, None) if unsupported or cross_host else ((0, 65536), (0, 65536)))
    assert module._GROUP_HOST_NUMA[group]["same-host", torch.device("cuda", 0)] is not cross_host


def test_registered_host_backing_declines_full_tmpfs_before_faulting(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from cubic_serve.ops.cuda.collective import host_sum as module
    path = tmp_path / "arena"
    def create(**kwargs):
        return os.open(path, os.O_RDWR | os.O_CREAT), str(path)
    owner = SimpleNamespace(rank=0, ranks=2, mapping=None, registered=False, device=torch.device("cpu"))
    closed = []
    owner.close = lambda: closed.append(True)
    monkeypatch.setattr(module.tempfile, "mkstemp", create)
    monkeypatch.setattr(module.os, "fstatvfs", lambda fd: SimpleNamespace(f_bavail=0, f_frsize=4096))
    monkeypatch.setattr(module.dist, "broadcast_object_list", lambda *a, **kw: None)
    monkeypatch.setattr(module.dist, "all_reduce", lambda *a, **kw: None)
    monkeypatch.setattr(module, "_prefault_rank_payload", lambda *a: False)
    with (patch.object(module.os, "ftruncate") as truncate,
          patch.object(module.mmap, "mmap", side_effect=AssertionError("full tmpfs mapped"))):
        assert not module._registered_host_arena(owner, 4096, object())
    truncate.assert_not_called()
    assert closed == [True] and not path.exists()


@pytest.mark.parametrize("fail_at", (None, "cuMemCreate", "cuMemExportToShareableHandle", "cuMemMap", "cuMemSetAccess"))
def test_host_numa_arena_failure_releases_handles_mappings_and_descriptors(monkeypatch, fail_at):
    from types import SimpleNamespace
    from cubic_serve.ops.cuda.collective import host_sum as module

    opened, handles, mapped, reservations = [], set(), set(), set()
    calls = {}
    class Descriptor:
        def __init__(self, fd):
            self.fd = os.dup(fd)
            opened.append(self.fd)
        def detach(self):
            fd, self.fd = self.fd, None
            return fd

    def operation(name):
        def call(*args):
            calls[name] = calls.get(name, 0) + 1
            if name == fail_at and (name != "cuMemMap" or calls[name] == 2):
                return (module.driver.CUresult.CUDA_ERROR_NOT_PERMITTED,)
            if name == "cuMemCreate":
                handle = len(handles) + 100
                handles.add(handle)
                return 0, handle
            if name == "cuMemExportToShareableHandle":
                fd = os.open(os.devnull, os.O_RDONLY)
                opened.append(fd)
                return 0, fd
            if name == "cuMemAddressReserve":
                reservations.add(0x100000)
                return 0, 0x100000
            if name == "cuMemMap":
                mapped.add((args[0], args[1]))
            if name == "cuMemUnmap":
                mapped.remove((args[0], args[1]))
            if name == "cuMemAddressFree":
                reservations.remove(args[0])
            if name == "cuMemRelease":
                handles.remove(args[0])
            return (0,)
        return call

    monkeypatch.setattr(module.dist, "get_rank", lambda *a: 0)
    monkeypatch.setattr(module.dist, "get_world_size", lambda *a: 2)
    def gather(out, value, **kwargs):
        # This rank owns the common arena; the peer exports no allocations.
        out[:] = [value, {} if isinstance(value, dict) else None]
        if isinstance(value, dict):
            # Model the peer receiving its exported descriptors.
            for destinations in value.values():
                os.close(destinations.pop(1).detach())
    monkeypatch.setattr(module.dist, "all_gather_object", gather)
    monkeypatch.setattr(module.dist, "barrier", lambda **kw: None)
    monkeypatch.setattr(module, "DupFd", Descriptor)
    monkeypatch.setattr(module.ctypes, "memset", lambda *a: None)
    for name in ("cuMemCreate", "cuMemExportToShareableHandle", "cuMemMap", "cuMemSetAccess",
                 "cuMemAddressReserve", "cuMemUnmap", "cuMemAddressFree", "cuMemRelease"):
        monkeypatch.setattr(module.driver, name, operation(name))
    plan = module._HostArenaPlan(4096, 20480, 4096,
                                ((0, 0, 0, 8192), (0, 0, 8192, 8192), (0, 0, 16384, 4096)))
    if fail_at:
        with pytest.raises(module._ArenaDeclined):
            module._HostNumaArena.create(plan, SimpleNamespace(index=0), object())
    else:
        arena = module._HostNumaArena.create(plan, SimpleNamespace(index=0), object())
        arena.close()
        arena.close()
    assert not handles and not mapped and not reservations
    for fd in opened:
        with pytest.raises(OSError):
            os.fstat(fd)


def _host_numa_storage_worker(rank, init_method):
    import ctypes
    import time
    from cubic_serve.ops.cuda.collective import host_sum as module

    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    dist.init_process_group("nccl", init_method=init_method, rank=rank, world_size=2,
                            timeout=timedelta(seconds=60))
    group = dist.group.WORLD
    try:
        peers = module._host_numa_capabilities(torch.device("cuda", rank), group)
        # A small unit exercises both physical layouts when the actual topology permits it.
        shape = (128, 8192)
        generator = torch.Generator().manual_seed(291018)
        parts = torch.randn(2, *shape, generator=generator).to(torch.bfloat16)
        expected = (parts[0].float() + parts[1].float()).to(torch.bfloat16)
        source, output = parts[rank].cuda(rank), torch.empty(shape, device=rank, dtype=torch.bfloat16)
        supported = all(p is not None for p in peers)
        for scenario in ("normal", "peer-import-declines", "capability-declines", "normal-again"):
            start = time.monotonic()
            patcher = (
                patch.object(module.driver, "cuMemImportFromShareableHandle", return_value=(
                    module.driver.CUresult.CUDA_ERROR_NOT_PERMITTED,))
                if scenario == "peer-import-declines" and rank == 1 else
                patch.object(module, "_host_numa_capabilities", return_value=(peers[0], None))
                if scenario == "capability-declines" else
                patch.object(module, "_host_numa_capabilities", return_value=peers))
            with patcher:
                binding = module.HostSum.create(output, group)
            assert binding is not None
            if supported and scenario in ("normal", "normal-again"):
                assert isinstance(binding.mapping, module._HostNumaArena)
            else:
                assert binding.registered
            graph = torch.cuda.CUDAGraph()
            try:
                binding.bind(source, output)
                binding.launch(source, output)
                assert torch.equal(output.cpu().view(torch.uint8), expected.view(torch.uint8))
                with torch.cuda.graph(graph):
                    binding.launch(source, output)
                for _ in range(3):
                    graph.replay()
                    assert torch.equal(output.cpu().view(torch.uint8), expected.view(torch.uint8))
                # Each rank owns exactly its plane; peer bytes survive unchanged.
                # The arithmetic result above is independently formed on CPU.
                assert binding.epochs.min().item() > 0
                print(json.dumps(dict(rank=rank, scenario=scenario,
                    backing=binding.workload_resource_identity, seconds=time.monotonic()-start)), flush=True)
            finally:
                torch.cuda.synchronize()
                graph.reset()
                dist.barrier()
                binding.close()
                binding.close()
            # New construction must not retain an old arena's mapped address.
            assert binding.mapping is None
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(os.environ.get("CUBIC_TEST_QUANT_CUDA") != "1", reason="explicit CUDA audit")
def test_host_numa_storage_real_graph_and_peer_failure():
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    with tempfile.TemporaryDirectory() as tmp:
        torch.multiprocessing.spawn(_host_numa_storage_worker,
            args=("file://" + str(Path(tmp) / "init"),), nprocs=2, join=True)


def _automatic_transport_worker(rank: int, world: int, init_method: str) -> None:
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    dist.init_process_group("nccl", init_method=init_method, rank=rank, world_size=world,
                            timeout=timedelta(seconds=60))
    try:
        for dtype in (torch.float16, torch.bfloat16):
            # Shared logical rows across padded capacities and different TP
            # group sizes. Poison has no identity and must not enter real rows.
            generators = [torch.Generator().manual_seed(190 + r) for r in range(world)]
            inputs = [torch.randn((37, 1536), generator=g).to(dtype) for g in generators]
            expected = torch.zeros((37, 1536), dtype=torch.float32)
            for part in inputs:
                expected = expected + part.float()
            expected = expected.to(dtype)
            common = None
            for capacity in (37, 64, 128):
                source = torch.full((capacity, 1536), float("nan"), dtype=dtype, device=rank)
                source[:37].copy_(inputs[rank])
                output = torch.empty_like(source)
                primitive = FixedRankBF16SumPrimitive(output, dist.group.WORLD)
                for _ in range(2):
                    output.copy_(source)
                    primitive.launch(output)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                try:
                    with torch.cuda.graph(graph):
                        output.copy_(source)
                        primitive.launch(output)
                    for reverse in (False, True):
                        source[:37].copy_(inputs[rank].flip(0) if reverse else inputs[rank])
                        graph.replay()
                        actual = output[:37].cpu()
                        if reverse:
                            actual = actual.flip(0)
                        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
                        assert torch.isnan(output[37:]).all().item()
                        if common is not None:
                            assert torch.equal(actual.view(torch.uint8), common.view(torch.uint8))
                        common = actual
                finally:
                    graph.reset()
            # Explicit user overrides continue to own transport selection.
            from cubic_serve.ops.strategies import OperatorConfig, operator_strategy_scope

            for latency, partitioned in ((0, True), (1 << 20, False), (1 << 50, False)):
                forced = OperatorConfig.from_dict({
                    "version": 1,
                    "overrides": {"collective": {"exchange_latency_bytes": latency}},
                })
                with operator_strategy_scope(forced, device="", model=""):
                    primitive = FixedRankBF16SumPrimitive(output, dist.group.WORLD)
                    # Two ranks save no traffic with exchange + gather, even
                    # when the additional-startup penalty is explicitly zero.
                    assert primitive._use_partitioned_sum is (partitioned and world > 2)
                    assert primitive._host is None  # Includes explicit default-valued overrides.
                    source[:37].copy_(inputs[rank])
                    output.copy_(source)
                    primitive.launch(output)
                    assert torch.equal(output[:37].cpu().view(torch.uint8),
                                       expected.view(torch.uint8))
    finally:
        dist.destroy_process_group()


def _route_completion_worker(rank: int, world: int, init_method: str) -> None:
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    dist.init_process_group("nccl", init_method=init_method, rank=rank, world_size=world,
                            timeout=timedelta(seconds=60))
    graphs = []
    try:
        for dtype in (torch.float16, torch.bfloat16):
            for index_dtype in (torch.int32, torch.int64):
                generator = torch.Generator().manual_seed(810 + rank)
                data = torch.randn((8, 3, 257), generator=generator).to(dtype).cuda(rank)
                ids = (torch.arange(24).reshape(8, 3) % 8).to(index_dtype).cuda(rank)
                weights = torch.linspace(0.125, 0.75, 24).reshape(8, 3).to(dtype).cuda(rank)
                common = None
                for rows in (1, 7, 8):
                    source = data[:rows].clone()
                    other = -source
                    outputs = [torch.empty((rows, 257), dtype=dtype, device=rank) for _ in range(2)]
                    first = RowPartitionedRouteCompletion(
                        source, ids[:rows], weights[:rows], outputs[0], dist.group.WORLD,
                        first_expert=2, experts=4,
                    )
                    second = RowPartitionedRouteCompletion(
                        other, ids[:rows], weights[:rows], outputs[1], dist.group.WORLD,
                        first_expert=2, experts=4, workspace=first.workspace,
                    )
                    partition_snapshot = torch.empty_like(first.workspace.partition)

                    def launch(
                        first=first, partition_snapshot=partition_snapshot, second=second,
                    ) -> None:
                        first.launch()
                        partition_snapshot.copy_(first.workspace.partition)
                        second.launch()

                    launch()
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        launch()
                    graphs.append(graph)
                    graph.replay()
                    torch.cuda.synchronize()
                    for index, operand in enumerate((source, other)):
                        parts = [torch.empty_like(operand) for _ in range(world)]
                        dist.all_gather(parts, operand)
                        total = torch.zeros_like(operand, dtype=torch.float32, device="cpu")
                        for part in parts:
                            total = total + part.cpu().float()
                        total = total.to(dtype)
                        expected = torch.zeros((rows, 257), dtype=dtype)
                        for k in range(3):
                            owned = (ids[:rows, k].cpu() >= 2) & (ids[:rows, k].cpu() < 6)
                            product = (total[:, k].float()
                                       * weights[:rows, k, None].cpu().float()).to(dtype)
                            expected = expected + torch.where(owned[:, None], product, 0)
                        assert torch.equal(outputs[index].cpu().view(torch.uint8),
                                           expected.view(torch.uint8))
                        local = first.workspace.partition.shape[0]
                        padded = torch.zeros((world * local, 3, 257), dtype=dtype)
                        padded[:rows] = total
                        actual = partition_snapshot if index == 0 else first.workspace.partition
                        local_expected = padded[rank * local:(rank + 1) * local]
                        assert torch.equal(actual.cpu().view(torch.uint8),
                                           local_expected.view(torch.uint8))
                    if common is not None:
                        assert torch.equal(outputs[0][:1].view(torch.uint8), common)
                    common = outputs[0][:1].view(torch.uint8).clone()
                    # Reorder physical rows with one fixed corpus/reduction plan.
                    source.copy_(data[:rows].flip(0))
                    other.copy_(-source)
                    ids[:rows].copy_(ids[:rows].flip(0).clone())
                    weights[:rows].copy_(weights[:rows].flip(0).clone())
                    original = outputs[0].clone()
                    graph.replay()
                    torch.cuda.synchronize()
                    assert torch.equal(outputs[0].flip(0).contiguous().view(torch.uint8),
                                       original.view(torch.uint8))
                    ids[:rows].copy_(ids[:rows].flip(0).clone())
                    weights[:rows].copy_(weights[:rows].flip(0).clone())
    finally:
        torch.cuda.synchronize()
        for graph in graphs:
            graph.reset()
        dist.destroy_process_group()


def _canonical_sum(gathered: torch.Tensor) -> torch.Tensor:
    # Each in-place CUDA launch completes before the next operation on the
    # stream, fixing rank 0 -> N FP32 rounding boundaries.  Conversion rounds
    # to BF16 only after the last rank has been accumulated.
    total = torch.zeros_like(gathered[0], dtype=torch.float32)
    for rank in range(gathered.shape[0]):
        total.add_(gathered[rank].float())
    return total.to(torch.bfloat16)


def _fixed_rank_worker(rank: int, init_method: str, report_path: str) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=init_method,
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=30),
    )
    try:
        device = torch.device("cuda", rank)
        generator = torch.Generator(device=device).manual_seed(9103 + rank)
        source = torch.randn(
            (9, 5120), dtype=torch.bfloat16, device=device, generator=generator
        )
        # Identical per-rank partials at distinct physical rows are a
        # metamorphic oracle: their reduced values must remain bit-exact.
        source[1].copy_(source[0])

        nccl = source.clone()
        dist.all_reduce(nccl, op=dist.ReduceOp.SUM)
        nccl_identical = torch.equal(nccl[0], nccl[1])

        fixed = source.clone()
        primitive = FixedRankBF16SumPrimitive(fixed, dist.group.WORLD)
        primitive.launch(fixed)
        # Oracle ownership must not depend on which measured transport won.
        oracle_parts = torch.empty((4, source.numel()), dtype=source.dtype, device=device)
        dist.all_gather_single(oracle_parts.reshape(-1), source.reshape(-1))
        expected = _canonical_sum(oracle_parts).reshape_as(fixed)
        assert torch.equal(fixed, expected)
        assert torch.equal(fixed[0], fixed[1])

        residual = torch.randn(
            source.shape, dtype=torch.bfloat16, device=device, generator=generator
        )
        weight = torch.randn((5120,), dtype=torch.bfloat16, device=device, generator=generator)
        expected_summed = torch.empty_like(source)
        expected_norm = torch.empty_like(source)
        expected_summed.copy_(residual + expected)
        ZeroCenteredRMSNormPrimitive(
            expected_summed, weight, expected_norm, epsilon=1e-6
        ).launch(expected_summed, weight, expected_norm)
        fused_value = source.clone()
        fused_summed = torch.empty_like(source)
        fused_norm = torch.empty_like(source)
        primitive.launch_residual_rms_norm(
            fused_value,
            residual,
            fused_summed,
            weight,
            fused_norm,
            epsilon=1e-6,
        )
        assert torch.equal(fused_summed, expected_summed)
        assert torch.equal(fused_norm, expected_norm)
        # The fused performance ABI does not materialize its canonical TP sum.
        assert torch.equal(fused_value, source)

        permutation = torch.tensor(
            [5, 0, 8, 7, 2, 1, 6, 3, 4], dtype=torch.int64, device=device
        )
        padded = torch.empty((11, 5120), dtype=torch.bfloat16, device=device)
        displaced = padded[1:10]
        displaced.copy_(source.index_select(0, permutation))
        primitive.launch(displaced)
        displaced_expected = expected.index_select(0, permutation)
        assert torch.equal(displaced, displaced_expected)

        # Exercise the partitioned transport with a nonzero storage offset and
        # a physical-row permutation, independently of its internal workspace.
        partition_source = source.repeat((15, 1))[:128]
        partition_gathered = torch.empty(
            (4, partition_source.numel()),
            dtype=torch.bfloat16,
            device=device,
        )
        dist.all_gather_single(
            partition_gathered.view(-1), partition_source.reshape(-1)
        )
        partition_expected = _canonical_sum(partition_gathered).reshape_as(
            partition_source
        )
        partition_input = partition_source.clone()
        partition_primitive = FixedRankBF16SumPrimitive(
            partition_input, dist.group.WORLD
        )
        partition_primitive.launch(partition_input)
        assert torch.equal(partition_input, partition_expected)
        partition_permutation = torch.arange(127, -1, -1, device=device)
        partition_padded = torch.empty(
            (130, 5120), dtype=torch.bfloat16, device=device
        )
        partition_displaced = partition_padded[1:129]
        partition_displaced.copy_(
            partition_source.index_select(0, partition_permutation)
        )
        FixedRankBF16SumPrimitive(
            partition_displaced, dist.group.WORLD
        ).launch(partition_displaced)
        assert torch.equal(
            partition_displaced,
            partition_expected.index_select(0, partition_permutation),
        )

        partition_residual = residual.repeat((15, 1))[:128]
        partition_summed = torch.empty_like(partition_source)
        partition_norm = torch.empty_like(partition_source)
        partition_expected_summed = partition_residual + partition_expected
        partition_expected_norm = torch.empty_like(partition_source)
        ZeroCenteredRMSNormPrimitive(
            partition_expected_summed,
            weight,
            partition_expected_norm,
            epsilon=1e-6,
        ).launch(partition_expected_summed, weight, partition_expected_norm)
        partition_fused_input = partition_source.clone()
        partition_primitive.launch_residual_rms_norm(
            partition_fused_input,
            partition_residual,
            partition_summed,
            weight,
            partition_norm,
            epsilon=1e-6,
        )
        assert torch.equal(partition_fused_input, partition_source)
        assert torch.equal(partition_summed, partition_expected_summed)
        assert torch.equal(partition_norm, partition_expected_norm)

        fp32_workspace = torch.empty_like(source, dtype=torch.float32)
        fp32_result = torch.empty_like(source)
        fp32_workspace.copy_(source)
        dist.all_reduce(fp32_workspace, op=dist.ReduceOp.SUM)
        fp32_result.copy_(fp32_workspace)
        assert torch.equal(fp32_result, expected)
        assert torch.equal(fp32_result[0], fp32_result[1])

        fp32_displaced = torch.empty_like(padded, dtype=torch.float32)[1:10]
        fp32_displaced.copy_(source.index_select(0, permutation))
        dist.all_reduce(fp32_displaced, op=dist.ReduceOp.SUM)
        displaced.copy_(fp32_displaced)
        assert torch.equal(displaced, displaced_expected)

        # This cancellation pattern distinguishes the required rank-ordered
        # FP32 contract (result 1) from legal collective trees (result 0 or 2).
        # Observing 1 from NCCL here qualifies only this run/topology, not NCCL
        # as a topology-independent implementation of the contract.
        cancellation_by_rank = (2**25, 1, -(2**25), 1)
        cancellation = torch.full_like(source, cancellation_by_rank[rank])
        cancellation_fixed = cancellation.clone()
        primitive.launch(cancellation_fixed)
        assert torch.equal(cancellation_fixed, torch.ones_like(cancellation_fixed))
        cancellation_fp32 = cancellation.float()
        dist.all_reduce(cancellation_fp32, op=dist.ReduceOp.SUM)
        fp32_cancellation_result = float(cancellation_fp32[0, 0].item())

        baseline_input = source.clone()
        fixed_input = source.clone()
        fp32_input = torch.empty_like(source, dtype=torch.float32)
        fp32_output = torch.empty_like(source)
        for _ in range(5):
            baseline_input.copy_(source)
            dist.all_reduce(baseline_input, op=dist.ReduceOp.SUM)
            fixed_input.copy_(source)
            primitive.launch(fixed_input)
            fp32_input.copy_(source)
            dist.all_reduce(fp32_input, op=dist.ReduceOp.SUM)
            fp32_output.copy_(fp32_input)
        torch.cuda.synchronize()

        iterations = 100
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            baseline_input.copy_(source)
            dist.all_reduce(baseline_input, op=dist.ReduceOp.SUM)
        end.record()
        end.synchronize()
        nccl_ms = start.elapsed_time(end) / iterations

        start.record()
        for _ in range(iterations):
            fixed_input.copy_(source)
            primitive.launch(fixed_input)
        end.record()
        end.synchronize()
        fixed_ms = start.elapsed_time(end) / iterations

        start.record()
        for _ in range(iterations):
            fp32_input.copy_(source)
            dist.all_reduce(fp32_input, op=dist.ReduceOp.SUM)
            fp32_output.copy_(fp32_input)
        end.record()
        end.synchronize()
        fp32_nccl_ms = start.elapsed_time(end) / iterations

        # A captured communicator should not return to eager collectives on a
        # different stream.  Keep graph qualification last, matching serving.
        graph_common_outputs = []
        for rows in (512, 257, 256, 129, 128, 64, 37, 32, 16, 9):
            repeats = (rows + source.shape[0] - 1) // source.shape[0]
            graph_source = source.repeat((repeats, 1))[:rows]
            graph_input = graph_source.clone()
            graph_gathered = torch.empty(
                (4, graph_input.numel()),
                dtype=torch.bfloat16,
                device=device,
            )
            dist.all_gather_single(
                graph_gathered.view(-1), graph_source.reshape(-1)
            )
            graph_expected = _canonical_sum(graph_gathered).reshape_as(graph_input)
            graph_primitive = FixedRankBF16SumPrimitive(graph_input, dist.group.WORLD)
            dist.barrier()
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                for _ in range(258):
                    graph_input.copy_(graph_source)
                    graph_primitive.launch(graph_input)
            stream.synchronize()
            dist.barrier()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                graph_input.copy_(graph_source)
                graph_primitive.launch(graph_input)
            stream.synchronize()
            dist.barrier()
            for _ in range(129):
                graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(graph_input, graph_expected)
            graph_common_outputs.append(graph_input[:9].clone())
            graph.reset()
            torch.cuda.synchronize()
        assert all(
            torch.equal(graph_common_outputs[0], common)
            for common in graph_common_outputs[1:]
        )

        if rank == 0:
            Path(report_path).write_text(
                json.dumps(
                    {
                        "elements": source.numel(),
                        "fixed_ms": fixed_ms,
                        "fp32_nccl_ms": fp32_nccl_ms,
                        "fp32_nccl_cancellation_result": fp32_cancellation_result,
                        "nccl_identical_rows": nccl_identical,
                        "nccl_ms": nccl_ms,
                        "overhead_ratio": fixed_ms / nccl_ms,
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(
    os.environ.get("CUBIC_RUN_NCCL_TESTS") == "1",
    "set CUBIC_RUN_NCCL_TESTS=1 to run the four-GPU fixed-rank sum test",
)
class FixedRankBF16SumTests(unittest.TestCase):
    def test_automatic_transport_fp16_bf16_tp3_tp4_padding(self) -> None:
        if not dist.is_nccl_available() or torch.cuda.device_count() < 4:
            self.skipTest("four visible CUDA devices with NCCL are required")
        for world in (3, 4):
            with tempfile.TemporaryDirectory(prefix="cubic-auto-transport-") as directory:
                init_method = (Path(directory) / "store").resolve().as_uri()
                torch.multiprocessing.spawn(_automatic_transport_worker,
                                            args=(world, init_method), nprocs=world, join=True)

    def test_route_completion_shared_workspace_and_cross_rows(self) -> None:
        if not dist.is_nccl_available() or torch.cuda.device_count() < 4:
            self.skipTest("four visible CUDA devices with NCCL are required")
        for world in (3, 4):
            with tempfile.TemporaryDirectory(prefix="cubic-route-completion-") as directory:
                init_method = (Path(directory) / "store").resolve().as_uri()
                torch.multiprocessing.spawn(_route_completion_worker, args=(world, init_method),
                                            nprocs=world, join=True)

    def test_singleton_is_identity_without_workspace_or_collective(self) -> None:
        value = torch.randn((9, 5120), dtype=torch.bfloat16, device="cuda:0")
        value[0, 0] = -0.0
        before = value.clone()
        with patch.object(dist, "get_world_size", return_value=1), patch.object(
            torch, "empty", side_effect=AssertionError("singleton workspace allocation")
        ):
            primitive = FixedRankBF16SumPrimitive(value, None)
        with patch.object(dist, "all_gather_single", side_effect=AssertionError("collective")):
            primitive.launch(value)
        self.assertTrue(torch.equal(value.view(torch.int16), before.view(torch.int16)))
        residual = torch.randn_like(value)
        weight = torch.randn(5120, dtype=value.dtype, device=value.device)
        summed, output = torch.empty_like(value), torch.empty_like(value)
        expected_sum = (residual + value).to(value.dtype)
        expected = torch.empty_like(value)
        ZeroCenteredRMSNormPrimitive(expected_sum, weight, expected, epsilon=1e-6).launch(
            expected_sum, weight, expected
        )
        primitive.launch_residual_rms_norm(value, residual, summed, weight, output, epsilon=1e-6)
        self.assertTrue(torch.equal(summed, expected_sum))
        self.assertTrue(torch.equal(output, expected))

    def test_tp4_is_address_and_permutation_invariant(self) -> None:
        if not dist.is_nccl_available() or torch.cuda.device_count() < 4:
            self.skipTest("four visible CUDA devices with NCCL are required")
        with tempfile.TemporaryDirectory(prefix="cubic-fixed-rank-sum-") as directory:
            init_method = (Path(directory) / "store").resolve().as_uri()
            report_path = str(Path(directory) / "report.json")
            torch.multiprocessing.spawn(
                _fixed_rank_worker,
                args=(init_method, report_path),
                nprocs=4,
                join=True,
            )
            report = json.loads(Path(report_path).read_text(encoding="utf-8"))
        self.assertEqual(report["elements"], 9 * 5120)
        self.assertGreater(report["nccl_ms"], 0.0)
        self.assertGreater(report["fixed_ms"], 0.0)
        print("FIXED_RANK_SUM_REPORT=" + json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    unittest.main()


def _host_sum_worker(rank: int, world: int, init_method: str) -> None:
    import ctypes
    import resource
    import time
    from types import SimpleNamespace

    from cubic_serve.ops.cuda.collective.host_sum import HostSum, _common_sm_count
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(rank)
    dist.init_process_group("nccl", init_method=init_method, rank=rank, world_size=world,
                            timeout=timedelta(seconds=90))
    try:
        # Different reported resource limits must not create different flag
        # strides on peers. The limit stays below each real GPU's SM count.
        reported_sms = max(1, torch.cuda.get_device_properties(rank).multi_processor_count - rank)
        minimum = torch.tensor(reported_sms, device=rank, dtype=torch.int32)
        dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
        with patch("torch.cuda.get_device_properties",
                   return_value=SimpleNamespace(multi_processor_count=reported_sms)):
            assert _common_sm_count(
                torch.device("cuda", rank), dist.group.WORLD
            ) == int(minimum.item())
        for partitioned in (False, True):
            for dtype in (torch.float16, torch.bfloat16):
                for m, n in ((1, 31), (7, 1027), (37, 513), (64, 5120)):
                    gen = torch.Generator().manual_seed(615 + m)
                    parts = torch.randn(world, m, n, generator=gen).to(dtype)
                    expected = torch.zeros(m, n, dtype=torch.float32)
                    for source in parts:
                        expected += source.float()
                    expected = expected.to(dtype)
                    source = parts[rank].to(device=rank)
                    output = torch.empty_like(source)
                    affinity = os.sched_getaffinity(0)
                    binding = HostSum.create(output, dist.group.WORLD, partitioned=partitioned)
                    assert binding is not None, "test requires mapped-host capability"
                    assert os.sched_getaffinity(0) == affinity, "startup must restore caller affinity"
                    output.copy_(source)
                    binding.launch(output)
                    torch.cuda.synchronize()
                    assert torch.equal(output.cpu().view(torch.uint8), expected.view(torch.uint8))
                    # Change every generation, including across graph replay,
                    # so a stale peer stripe cannot match the expected value.
                    generations = torch.randn(8, world, m, n, generator=gen).to(dtype)
                    expected_generations = torch.zeros(8, m, n, dtype=torch.float32)
                    for peer in range(world):
                        expected_generations += generations[:, peer].float()
                    expected_generations = expected_generations.to(dtype)
                    inputs = generations[:, rank].contiguous().to(device=rank)
                    snapshots = torch.empty_like(inputs)
                    graph = torch.cuda.CUDAGraph()
                    try:
                        with torch.cuda.graph(graph):
                            for step in range(8):
                                if rank == step % world:
                                    torch.cuda._sleep(50000)
                                output.copy_(inputs[step])
                                binding.launch(output)
                                snapshots[step].copy_(output)
                        for late in range(world):
                            dist.barrier()
                            if rank == late:
                                torch.cuda._sleep(50000)
                            graph.replay()
                            actual = snapshots.cpu()
                            assert torch.equal(actual.view(torch.uint8),
                                               expected_generations.view(torch.uint8))
                        # Snapshot only the changed transport's protected tail,
                        # after completion, not model-wide state at every case.
                        torch.cuda.synchronize()
                        dist.barrier()
                        for plane in range(2):
                            for peer in range(world):
                                tail = binding.stride - binding.count * 2
                                pointer = binding.address + (plane * world + peer) * binding.stride + binding.count * 2
                                assert ctypes.string_at(pointer, tail) == bytes(tail)
                        end = (2 * world * binding.stride
                               + (3 if partitioned else 2) * world * binding.blocks * 64)
                        tail = len(binding.mapping) - end
                        assert ctypes.string_at(binding.address + end, tail) == bytes(tail)
                    finally:
                        torch.cuda.synchronize()
                        graph.reset()
                        dist.barrier()
                        binding.close()
    finally:
        dist.destroy_process_group()
        print(json.dumps(dict(rank=rank, world=world, proof_seconds=time.monotonic()-started,
            peak_host_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            peak_device_bytes=torch.cuda.max_memory_allocated(rank))), flush=True)


def test_host_sum_delayed_peer_tails_and_dtype() -> None:
    import pytest
    if torch.cuda.device_count() < 4:
        pytest.skip("requires four CUDA devices with shared host memory")
    for world in (2, 3, 4):
        with tempfile.TemporaryDirectory() as tmp:
            torch.multiprocessing.spawn(_host_sum_worker,
                args=(world, "file://" + str(Path(tmp) / "init")), nprocs=world, join=True)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("ranks", [2, 3, 4, 8])
def test_transport_fit_domain_covers_payload_without_moving_small_anchors(dtype, ranks) -> None:
    from cubic_serve.ops.cuda.collective.fixed_rank_sum import (
        _transport_key,
        _transport_probe_sizes,
    )
    from cubic_serve.ops.strategies import CollectiveParameters

    policy = CollectiveParameters()
    element_bytes = torch.empty((), dtype=dtype).element_size()
    alignment = ranks * element_bytes
    budget = max(32 * alignment, policy.transport_calibration_bytes)
    base_maximum = 4 * budget
    base = _transport_probe_sizes(budget, base_maximum, alignment)
    expected = [budget // 32, budget // 4, budget // 2, 3 * budget // 4,
                budget, 2 * budget, 4 * budget]
    assert base == [((value + alignment - 1) // alignment) * alignment for value in expected]
    for payload in (base[0], base[0] + alignment, budget // 2,
                    base_maximum - alignment, base_maximum, base_maximum + alignment,
                    3 * base_maximum + alignment):
        template = torch.empty(payload // element_bytes, dtype=dtype, device="meta")
        key = _transport_key(template, policy, ranks)
        sizes = _transport_probe_sizes(budget, key[-1], alignment)
        assert sizes[-1] >= payload
        assert len(sizes) == 7 and all(n % alignment == 0 for n in sizes)
        assert sizes[:5] == base[:5]
    # A real small completion must be inside its fitted interval; its payload
    # cannot be used to extrapolate a fit beginning at budget/32.
    for payload in (alignment, 4 * alignment, 32 * alignment,
                    33 * alignment, 79 * alignment, 5120,
                    budget // 32 - alignment):
        payload = ((payload + alignment - 1) // alignment) * alignment
        template = torch.empty(payload // element_bytes, dtype=dtype, device="meta")
        key = _transport_key(template, policy, ranks)
        sizes = _transport_probe_sizes(budget, key[-1], alignment)
        assert len(sizes) == 7 and all(n % alignment == 0 for n in sizes)
        assert sizes[0] <= payload <= sizes[-1]
        assert key[-1] < base_maximum
        # Actual small messages above the minimum probe domain are measured
        # training anchors, including sizes adjacent to power-of-two boundaries.
        if payload >= 32 * alignment:
            assert sizes[4] == payload


def _transport_submission_skew_worker(rank: int, init_method: str) -> None:
    import time

    from cubic_serve.ops.cuda.collective.fixed_rank_sum import _calibrate_transport
    from cubic_serve.ops.strategies import CollectiveParameters
    from cubic_serve.ops.cuda.startup import startup_weight_identity

    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    dist.init_process_group("nccl", init_method=init_method, rank=rank, world_size=2,
                            timeout=timedelta(seconds=60))
    replay = torch.cuda.CUDAGraph.replay

    def delayed_replay(graph):
        if rank == 0:
            time.sleep(.004)
        return replay(graph)

    try:
        policy = CollectiveParameters()
        for dtype in (torch.float16, torch.bfloat16):
            with startup_weight_identity(("transport-test", str(dtype))):
                ordinary = _calibrate_transport(torch.device("cuda", rank), dtype,
                                                dist.group.WORLD, policy, maximum_bytes=81920)
                with patch.object(torch.cuda.CUDAGraph, "replay", delayed_replay):
                    delayed = _calibrate_transport(torch.device("cuda", rank), dtype,
                                                   dist.group.WORLD, policy, maximum_bytes=81920)
            # A deliberate 4-ms host delay must not become 400 us per transfer.
            # Compare the measured transport, without requiring a device winner.
            for baseline, actual in zip(
                ordinary.predict(20480), delayed.predict(20480), strict=True,
            ):
                assert actual < baseline * 1.5, (dtype, baseline, actual)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(os.environ.get("CUBIC_TEST_QUANT_CUDA") != "1",
                    reason="explicit CUDA audit")
def test_transport_cost_excludes_host_submission_skew() -> None:
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    with tempfile.TemporaryDirectory() as tmp:
        torch.multiprocessing.spawn(_transport_submission_skew_worker,
            args=("file://" + str(Path(tmp) / "init"),), nprocs=2, join=True)


@pytest.fixture
def offline_transport_permission():
    from cubic_serve.ops.cuda.startup import startup_weight_identity
    with startup_weight_identity("ordered-transport-test"):
        yield


def test_normal_cold_transport_missing_price_does_not_search_candidates():
    from cubic_serve.ops.cuda.collective import fixed_rank_sum as module
    from cubic_serve.ops.strategies import CollectiveParameters
    class Group:
        pass
    with (patch.object(dist, "get_world_size", return_value=4),
          patch.object(torch.cuda, "is_current_stream_capturing", return_value=False),
          patch.object(module, "_calibrate_transport", side_effect=AssertionError("candidate search")) as calibrate):
        assert module._transport_cost(torch.empty(10240, device="meta", dtype=torch.bfloat16),
                                      Group(), CollectiveParameters()) is None
    calibrate.assert_not_called()


def test_competitive_invalid_transport_interval_gets_one_cached_local_fit(offline_transport_permission) -> None:
    from dataclasses import replace

    from cubic_serve.ops.cuda.collective import fixed_rank_sum as module
    from cubic_serve.ops.cuda.launch_cost import OrderedCollectiveCost
    from cubic_serve.ops.strategies import CollectiveParameters

    def point(size, host):
        return size, (20 + size * .00015, 30 + size * .00008,
                      host, 12 + size * .00004)

    primary = OrderedCollectiveCost.fit(
        [point(b, t) for b, t in ((8192, 8), (131072, 18), (262144, 50), (1048576, 96))],
        [point(b, t) for b, t in ((65536, 10), (196608, 20), (524288, 65))],
    )
    local = OrderedCollectiveCost.fit(
        [point(b, 8 + b * .00003) for b in (640, 10240, 20480, 81920)],
        [point(b, 8 + b * .00003) for b in (5120, 15360, 40960)],
    )
    class Group:
        pass

    policy = CollectiveParameters()
    template = torch.empty(10240, dtype=torch.bfloat16, device="meta")
    for usable in (True, False):
        group = Group()
        candidate = local if usable else replace(local, interval_error_fractions=(
            *local.interval_error_fractions[:2], (.4,) * 3, local.interval_error_fractions[3],
        ))
        with (
            patch.object(dist, "get_world_size", return_value=4),
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=False),
            patch.object(module, "_calibrate_transport", side_effect=(primary, candidate)) as calibrate,
        ):
            first = module._transport_cost(template, group, policy)
            assert first is (local if usable else primary)
            assert module._transport_cost(template, group, policy) is first
            assert [c.kwargs["maximum_bytes"] for c in calibrate.call_args_list] == [1048576, 81920]
            # The local supplement cannot overwrite the retained large-payload fit.
            large = torch.empty(327680, dtype=torch.bfloat16, device="meta")
            assert module._transport_cost(large, group, policy) is primary
            assert calibrate.call_count == 2


@pytest.mark.parametrize("incumbent", [0, 1])
@pytest.mark.parametrize("usable", [False, True])
def test_invalid_incumbent_cannot_block_qualified_faster_transport(incumbent, usable, offline_transport_permission) -> None:
    from dataclasses import replace

    from cubic_serve.ops.cuda.collective import fixed_rank_sum as module
    from cubic_serve.ops.cuda.launch_cost import OrderedCollectiveCost
    from cubic_serve.ops.strategies import CollectiveParameters

    def point(size):
        return size, (20 + size * .00015, 30 + size * .00008,
                      8 + size * .00003, 12 + size * .00004)

    primary = OrderedCollectiveCost.fit(
        [point(b) for b in (8192, 131072, 262144, 1048576)],
        [point(b) for b in (65536, 196608, 524288)],
    )
    local = OrderedCollectiveCost.fit(
        [point(b) for b in (640, 10240, 20480, 81920)],
        [point(b) for b in (5120, 15360, 40960)],
    )
    errors = list(primary.interval_error_fractions)
    errors[incumbent] = (.4,) * 3
    primary = replace(primary, interval_error_fractions=tuple(errors))
    candidate = local if usable else replace(local, interval_error_fractions=tuple(errors))
    policy = CollectiveParameters(exchange_latency_bytes=0 if incumbent else 1 << 20)
    template = torch.empty(10240, dtype=torch.bfloat16, device="meta")

    class Group:
        pass

    group = Group()
    with (
        patch.object(dist, "get_world_size", return_value=4),
        patch.object(torch.cuda, "is_current_stream_capturing", return_value=False),
        patch.object(module, "_calibrate_transport", side_effect=(primary, candidate)) as calibrate,
    ):
        first = module._transport_cost(template, group, policy)
        assert first is (local if usable else primary)
        assert module._transport_cost(template, group, policy) is first
        assert calibrate.call_count == 2
        # An invalid local comparison must still fail closed.
        assert first.select_host(20480, partitioned=bool(incumbent),
                                 min_gain_percent=5, max_error_percent=20) is usable


def test_competitive_large_payload_gets_one_actual_anchor_supplement(offline_transport_permission) -> None:
    from dataclasses import replace

    from cubic_serve.ops.cuda.collective import fixed_rank_sum as module
    from cubic_serve.ops.cuda.launch_cost import OrderedCollectiveCost
    from cubic_serve.ops.strategies import CollectiveParameters

    def point(size):
        return size, (20 + size * .00015, 30 + size * .00008,
                      8 + size * .00003, 12 + size * .00004)

    primary = OrderedCollectiveCost.fit(
        [point(b) for b in (8192, 131072, 262144, 1048576)],
        [point(b) for b in (65536, 196608, 524288)],
    )
    errors = list(primary.interval_error_fractions)
    errors[2:] = [(0, 0, .4), (0, 0, .4)]
    primary = replace(primary, interval_error_fractions=tuple(errors))
    local = OrderedCollectiveCost.fit(
        [point(b) for b in (20480, 327680, 655360, 2621440)],
        [point(b) for b in (163840, 491520, 1310720)],
    )

    class Group:
        pass

    group = Group()
    template = torch.empty(327680, dtype=torch.bfloat16, device="meta")
    with (
        patch.object(dist, "get_world_size", return_value=4),
        patch.object(torch.cuda, "is_current_stream_capturing", return_value=False),
        patch.object(module, "_calibrate_transport", side_effect=(primary, local)) as calibrate,
    ):
        assert module._transport_cost(template, group, CollectiveParameters()) is local
        assert module._transport_cost(template, group, CollectiveParameters()) is local
        maxima = [c.kwargs["maximum_bytes"] for c in calibrate.call_args_list]
        assert maxima == [1048576, 2621440]


@pytest.mark.skipif(os.environ.get('CUBIC_TEST_QUANT_CUDA') != '1', reason='explicit CUDA audit')
def test_transport_receipt_requires_matching_complete_peer_evidence(tmp_path, monkeypatch, offline_transport_permission):
    from cubic_serve.ops.cuda.collective import fixed_rank_sum as module
    from cubic_serve.ops.cuda.launch_cost import OrderedCollectiveCost
    from cubic_serve.ops.strategies import CollectiveParameters
    monkeypatch.setenv('CUBIC_SERVE_CACHE_DIR', str(tmp_path))
    def point(n):
        return n, (10 + .2 * n, 20 + .1 * n)
    cost = OrderedCollectiveCost.fit([point(n) for n in (10, 100, 400, 1000)],
                                     [point(n) for n in (50, 250, 600)])
    class Group:
        pass
    group = Group()
    monkeypatch.setattr(module.dist, 'is_initialized', lambda: True)
    monkeypatch.setattr(module.dist, 'get_world_size', lambda *a: 2)
    monkeypatch.setattr(module, '_transport_group_identity', lambda *a: ('peer-A', 'peer-B'))
    miss = [False]
    def gather(result, local, **kw):
        result[:] = [local, None if miss[0] else local]
    monkeypatch.setattr(module.dist, 'all_gather_object', gather)
    calls = []
    monkeypatch.setattr(module, '_calibrate_transport', lambda *a, **kw: calls.append('fit') or cost)
    template = torch.empty(128, dtype=torch.bfloat16, device='cuda:0')
    policy = CollectiveParameters()
    first = module._transport_cost(template, group, policy)
    assert calls == ['fit']
    assert list((tmp_path / 'calibration/launch').glob('*.json'))
    module._TRANSPORT_COSTS.pop(group)
    second = module._transport_cost(template, group, policy)
    assert first == second and calls == ['fit']
    module._TRANSPORT_COSTS.pop(group)
    miss[0] = True
    module._transport_cost(template, group, policy)
    assert calls == ['fit', 'fit']
