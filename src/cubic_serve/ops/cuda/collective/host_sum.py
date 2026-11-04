"""Private mapped-host ordered sums, bound before capture on a shared host.

Memory ownership is per primitive/graph lease, never a process-global arena.
NCCL remains the capability/cost fallback chosen before capture; replay cannot
change transports. Peer polling and ordered arithmetic use the same portable
CUDA baseline on every device. A new device still needs numerical qualification.
"""
from __future__ import annotations

import ctypes
import errno
import hashlib
import logging
import math
import mmap
import os
import socket
import tempfile
from dataclasses import dataclass
from multiprocessing.reduction import DupFd
from pathlib import Path
from weakref import WeakKeyDictionary

import torch
import torch.distributed as dist
from cuda.bindings import driver, runtime

from cubic_serve.ops.cuda.compiled_artifact import stable_digest
from cubic_serve.ops.cuda.native import CudaScalar, NativeLaunch, _check
from cubic_serve.ops.cuda.native_compiler import compile_native

_GROUP_SMS: WeakKeyDictionary = WeakKeyDictionary()
_GROUP_HOST_NUMA: WeakKeyDictionary = WeakKeyDictionary()


class _ArenaDeclined(Exception):
    """All peers may retain the original backing before capture."""


def _numa_property(node):
    prop = driver.CUmemAllocationProp()
    prop.type = driver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST_NUMA
    prop.location.id = node
    prop.requestedHandleTypes = driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
    return prop


def _arena_driver(result):
    status = int(result[0])
    if status:
        if status in {int(driver.CUresult.CUDA_ERROR_NOT_SUPPORTED),
                      int(driver.CUresult.CUDA_ERROR_NOT_PERMITTED),
                      int(driver.CUresult.CUDA_ERROR_OUT_OF_MEMORY)}:
            raise _ArenaDeclined(f"CUDA error {status}")
        raise RuntimeError(f"host-sum backing failed: CUDA error {status}")
    return result[1] if len(result) == 2 else None


def _host_numa_capabilities(device, group):
    """Query once per group, including systems without NUMA or VMM support."""
    cached = _GROUP_HOST_NUMA.setdefault(group, {})
    if device in cached:
        return cached[device]
    local, failure = None, None
    try:
        dev = _arena_driver(driver.cuDeviceGet(device.index))
        attribute = driver.CUdevice_attribute
        supported = all(int(_arena_driver(driver.cuDeviceGetAttribute(a, dev))) for a in (
            attribute.CU_DEVICE_ATTRIBUTE_HOST_NUMA_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
            attribute.CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED))
        if supported:
            # CUDA specifies node zero when the machine has no NUMA topology.
            node = max(0, int(_arena_driver(driver.cuDeviceGetAttribute(
                attribute.CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, dev))))
            granularity = int(_arena_driver(driver.cuMemGetAllocationGranularity(
                _numa_property(node), driver.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM)))
            local = node, granularity
    except (AttributeError, OSError, _ArenaDeclined):
        pass
    except RuntimeError as error:
        failure = str(error)
    peers = [None] * dist.get_world_size(group)
    dist.all_gather_object(peers, (local, failure, socket.gethostname()), group=group)
    if any(error is not None for _, error, _ in peers):
        raise RuntimeError(next(error for _, error, _ in peers if error is not None))
    cached["same-host", device] = len({host for _, _, host in peers}) == 1
    cached[device] = (tuple(value for value, _, _ in peers)
                      if cached["same-host", device] else (None,) * len(peers))
    return cached[device]


@dataclass(frozen=True)
class _HostArenaPlan:
    stride: int
    size: int
    granularity: int
    # owner rank, host node, virtual offset, physical size
    allocations: tuple[tuple[int, int, int, int], ...]

    @classmethod
    def from_capabilities(cls, count, blocks, partitioned, peers):
        if not peers or any(p is None for p in peers):
            return None
        ranks = len(peers)
        granularity = math.lcm(*(p[1] for p in peers))
        align = lambda n, g: (n + g - 1) // g * g
        stride = align(count * 2, mmap.PAGESIZE)
        flags = (3 if partitioned else 2) * ranks * blocks * 64
        original = align(2 * ranks * stride + flags, mmap.PAGESIZE)
        # A tiny arena must not become a driver-granularity-sized allocation.
        if original < granularity:
            return None
        nodes = tuple(p[0] for p in peers)
        if len(set(nodes)) == 1:
            size = align(original, granularity)
            allocations = ((0, nodes[0], 0, size),)
        else:
            stride = align(count * 2, granularity)
            allocations = tuple((rank, nodes[rank], (plane * ranks + rank) * stride, stride)
                                for plane in range(2) for rank in range(ranks))
            flag_offset = 2 * ranks * stride
            flag_size = align(flags, granularity)
            size = flag_offset + flag_size
            allocations += ((0, nodes[0], flag_offset, flag_size),)
        # Same bound for common-node and cross-node layouts; never assume server RAM.
        if stride >= 2**31 or size > 2 * original:
            return None
        return cls(stride, size, granularity, allocations)


class _HostNumaArena:
    """Collectively constructed VMM backing with transactional local ownership.

    Only handles cross processes, never virtual addresses. Every process maps
    the same planes in the original rank order. Each construction phase reaches
    peer consensus before the next phase; a local failure cannot strand a peer
    in an import/broadcast. Graphs and leases must be released before close.
    """
    def __init__(self, plan, device):
        self.plan, self.device = plan, device
        self.address = 0
        self.handles = {}
        self.mappings = []

    def __len__(self):
        return self.plan.size

    @classmethod
    def create(cls, plan, device, group):
        arena = cls(plan, device)
        rank, ranks = dist.get_rank(group), dist.get_world_size(group)
        exports, received = {}, None

        def phase(action):
            failure = None
            try:
                action()
            except Exception as error:
                declined = (isinstance(error, _ArenaDeclined) or isinstance(error, OSError)
                            and error.errno in (errno.ENOMEM, errno.ENOSPC, errno.EPERM,
                                                errno.EACCES, errno.ENOTSUP))
                failure = (declined, str(error))
            failures = [None] * ranks
            dist.all_gather_object(failures, failure, group=group)
            if any(f is not None for f in failures):
                fatal = next((f for f in failures if f is not None and not f[0]), None)
                if fatal:
                    raise RuntimeError(fatal[1])
                raise _ArenaDeclined("host NUMA backing declined by a peer")

        def allocate():
            for index, (owner, node, _, size) in enumerate(plan.allocations):
                if owner != rank:
                    continue
                handle = _arena_driver(driver.cuMemCreate(size, _numa_property(node), 0))
                arena.handles[index] = handle
                fd = int(_arena_driver(driver.cuMemExportToShareableHandle(handle,
                    driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0)))
                exports[index] = {}
                try:
                    for peer in range(ranks):
                        if peer != rank:
                            exports[index][peer] = DupFd(fd)
                finally:
                    os.close(fd)

        def import_handles():
            for peer, items in enumerate(received):
                if peer == rank:
                    continue
                for index, destinations in items.items():
                    wrapped = destinations.pop(rank)
                    fd = wrapped.detach()
                    try:
                        arena.handles[index] = _arena_driver(driver.cuMemImportFromShareableHandle(
                            fd, driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR))
                    finally:
                        os.close(fd)

        def map_handles():
            arena.address = int(_arena_driver(driver.cuMemAddressReserve(plan.size, 0, 0, 0)))
            for index, (_, _, offset, size) in enumerate(plan.allocations):
                pointer = arena.address + offset
                _arena_driver(driver.cuMemMap(pointer, size, 0, arena.handles[index], 0))
                arena.mappings.append((pointer, size))
            cpu, gpu = driver.CUmemAccessDesc(), driver.CUmemAccessDesc()
            cpu.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST_NUMA
            cpu.location.id = plan.allocations[0][1]  # CUDA ignores id for host access.
            gpu.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
            gpu.location.id = device.index
            cpu.flags = gpu.flags = driver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
            _arena_driver(driver.cuMemSetAccess(arena.address, plan.size, [cpu, gpu], 2))

        try:
            phase(allocate)
            received = [None] * ranks
            dist.all_gather_object(received, exports, group=group)
            phase(import_handles)
            phase(map_handles)
            for plane in range(2):
                ctypes.memset(arena.address + (plane * ranks + rank) * plan.stride, 0, plan.stride)
            if rank == 0:
                offset = 2 * ranks * plan.stride
                ctypes.memset(arena.address + offset, 0, plan.size - offset)
            dist.barrier(group=group)
            return arena
        except BaseException:
            arena.close()
            raise
        finally:
            # DupFd's sender retains a descriptor until detach. Drain every
            # unpublished export, or this rank's unconsumed imports on failure.
            remaining = (exports.values() if received is None else
                         (destinations for peer, items in enumerate(received) if peer != rank
                          for destinations in items.values()))
            for destinations in remaining:
                targets = tuple(destinations) if received is None else (rank,)
                for peer in targets:
                    wrapped = destinations.pop(peer, None)
                    if wrapped is not None:
                        os.close(wrapped.detach())

    def close(self):
        # Pop only after release succeeds, so a failed cleanup is retryable.
        while self.mappings:
            pointer, size = self.mappings[-1]
            _arena_driver(driver.cuMemUnmap(pointer, size))
            self.mappings.pop()
        if self.address:
            _arena_driver(driver.cuMemAddressFree(self.address, self.plan.size))
            self.address = 0
        for index in tuple(self.handles):
            _arena_driver(driver.cuMemRelease(self.handles[index]))
            del self.handles[index]


def _prefault_rank_payload(owner, group: dist.ProcessGroup) -> bool:
    """Best-effort local first touch before peers register the whole arena."""
    previous = os.sched_getaffinity(0)
    changed = False
    okay = owner.mapping is not None
    try:
        if okay:
            bus = _check(runtime.cudaDeviceGetPCIBusId(32, owner.device.index))[0]
            bus = bus.decode().split("\0")[0].lower()
            try:
                spans = (Path("/sys/bus/pci/devices") / bus / "local_cpulist").read_text().strip()
                local = set()
                for span in spans.split(","):
                    bounds = span.split("-")
                    local.update(range(int(bounds[0]), int(bounds[-1]) + 1))
            except (OSError, ValueError):
                local = set()
            # Honor an explicit CPU affinity. With no usable topology, one
            # allowed CPU still gives calibration and serving stable first touch.
            os.sched_setaffinity(0, previous & local or {min(previous)})
            changed = True
    except OSError:
        okay = False
    ready = torch.tensor(int(okay), dtype=torch.int32, device=owner.device)
    try:
        dist.all_reduce(ready, op=dist.ReduceOp.MIN, group=group)
        if not int(ready.item()):
            return False
        for plane in range(2):
            offset = (plane * owner.ranks + owner.rank) * owner.stride
            ctypes.memset(owner.address + offset, 0, owner.stride)
        if owner.rank == 0:
            offset = 2 * owner.ranks * owner.stride
            ctypes.memset(owner.address + offset, 0, len(owner.mapping) - offset)
        # Payload pages were faulted by their writer on the selected CPU node.
        # Registration must not race the first touch of any peer's plane.
        dist.barrier(group=group)
        return True
    finally:
        if changed:
            os.sched_setaffinity(0, previous)


def _common_sm_count(device: torch.device, group: dist.ProcessGroup) -> int:
    """All peers must use identical stripe and flag geometry, even on mixed GPUs."""
    counts = _GROUP_SMS.setdefault(group, {})
    if device not in counts:
        count = torch.tensor(torch.cuda.get_device_properties(device).multi_processor_count,
                             dtype=torch.int32, device=device)
        dist.all_reduce(count, op=dist.ReduceOp.MIN, group=group)
        counts[device] = int(count.item())
    return counts[device]


def _registered_host_arena(obj, size, group):
    names = [None]
    if obj.rank == 0:
        try:
            fd, names[0] = tempfile.mkstemp(prefix="cubic-host-sum-", dir="/dev/shm")
            try:
                space = os.fstatvfs(fd)
                if size > space.f_bavail * space.f_frsize:
                    # ftruncate alone can succeed on a full tmpfs and later
                    # turn first touch into SIGBUS. Decline before peer mapping.
                    raise OSError(errno.ENOSPC, "shared host backing capacity exhausted")
                os.ftruncate(fd, size)
            finally:
                os.close(fd)
        except OSError:
            if names[0] is not None:
                os.unlink(names[0])
            names[0] = None
    dist.broadcast_object_list(names, group=group, group_src=0, device=obj.device)
    okay = names[0] is not None
    failure = 0
    try:
        if okay:
            fd = os.open(names[0], os.O_RDWR)
            try:
                obj.mapping = mmap.mmap(fd, size)
            finally:
                os.close(fd)
            obj.address = ctypes.addressof(ctypes.c_char.from_buffer(obj.mapping))
    except OSError as error:
        logging.getLogger(__name__).warning(
            "host-sum shared arena unavailable rank=%d error=%s", obj.rank, error)
        okay = False
    # CUDA registration can fault the entire file from any registering CPU.
    # Random page placement made the fitted and retained arenas disagree.
    okay = _prefault_rank_payload(obj, group)
    try:
        if okay:
            flags = (int(driver.CU_MEMHOSTREGISTER_PORTABLE)
                     | int(driver.CU_MEMHOSTREGISTER_DEVICEMAP))
            with torch.cuda.device(obj.device):
                result = driver.cuMemHostRegister(obj.address, size, flags)
            failure = int(result[0])
            okay = failure == 0
            obj.registered = okay
            if okay:
                with torch.cuda.device(obj.device):
                    result = driver.cuMemHostGetDevicePointer(obj.address, 0)
                failure = int(result[0])
                okay = failure == 0
                if okay:
                    obj.mapped_address = int(result[1])
            if failure:
                logging.getLogger(__name__).warning(
                    "host-sum mapping unavailable rank=%d error=%s", obj.rank, failure)
    except OSError as error:
        logging.getLogger(__name__).warning("host-sum shared arena unavailable rank=%d error=%s", obj.rank, error)
        okay = False
    declined = {int(driver.CUresult.CUDA_ERROR_NOT_SUPPORTED),
                int(driver.CUresult.CUDA_ERROR_NOT_PERMITTED),
                int(driver.CUresult.CUDA_ERROR_OUT_OF_MEMORY)}
    status = -failure if failure and failure not in declined else int(okay)
    capability = torch.tensor(status, dtype=torch.int32, device=obj.device)
    dist.all_reduce(capability, op=dist.ReduceOp.MIN, group=group)
    # NCCL return only enqueues work. Observe completion before unlink:
    # every peer must have opened/registered the shared file first.
    admitted = int(capability.item())
    if obj.rank == 0 and names[0] is not None:
        os.unlink(names[0])
    if admitted <= 0:
        obj.close()
        if admitted < 0:
            raise RuntimeError(f"host-sum registration failed on a peer: CUDA error {-admitted}")
        return False
    return True


class HostSum:
    @classmethod
    def create(cls, template: torch.Tensor, group: dist.ProcessGroup,
               threads: int = 128, *, partitioned: bool = False) -> HostSum | None:
        """Collectively admit host mapping; no probe or fallback during capture."""
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("host-sum ownership must be bound before capture")
        if (template.dtype not in (torch.float16, torch.bfloat16)
                or not template.is_contiguous() or template.device.type != "cuda"
                or not 0 < template.numel() < 2**31 or threads not in (64, 128, 256)):
            raise ValueError("host sum requires contiguous half operands and legal resources")
        obj = cls()
        obj.mapping = None
        obj.registered = False
        obj.bindings: dict[tuple[int, int], NativeLaunch] = {}
        obj.device, obj.dtype, obj.shape = template.device, template.dtype, tuple(template.shape)
        obj.count, obj.group = template.numel(), group
        obj.rank, obj.ranks = dist.get_rank(group), dist.get_world_size(group)
        obj.threads = threads
        obj.partitioned = partitioned
        # Each partition gets a full CTA of useful reduction work. Publication
        # and gathering cover the same rank-striped range as that CTA.
        stripe_ranks = obj.ranks if partitioned else 1
        obj.blocks = min(_common_sm_count(template.device, group),
                         math.ceil(obj.count / (threads * 8 * stripe_ranks)))
        obj.stride = math.ceil(obj.count * 2 / mmap.PAGESIZE) * mmap.PAGESIZE
        if obj.stride >= 2**31:
            return None
        # Direct sums alternate payload and ready slots by generation. Publishing
        # the next generation proves the previous slot is no longer being read.
        planes, phases = (2, 3) if partitioned else (2, 2)
        size = math.ceil((planes * obj.stride * obj.ranks + phases * obj.ranks * obj.blocks * 64)
                         / mmap.PAGESIZE) * mmap.PAGESIZE
        obj.workload_resource_identity = ("ordered_host_backing_v1", "registered-mmap", obj.stride, size)
        capabilities = _host_numa_capabilities(obj.device, group)
        if not _GROUP_HOST_NUMA[group].get(("same-host", obj.device), True):
            return None  # A shared file name is not coherent cross-host memory.
        plan = _HostArenaPlan.from_capabilities(obj.count, obj.blocks, partitioned,
                                               capabilities)
        obj.planned_workload_resource_identity = obj.workload_resource_identity
        if plan is not None:
            obj.planned_workload_resource_identity = ("ordered_host_backing_v1", "host-numa-vmm",
                                                      plan.stride, plan.size, plan.granularity,
                                                      plan.allocations)
            try:
                arena = _HostNumaArena.create(plan, obj.device, group)
            except _ArenaDeclined:
                pass  # Every peer declined together; bind original backing instead.
            else:
                obj.mapping = arena
                obj.address = obj.mapped_address = arena.address
                obj.stride = plan.stride
                obj.workload_resource_identity = obj.planned_workload_resource_identity
        if obj.mapping is None and not _registered_host_arena(obj, size, group):
            return None
        # One immutable pointer descriptor avoids embedding allocation addresses
        # in the semantic launch key. This owner retains the actual mapping.
        obj.pointer = torch.tensor([obj.mapped_address], dtype=torch.uint64, device=template.device)
        obj.epochs = torch.zeros(obj.blocks, dtype=torch.int32, device=template.device)
        source = (f"#define HOST_SUM_PARTITIONED {int(partitioned)}\n"
                  f"#define HOST_SUM_FP16 {int(template.dtype is torch.float16)}\n"
                  + Path(__file__).with_suffix(".cu").read_text())
        obj.module, obj.entry = compile_native((("entry.cu", source.encode()),), template.device, "collective")
        obj.signature = stable_digest(("ordered_host_sum_v3", obj.shape, str(obj.dtype), partitioned,
            obj.rank, obj.ranks, obj.blocks, obj.threads, obj.module.source_hash,
            hashlib.blake2s(obj.module.cubin).hexdigest(), obj.workload_resource_identity))
        obj.bind(template, template)
        return obj

    def bind(self, value: torch.Tensor, output: torch.Tensor) -> NativeLaunch:
        if (value.device != self.device or value.dtype != self.dtype
                or tuple(value.shape) != self.shape or not value.is_contiguous()
                or output.device != self.device or output.dtype != self.dtype
                or tuple(output.shape) != self.shape or not output.is_contiguous()):
            raise ValueError("host-sum operands differ from their sealed shape")
        key = (value.data_ptr(), output.data_ptr())
        call = self.bindings.get(key)
        if call is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("host sum encountered an unbound operand during capture")
            args = (value, output, self.pointer,
                    *(CudaScalar("i32", n) for n in (self.count, self.stride, self.ranks, self.rank)),
                    self.epochs)
            call = self.module.bind(self.entry, args, operator_id="collective.ordered_host_sum",
                                    grid=(self.blocks, 1, 1), block=(self.threads, 1, 1))
            self.bindings[key] = call
        return call

    def launch(self, value: torch.Tensor, output: torch.Tensor | None = None) -> None:
        self.bind(value, value if output is None else output).launch()

    def close(self) -> None:
        """Caller releases graphs/leases first; unregister before closing mapping."""
        if self.registered:
            with torch.cuda.device(self.device):
                _check(driver.cuMemHostUnregister(self.address))
            self.registered = False
        if self.mapping is not None:
            self.mapping.close()
            self.mapping = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass  # CUDA context may already have been destroyed at process exit.
