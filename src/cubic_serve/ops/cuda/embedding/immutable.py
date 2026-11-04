"""Vocabulary-sharded lookup with a shared immutable host backing table.

Pinned upstreams: vLLM 89af4f1f offloader/uva.py (UVA capability/lifetime),
SGLang ba433bb4 utils/host_shared_memory.py (one allocation, per-process
registration), llama.cpp 8b4b3558 ggml-cuda.cu (explicit host registration).
Unlike those general offload paths, local vocabulary rows stay on the GPU.
Mapped rows are published once before capture and never modified at replay.
"""

from __future__ import annotations

import contextlib
import ctypes
import itertools
import logging
import mmap
import os
import socket
import tempfile
import weakref
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from cuda.bindings import driver

from cubic_serve.ops.cuda.native import CudaScalar, _check
from cubic_serve.ops.cuda.native_compiler import compile_native
from cubic_serve.ops.eager.quantization import QuantizedMatrix

_TABLES = weakref.WeakValueDictionary()
_LOG = logging.getLogger(__name__)


def mapped_host_memory_supported(device: torch.device) -> bool:
    """Query the execution device's mapping capability, without allocating a table."""
    if device.type != "cuda":
        return False
    with torch.cuda.device(device):
        index = device.index if device.index is not None else torch.cuda.current_device()
        return bool(_check(driver.cuDeviceGetAttribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY, index
        ))[0])


def host_registration_supported(index: int) -> bool:
    """Check registration without creating a context on another visible GPU."""
    return bool(_check(driver.cuDeviceGetAttribute(
        driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED, index
    ))[0])


def embedding_pointer(tensor: torch.Tensor, device: torch.device) -> int:
    """Bind GPU or explicitly pinned, immutable CPU storage before capture.

    The binding must retain the tensor until its last graph is destroyed. No
    registration, host lookup, allocation or transfer is performed at replay.
    Query the CUDA alias rather than assuming host/device pointers are equal.
    """
    if tensor.device == device:
        return tensor.data_ptr()
    if tensor.device.type != "cpu" or not tensor.is_pinned() or not tensor.is_contiguous():
        raise ValueError("host embedding storage must be contiguous pinned CPU memory")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("host embedding pointers must be bound before capture")
    with torch.cuda.device(device):
        if not mapped_host_memory_supported(device):
            raise RuntimeError("host embedding storage requires mapped host memory support")
        return int(_check(driver.cuMemHostGetDevicePointer(tensor.data_ptr(), 0))[0])


class ImmutableEmbeddingTable:
    """Shared weight lifetime; graph bindings hold strong references to this table."""

    @classmethod
    def create(cls, weight, span, group, device, dtype, *, operators=None):
        if group is None or dist.get_world_size(group) == 1:
            return None
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("immutable embedding must be constructed before capture")
        key = id(weight), id(group), device, dtype
        existing = _TABLES.get(key)
        reuse = torch.tensor(int(existing is not None), dtype=torch.int32, device=device)
        dist.all_reduce(reuse, op=dist.ReduceOp.MIN, group=group)
        if bool(reuse.item()):
            return existing
        supported = (
            isinstance(weight, torch.Tensor)
            and weight.dtype == dtype
            and dtype in (torch.float16, torch.bfloat16)
        ) or (
            isinstance(weight, QuantizedMatrix)
            and operators is not None
            and weight.encoding.format == "cubic-bitstream-v1"
            and weight.encoding.metadata == "e5m9-scale-curve2-tensor-fp16"
            and 2 <= weight.encoding.bits <= 8
            and dtype in (torch.float16, torch.bfloat16)
        )
        descriptors = [None] * dist.get_world_size(group)
        dist.all_gather_object(
            descriptors,
            (
                socket.gethostname(),
                supported,
                span.start,
                span.stop,
                weight.shape[1],
                str(dtype),
                repr(weight.encoding) if isinstance(weight, QuantizedMatrix) else "plain",
            ),
            group=group,
        )
        if (
            not all(d[1] for d in descriptors)
            or len({(d[0], d[4], d[5], d[6]) for d in descriptors}) != 1
        ):
            return None
        spans = sorted((d[2], d[3]) for d in descriptors)
        if spans[0][0] != 0 or any(a[1] != b[0] for a, b in itertools.pairwise(spans)):
            raise ValueError("embedding ownership must cover the vocabulary exactly once")
        obj = cls()
        obj.mapping = None
        obj.registered = False
        obj.weight, obj.group, obj.device, obj.dtype = weight, group, device, dtype
        obj.start, obj.stop, obj.width = span.start, span.stop, weight.shape[1]
        obj.vocabulary = spans[-1][1]
        if isinstance(weight, QuantizedMatrix):
            from cubic_serve.quantization.extensions.cubic.embedding import row_source

            spec = row_source(operators, weight, device, dtype)
            if spec is None:
                return None
            packets, obj.operands, obj.constants, source, obj.shared_bytes = spec
        else:
            stride = ((obj.width * 2 + 127) // 128) * 128
            packets = torch.zeros((weight.shape[0], stride), dtype=torch.uint8)
            packets[:, : obj.width * 2] = weight.detach().cpu().contiguous().view(torch.uint8)
            obj.operands, obj.constants = (weight,), (stride,)
            source = Path(__file__).with_suffix(".cu").read_bytes()
            obj.shared_bytes = 0
        obj.stride = packets.shape[1]
        obj.host_bytes = obj.vocabulary * obj.stride
        names = [None]
        if dist.get_rank(group) == 0:
            # Reject before touching pages rather than risking tmpfs SIGBUS.
            available = os.statvfs("/dev/shm")
            if obj.host_bytes <= available.f_bavail * available.f_frsize:
                fd, names[0] = tempfile.mkstemp(prefix="cubic-embedding-", dir="/dev/shm")
                os.ftruncate(fd, obj.host_bytes)
                os.close(fd)
        dist.broadcast_object_list(names, group=group, group_src=0, device=device)
        if names[0] is None:
            return None
        okay = False
        try:
            fd = os.open(names[0], os.O_RDWR)
            try:
                obj.mapping = mmap.mmap(fd, obj.host_bytes)
            finally:
                os.close(fd)
            view = np.ndarray((obj.vocabulary, obj.stride), dtype=np.uint8, buffer=obj.mapping)
            view[obj.start : obj.stop] = packets.numpy()
            del view, packets
            obj.address = ctypes.addressof(ctypes.c_char.from_buffer(obj.mapping))
            flags = int(driver.CU_MEMHOSTREGISTER_PORTABLE) | int(
                driver.CU_MEMHOSTREGISTER_DEVICEMAP
            )
            result = driver.cuMemHostRegister(obj.address, obj.host_bytes, flags)
            obj.registered = int(result[0]) == 0
            if obj.registered:
                mapped = _check(driver.cuMemHostGetDevicePointer(obj.address, 0))[0]
                obj.pointer = torch.tensor([int(mapped)], dtype=torch.uint64, device=device)
                okay = True
        except (OSError, RuntimeError) as error:
            _LOG.warning("immutable embedding admission failed: %s", error)
        status = torch.tensor(int(okay), device=device, dtype=torch.int32)
        dist.all_reduce(status, op=dist.ReduceOp.MIN, group=group)
        admitted = bool(status.item())  # Wait for every peer to open and publish.
        if dist.get_rank(group) == 0:
            os.unlink(names[0])
        if not admitted:
            obj.close()
            return None
        # Publication is complete on all ranks before any CUDA graph can read.
        obj.module, obj.entry = compile_native(
            (("entry.cu", source),), device, "embedding_immutable"
        )
        limit = int(
            _check(
                driver.cuDeviceGetAttribute(
                    driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                    device.index,
                )
            )[0]
        )
        vectors = obj.stride // 16
        obj.threads = min(limit, 1 << (max(32, vectors) - 1).bit_length())
        _TABLES[key] = obj
        _LOG.info(
            "immutable embedding admitted vocabulary=%d width=%d host_bytes=%d local_rows=%d",
            obj.vocabulary,
            obj.width,
            obj.host_bytes,
            obj.stop - obj.start,
        )
        return obj

    def close(self):
        if self.registered:
            with torch.cuda.device(self.device):
                torch.cuda.synchronize(self.device)
                _check(driver.cuMemHostUnregister(self.address))
            self.registered = False
        if self.mapping is not None:
            self.mapping.close()
            self.mapping = None

    def __del__(self):
        with contextlib.suppress(Exception):
            self.close()

    def bind(self, ids, output):
        return ImmutableEmbedding(self, ids, output)


class ImmutableEmbedding:
    """Fixed global-ID ABI, no host work or communication during graph replay."""

    def __init__(self, table, ids, output):
        if (
            ids.dtype != torch.int64
            or ids.ndim != 1
            or not ids.is_contiguous()
            or output.shape != (ids.numel(), table.width)
            or not output.is_contiguous()
            or ids.device != table.device
            or output.device != table.device
            or output.dtype != table.dtype
        ):
            raise ValueError("immutable embedding binding differs from fixed row ABI")
        self.table, self.ids, self.output = table, ids, output
        constants = table.width, table.start, table.stop, *table.constants
        self.call = table.module.bind(
            table.entry,
            (
                ids,
                *table.operands,
                table.pointer,
                output,
                *(CudaScalar("i32", x) for x in constants),
            ),
            operator_id="embedding.immutable",
            grid=(ids.numel(), 1, 1),
            block=(table.threads, 1, 1),
            shared_bytes=table.shared_bytes,
        )

    def launch(self, ids, weight, output):
        if ids is not self.ids or output is not self.output or weight is not self.table.weight:
            raise ValueError("immutable embedding operands changed after binding")
        self.call.launch()
