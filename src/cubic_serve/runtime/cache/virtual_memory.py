"""Fixed CUDA addresses with explicit, granularity-rounded physical backing.

Mappings may grow only before storage is sealed for serving. Tensor views own
this allocation through CUDA array interoperability, including their last view.
No CUDA kernel, model arithmetic, allocator tuning or runtime capture lives here.
"""
from __future__ import annotations

import logging
import math
from threading import RLock

import torch
from cuda.bindings import driver

from cubic_serve.ops.cuda.native import _check

_LOGGER = logging.getLogger(__name__)


def virtual_memory_supported(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    index = torch.cuda.current_device() if device.index is None else device.index
    return bool(_check(driver.cuDeviceGetAttribute(
        driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
        index,
    ))[0])


class _ArrayView:
    def __init__(self, owner: VirtualAllocation, offset: int, size: int):
        self.owner, self.offset, self.size = owner, offset, size

    @property
    def __cuda_array_interface__(self):
        return {"shape": (self.size,), "typestr": "|u1",
                "data": (self.owner.base + self.offset, False), "version": 3}


class VirtualAllocation:
    """An owned reservation; each map covers an entire independent handle."""

    def __init__(self, size: int, device: torch.device, *, peers: tuple[int, ...] = ()):
        if type(size) is not int or size <= 0 or device.type != "cuda":
            raise ValueError("virtual storage requires positive bytes and a CUDA device")
        index = torch.cuda.current_device() if device.index is None else device.index
        self.device = torch.device("cuda", index)
        self.logical_bytes = size
        self.base = 0
        self._mappings: list[tuple[int, int]] = []
        self._sealed = False
        self._closed = False
        self._lock = RLock()
        with torch.cuda.device(self.device):
            prop = driver.CUmemAllocationProp()
            prop.type = driver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
            prop.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
            prop.location.id = index
            prop.requestedHandleTypes = driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_NONE
            self._prop = prop
            self.granularity = int(_check(driver.cuMemGetAllocationGranularity(
                prop, driver.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM,
            ))[0])
            self.reserved_bytes = self._align(size)
            accessible = [index]
            for peer in sorted(set(peers) - {index}):
                if _check(driver.cuDeviceCanAccessPeer(peer, index))[0]:
                    accessible.append(peer)
            self._access = []
            for peer in accessible:
                desc = driver.CUmemAccessDesc()
                desc.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
                desc.location.id = peer
                desc.flags = driver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
                self._access.append(desc)
            self.base = int(_check(driver.cuMemAddressReserve(
                self.reserved_bytes, self.granularity, 0, 0,
            ))[0])

    def _align(self, n: int) -> int:
        return (n + self.granularity - 1) // self.granularity * self.granularity

    @property
    def backed_bytes(self) -> int:
        return sum(end - start for start, end in self._mappings)

    def _missing(self, offset: int, size: int) -> tuple[tuple[int, int], ...]:
        if (type(offset) is not int or type(size) is not int or offset < 0 or size < 0
                or offset + size > self.logical_bytes):
            raise ValueError("physical mapping exceeds its reserved tensor extent")
        if not size:
            return ()
        cursor, stop = offset // self.granularity * self.granularity, self._align(offset + size)
        gaps = []
        for start, end in sorted(self._mappings):
            if end <= cursor:
                continue
            if start >= stop:
                break
            if start > cursor:
                gaps.append((cursor, start))
            cursor = max(cursor, end)
        if cursor < stop:
            gaps.append((cursor, stop))
        return tuple(gaps)

    def additional_bytes(self, offset: int, size: int) -> int:
        with self._lock:
            return sum(end - start for start, end in self._missing(offset, size))

    def _bytes_tensor(self, offset: int, size: int) -> torch.Tensor:
        value = torch.as_tensor(_ArrayView(self, offset, size), device=self.device)
        if value.data_ptr() != self.base + offset:
            raise RuntimeError("CUDA interoperability copied a virtual storage view")
        return value

    def ensure_range(self, offset: int, size: int) -> None:
        with self._lock, torch.cuda.device(self.device):
            if self._closed or self._sealed or torch.cuda.is_current_stream_capturing():
                raise RuntimeError("virtual storage mapping requires unsealed startup ownership")
            before = len(self._mappings)
            try:
                for start, end in self._missing(offset, size):
                    count = end - start
                    handle = _check(driver.cuMemCreate(count, self._prop, 0))[0]
                    mapped = False
                    try:
                        _check(driver.cuMemMap(self.base + start, count, 0, handle, 0))
                        mapped = True
                        _check(driver.cuMemSetAccess(
                            self.base + start, count, self._access, len(self._access),
                        ))
                    except BaseException:
                        if mapped:
                            _check(driver.cuMemUnmap(self.base + start, count))
                        raise
                    finally:
                        _check(driver.cuMemRelease(handle))
                    self._mappings.append((start, end))
                    # The aligned allocation is readable even if its last bytes
                    # lie beyond the logical tensor. Initialize only new maps.
                    self._bytes_tensor(start, count).zero_()
            except BaseException:
                torch.cuda.current_stream(self.device).synchronize()
                for start, end in reversed(self._mappings[before:]):
                    _check(driver.cuMemUnmap(self.base + start, end - start))
                del self._mappings[before:]
                raise

    def tensor(self, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        if math.prod(shape) * dtype.itemsize != self.logical_bytes:
            raise ValueError("virtual tensor shape differs from its reserved byte extent")
        if not any(start == 0 for start, _ in self._mappings):
            raise ValueError("virtual tensor needs a backed first address for CUDA interoperability")
        return self._bytes_tensor(0, self.logical_bytes).view(dtype).reshape(shape)

    def seal(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("closed virtual storage cannot be sealed")
            torch.cuda.current_stream(self.device).synchronize()
            self._sealed = True

    def __del__(self):
        if not getattr(self, "base", 0) or getattr(self, "_closed", True):
            return
        self._closed = True
        try:
            # Tensor views retain this owner. Release only after its last view,
            # and only after any queued operation on this device has completed.
            with torch.cuda.device(self.device):
                torch.cuda.synchronize(self.device)
                for start, end in reversed(self._mappings):
                    _check(driver.cuMemUnmap(self.base + start, end - start))
                _check(driver.cuMemAddressFree(self.base, self.reserved_bytes))
        except Exception:
            _LOGGER.warning("virtual storage teardown failed", exc_info=True)
