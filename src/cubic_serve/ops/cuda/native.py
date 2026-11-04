"""Startup-bound CUDA driver kernels with concrete image and operand identities.

The format owner supplies PTX and its source manifest. Linking happens before
capture, produces a real device cubin, and never selects an inference fallback.
Modules live for the CUDA context lifetime; a captured graph can outlive the
Python call that recorded its launch. Launch bindings retain their module too.
"""
from __future__ import annotations

import ctypes
import hashlib
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from importlib.metadata import version
from typing import Any, Literal

import torch
from cuda.bindings import driver as cuda_driver  # type: ignore[import-untyped]
from torch import Tensor

from cubic_serve.ops.cuda.compiled_artifact import (
    COMPILED_ARTIFACT_SCHEMA_ID,
    CompiledArtifactIdentity,
    LaunchReceipt,
    _json,
    bind_tensor,
    stable_digest,
)


def _check(result: tuple[Any, ...]) -> tuple[Any, ...]:
    if int(result[0]):
        raise RuntimeError(f"CUDA driver operation failed: {result[0]!r}")
    return result[1:]


def _hash(value: bytes) -> str:
    return "b2s256:" + hashlib.blake2s(value).hexdigest()


_module_images: dict[tuple[int, str], tuple[bytes, Any]] = {}
_module_images_lock = threading.Lock()


def _load_cubin_image(image: bytes) -> Any:
    """Intern context-lifetime code without retaining duplicate driver modules.

    Graphs can outlive their Python launch bindings, so images stay loaded.
    A unique driver context ID prevents reuse after a context address recycles.
    Only identical executable bytes share a handle; source receipts stay owned
    by each binding. No lookup occurs during graph replay.
    """
    context = _check(cuda_driver.cuCtxGetCurrent())[0]
    context_id = int(_check(cuda_driver.cuCtxGetId(context))[0])
    key = context_id, _hash(image)
    with _module_images_lock:
        previous = _module_images.get(key)
        if previous is not None:
            if previous[0] != image:
                raise RuntimeError("native executable identity collision")
            return previous[1]
        handle = _check(cuda_driver.cuModuleLoadData(image))[0]
        _module_images[key] = image, handle
        return handle


@cache
def _runtime_versions() -> tuple[tuple[str, str], ...]:
    return (("torch", str(torch.__version__)), ("cuda", str(torch.version.cuda)),
            ("cuda-bindings", version("cuda-bindings")))


@dataclass(frozen=True)
class CudaScalar:
    dtype: Literal["i32", "i64", "f32", "bool"]
    value: int | float | bool

    def pack(self) -> Any:
        if self.dtype in ("i32", "i64"):
            bits = 32 if self.dtype == "i32" else 64
            if (type(self.value) is not int
                    or not -(1 << (bits - 1)) <= self.value < 1 << (bits - 1)):
                raise ValueError("native integer argument exceeds its declared ABI")
            return (ctypes.c_int32 if bits == 32 else ctypes.c_int64)(self.value)
        if self.dtype == "bool" and type(self.value) is bool:
            return ctypes.c_bool(self.value)
        if self.dtype == "f32" and type(self.value) is float:
            return ctypes.c_float(self.value)
        raise ValueError("native scalar argument differs from its declared ABI")


class _ImmutableBinding:
    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("native CUDA bindings are immutable after startup")
        object.__setattr__(self, name, value)


class CudaModule(_ImmutableBinding):
    """Link supplied PTX in an already initialized, caller-owned CUDA context."""

    def __init__(
        self, ptx: bytes, *, sources: Mapping[str, bytes],
        compiler_metadata: Mapping[str, object], device: torch.device,
        cubin: bytes | None = None,
    ) -> None:
        if not ptx or not sources or any(
            type(k) is not str or not k or type(v) is not bytes or not v
            for k, v in sources.items()
        ):
            raise ValueError("native module requires PTX and a complete source manifest")
        self.device = torch.device(device)
        if self.device.type != "cuda" or self.device.index is None:
            raise ValueError("native module requires an explicit CUDA device")
        self.source_hash = stable_digest(tuple((k, _hash(v)) for k, v in sorted(sources.items())))
        self.compiler_metadata_json = _json(dict(compiler_metadata))
        self.ptx = ptx
        self.ptx_hash = _hash(ptx)
        with torch.cuda.device(self.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("native modules must be linked before CUDA Graph capture")
            self.sm = torch.cuda.get_device_capability(self.device)
            self.driver_version = int(_check(cuda_driver.cuDriverGetVersion())[0])
            if cubin is not None:
                if not cubin:
                    raise ValueError("native installed executable is empty")
                self.cubin = cubin
                self.handle = _load_cubin_image(cubin)
                self.cubin_hash = _hash(cubin)
                self._sealed = True
                return
            state = _check(cuda_driver.cuLinkCreate(
                1, [cuda_driver.CUjit_option.CU_JIT_OPTIMIZATION_LEVEL], [4]
            ))[0]
            try:
                terminated = ptx.rstrip(b"\0") + b"\0"
                _check(cuda_driver.cuLinkAddData(
                    state, cuda_driver.CUjitInputType.CU_JIT_INPUT_PTX,
                    terminated, len(terminated), b"cubic-native", 0, [], [],
                ))
                pointer, size = _check(cuda_driver.cuLinkComplete(state))
                self.cubin = ctypes.string_at(pointer, size)
                self.handle = _load_cubin_image(self.cubin)
            finally:
                _check(cuda_driver.cuLinkDestroy(state))
        self.cubin_hash = _hash(self.cubin)
        self._sealed = True

    def bind(
        self, entry: str, arguments: tuple[Tensor | CudaScalar | None, ...], *,
        operator_id: str, grid: tuple[int, int, int], block: tuple[int, int, int],
        shared_bytes: int = 0,
    ) -> NativeLaunch:
        return NativeLaunch(self, entry, arguments, operator_id, grid, block, shared_bytes)


class NativeLaunch(_ImmutableBinding):
    """One fixed function/argument/grid binding; data contents may change."""

    def __init__(
        self, module: CudaModule, entry: str,
        arguments: tuple[Tensor | CudaScalar | None, ...], operator_id: str,
        grid: tuple[int, int, int], block: tuple[int, int, int], shared_bytes: int,
    ) -> None:
        if not entry or not operator_id or any(
            type(dims) is not tuple or len(dims) != 3
            or any(type(v) is not int or v < 1 for v in dims) for dims in (grid, block)
        ) or type(shared_bytes) is not int or shared_bytes < 0:
            raise ValueError("native launch geometry or identity is invalid")
        self.module, self.grid, self.block = module, grid, block
        self.entry, self.arguments = entry, arguments
        self.operator_id, self.shared_bytes = operator_id, shared_bytes
        self.tensors = tuple(v for v in arguments if isinstance(v, Tensor))
        if not self.tensors or any(t.device != module.device for t in self.tensors):
            raise ValueError("native launch requires operands on its module device")
        self.bindings = tuple(bind_tensor(t) for t in self.tensors)
        packed: list[Any] = []
        abi: list[object] = []
        for value in arguments:
            if isinstance(value, Tensor):
                packed.append(ctypes.c_void_p(value.data_ptr()))
                abi.append(("tensor", str(value.dtype), tuple(value.shape), tuple(value.stride())))
            elif isinstance(value, CudaScalar):
                packed.append(value.pack())
                abi.append((value.dtype, bytes(packed[-1]).hex()))
            elif value is None:
                packed.append(ctypes.c_void_p(0))
                abi.append(("null",))
            else:
                raise TypeError("native arguments require tensors, explicit scalars or null")
        self._packed = tuple(packed)
        self._argument_values = tuple(bytes(v) for v in packed)
        self._parameters = (ctypes.c_void_p * len(packed))(
            *(ctypes.cast(ctypes.byref(v), ctypes.c_void_p) for v in packed)
        )
        self._parameter_addresses = tuple(self._parameters)
        self.specialization_json = _json(dict(entry=entry, arguments=abi, block=block,
                                              shared_bytes=shared_bytes))
        self.signature = stable_digest((module.source_hash, self.specialization_json, grid))
        self._ordinal, self._token = 0, object()
        with torch.cuda.device(module.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("native functions must be bound before graph capture")
            self.function = _check(
                cuda_driver.cuModuleGetFunction(module.handle, entry.encode())
            )[0]
            if shared_bytes:
                attribute = (
                    cuda_driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
                )
                current = int(_check(cuda_driver.cuFuncGetAttribute(attribute, self.function))[0])
                if shared_bytes > current:
                    _check(cuda_driver.cuFuncSetAttribute(self.function, attribute, shared_bytes))
        self._artifact = CompiledArtifactIdentity(
            COMPILED_ARTIFACT_SCHEMA_ID, operator_id, module.cubin_hash, module.source_hash,
            (("ptx", module.ptx_hash), ("cubin", module.cubin_hash)),
            module.compiler_metadata_json, self.specialization_json,
            ("cuda", module.sm[0] * 10 + module.sm[1], 32), module.driver_version, module.sm,
            (("optimization_level", "4"),),
            _runtime_versions(), self.signature, grid,
            backend="cuda-driver",
        )
        self._sealed = True

    def project_artifacts(self) -> tuple[CompiledArtifactIdentity, ...]:
        return (self._artifact,)

    def rebind_scratch(self, old: Tensor, new: Tensor) -> NativeLaunch:
        if self._ordinal:
            raise ValueError("native scratch must be rebound before the first launch")
        return self.module.bind(
            self.entry, tuple(new if v is old else v for v in self.arguments),
            operator_id=self.operator_id, grid=self.grid, block=self.block,
            shared_bytes=self.shared_bytes,
        )

    def validate_operands(self) -> None:
        if (tuple(bind_tensor(t) for t in self.tensors) != self.bindings
                or tuple(bytes(v) for v in self._packed) != self._argument_values
                or tuple(self._parameters) != self._parameter_addresses):
            raise ValueError("native launch operands changed after startup binding")

    def launch(self) -> LaunchReceipt:
        self.validate_operands()
        with torch.cuda.device(self.module.device):
            stream = int(torch.cuda.current_stream(self.module.device).cuda_stream)
            _check(cuda_driver.cuLaunchKernel(
                self.function, *self.grid, *self.block, self.shared_bytes,
                cuda_driver.CUstream(stream), ctypes.addressof(self._parameters), 0,
            ))
        object.__setattr__(self, "_ordinal", self._ordinal + 1)
        return LaunchReceipt(id(self), self.bindings, self.signature, self.grid,
                             stream, self._ordinal, self._token)
