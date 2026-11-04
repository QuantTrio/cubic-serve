"""Exact identity for a retained CUDA artifact or a content-addressed vendor dispatch."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from functools import cache, lru_cache
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch
import triton  # type: ignore[import-untyped]
from cuda.bindings import driver as cuda_driver  # type: ignore[import-untyped]
from torch import Tensor
from torch.utils.weak import WeakIdKeyDictionary
from triton.compiler.compiler import CompiledKernel  # type: ignore[import-untyped]

from cubic_serve._compat import file_digest

COMPILED_ARTIFACT_SCHEMA_ID = "cubic.cuda.compiled-artifact.v1"
ARTIFACT_STAGES = ("ttir", "ttgir", "llir", "ptx", "cubin")
COMPILER_FLAGS = ("debug", "enable_fp_fusion", "enable_reflect_ftz", "num_ctas", "num_stages",
                  "num_warps", "sanitize_overflow")
RUNTIME_COMPONENTS = ("torch", "triton", "cuda")
NATIVE_STAGES = ("ptx", "cubin")
NATIVE_FLAGS = ("optimization_level",)
NATIVE_RUNTIME_COMPONENTS = ("torch", "cuda", "cuda-bindings")
VENDOR_BINARIES = ("libcublas", "libcublasLt")
VENDOR_FLAGS = ("allow_bf16_reduced_precision_reduction", "allow_tf32",
                "cublas_workspace_bytes", "cublaslt_workspace_bytes",
                "deterministic_algorithms", "preferred_blas_library", "workspace_config")
VENDOR_RUNTIME_COMPONENTS = ("torch", "cuda", "cublas")


def require_content_digest(value: object, label: str) -> None:
    if type(value) is not str or len(value) != 71 or value[:7] != "b2s256:":
        raise ValueError(f"{label} requires a content_digest digest")
    try:
        int(value[7:], 16)
    except ValueError as error:
        raise ValueError(f"{label} requires a content_digest digest") from error


def _json(value: Any) -> str:
    return json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _require_json(value: object, label: str) -> None:
    try:
        if type(value) is not str or _json(json.loads(value)) != value:
            raise ValueError
    except (TypeError, ValueError) as error:
        raise ValueError(f"compiled artifact {label} must be canonical JSON") from error


def _pairs(value: object, names: tuple[str, ...], label: str) -> tuple[tuple[str, str], ...]:
    if type(value) is not tuple or len(value) != len(names) or any(
        type(item) is not tuple or len(item) != 2
        or type(item[0]) is not str or type(item[1]) is not str
        for item in value
    ) or tuple(item[0] for item in value) != names:
        raise ValueError(f"compiled artifact omits or reorders {label}")
    return value


def _positive_tuple(value: object, length: int | None = None) -> bool:
    return (type(value) is tuple and bool(value) and (length is None or len(value) == length)
            and all(type(item) is int and item > 0 for item in value))


@dataclass(frozen=True, slots=True)
class TensorBinding:
    object_id: int
    pointer: int
    storage_pointer: int
    storage_offset: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: str
    device: str

    def matches(self, tensor: Tensor) -> bool:
        """Compare the complete live ABI without rebuilding its validated record."""
        return (
            isinstance(tensor, Tensor)
            and id(tensor) == self.object_id
            and tensor.data_ptr() == self.pointer
            and tensor.untyped_storage().data_ptr() == self.storage_pointer
            and tensor.storage_offset() == self.storage_offset
            and tuple(tensor.shape) == self.shape
            and tuple(tensor.stride()) == self.stride
            and str(tensor.dtype) == self.dtype
            and str(tensor.device) == self.device
        )

    def __post_init__(self) -> None:
        if any(type(value) is not int or value <= 0 for value in
               (self.object_id, self.pointer, self.storage_pointer)):
            raise ValueError("tensor binding requires positive exact integer identities")
        if type(self.storage_offset) is not int or self.storage_offset < 0:
            raise ValueError("tensor binding requires a nonnegative storage offset")
        # A zero-dimensional checkpoint scalar owns one element, with an empty
        # shape/stride tuple. It is distinct from a tensor with a zero extent.
        scalar_shape = type(self.shape) is tuple and self.shape == ()
        if (
            (not scalar_shape and not _positive_tuple(self.shape))
            or type(self.stride) is not tuple
            or len(self.stride) != len(self.shape)
            or any(type(item) is not int or item < 0 for item in self.stride)
        ):
            raise ValueError("tensor binding requires canonical positive shape and stride")
        if any(type(value) is not str or not value for value in (self.dtype, self.device)):
            raise ValueError("tensor binding requires dtype and device identities")

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("compiled tensor bindings cannot be subclassed")


@dataclass(frozen=True, slots=True)
class CompiledArtifactIdentity:
    schema_id: str
    operator_id: str
    compiled_hash: str
    source_hash: str
    stage_hashes: tuple[tuple[str, str], ...]
    compiler_metadata_json: str
    specialization_json: str
    target: tuple[str, int, int]
    driver_version: int
    sm: tuple[int, int]
    compiler_flags: tuple[tuple[str, str], ...]
    runtime: tuple[tuple[str, str], ...]
    launch_signature: str
    grid: tuple[int, int, int]
    backend: str = "triton"
    backend_binary_hashes: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if type(self.schema_id) is not str or self.schema_id != COMPILED_ARTIFACT_SCHEMA_ID:
            raise ValueError("compiled artifact schema is unsupported")
        if type(self.operator_id) is not str or not self.operator_id.strip():
            raise ValueError("compiled artifact requires an operator identity")
        require_content_digest(self.compiled_hash, "compiled artifact")
        require_content_digest(self.source_hash, "compiled source")
        _require_json(self.compiler_metadata_json, "compiler metadata")
        _require_json(self.specialization_json, "launch specialization")
        if type(self.target) is not tuple or len(self.target) != 3 or (
            type(self.target[0]) is not str or self.target[0] != "cuda"
            or any(type(item) is not int or item <= 0 for item in self.target[1:])
            or type(self.sm) is not tuple or len(self.sm) != 2
            or type(self.sm[0]) is not int or self.sm[0] <= 0
            or type(self.sm[1]) is not int or self.sm[1] < 0
            or self.target[1] != 10 * self.sm[0] + self.sm[1]
        ):
            raise ValueError("compiled artifact target differs from the active SM")
        if type(self.driver_version) is not int or self.driver_version <= 0:
            raise ValueError("compiled artifact requires a CUDA driver version")
        stages: tuple[str, ...]
        flags: tuple[str, ...]
        runtime: tuple[str, ...]
        if self.backend == "triton":
            stages, flags, runtime = ARTIFACT_STAGES, COMPILER_FLAGS, RUNTIME_COMPONENTS
            if self.backend_binary_hashes:
                raise ValueError("Triton artifact cannot contain vendor binaries")
        elif self.backend == "cuda-driver":
            stages, flags, runtime = NATIVE_STAGES, NATIVE_FLAGS, NATIVE_RUNTIME_COMPONENTS
            if self.backend_binary_hashes:
                raise ValueError("native CUDA image is identified by its cubin stage")
        elif self.backend == "torch-cublas":
            if self.stage_hashes:
                raise ValueError("vendor dispatch has no readable compiler stages or cubin")
            stages, flags, runtime = (), VENDOR_FLAGS, VENDOR_RUNTIME_COMPONENTS
            for _, digest in _pairs(
                self.backend_binary_hashes, VENDOR_BINARIES, "vendor binaries"
            ):
                require_content_digest(digest, "vendor binary")
        else:
            raise ValueError("compiled artifact backend is unsupported")
        for _, digest in _pairs(self.stage_hashes, stages, "compiler stages"):
            require_content_digest(digest, "compiler stage")
        if self.backend == "cuda-driver" and (
            self.compiled_hash != dict(self.stage_hashes)["cubin"]
        ):
            raise ValueError("native executable hash differs from its cubin stage")
        for _, value in _pairs(self.compiler_flags, flags, "compiler flags"):
            _require_json(value, "compiler flag")
        runtime_pairs = _pairs(self.runtime, runtime, "runtime")
        if any(not value for _, value in runtime_pairs):
            raise ValueError("compiled artifact runtime identity is incomplete")
        if type(self.launch_signature) is not str or not self.launch_signature.strip():
            raise ValueError("compiled artifact requires a launch signature")
        if not _positive_tuple(self.grid, 3):
            raise ValueError("compiled artifact requires a positive three-dimensional grid")

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("compiled artifact identities cannot be subclassed")

    @property
    def launch_specialization_digest(self) -> str:
        return stable_digest((self.schema_id, self.backend, self.operator_id,
                              self.specialization_json,
                              self.target, self.compiler_flags, self.launch_signature, self.grid))

    @property
    def artifact_digest(self) -> str:
        return _immutable_artifact_digest(self)


@dataclass(frozen=True, slots=True)
class LaunchReceipt:
    artifact_object_id: int
    bindings: tuple[TensorBinding, ...]
    launch_signature: str
    grid: tuple[int, int, int]
    stream: int
    ordinal: int
    _primitive_token: object = field(repr=False)

    def __post_init__(self) -> None:
        if type(self.artifact_object_id) is not int or self.artifact_object_id <= 0:
            raise ValueError("launch receipt requires an artifact object identity")
        if type(self.bindings) is not tuple or not self.bindings:
            raise ValueError("launch receipt requires exact tensor bindings")
        for binding in self.bindings:
            if type(binding) is not TensorBinding:
                raise TypeError("launch receipt requires exact tensor bindings")
            TensorBinding.__post_init__(binding)
        if type(self.launch_signature) is not str or not self.launch_signature:
            raise ValueError("launch receipt requires a launch signature")
        if not _positive_tuple(self.grid, 3) or type(self.stream) is not int or (
            self.stream < 0 or type(self.ordinal) is not int or self.ordinal <= 0
        ):
            raise ValueError("launch receipt requires valid grid, stream, and ordinal")

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("compiled launch receipts cannot be subclassed")


_TENSOR_BINDINGS = WeakIdKeyDictionary()
_ARTIFACT_STAGE_HASHES = WeakIdKeyDictionary()
_ARTIFACT_IDENTITIES = WeakIdKeyDictionary()


@lru_cache(maxsize=512)
def _immutable_artifact_digest(identity: CompiledArtifactIdentity) -> str:
    # Every field is already validated immutable data, never a tensor/handle.
    return stable_digest(asdict(identity))


def _artifact_stage_hash(artifact: CompiledKernel, name: str, value: str | bytes) -> str:
    if type(value) not in (str, bytes):
        return _hash(value)
    stages = _ARTIFACT_STAGE_HASHES.setdefault(artifact, {})
    previous = stages.get(name)
    if previous is not None and previous[0] is value:
        return previous[1]
    digest = _hash(value)
    stages[name] = (value, digest)
    return digest


def bind_tensor(tensor: Tensor) -> TensorBinding:
    if not isinstance(tensor, Tensor):
        raise TypeError("compiled tensor binding requires a tensor")
    previous = _TENSOR_BINDINGS.get(tensor)
    if previous is not None and previous.matches(tensor):
        return previous
    binding = TensorBinding(
        id(tensor), tensor.data_ptr(), tensor.untyped_storage().data_ptr(),
        int(tensor.storage_offset()), tuple(tensor.shape), tuple(tensor.stride()),
        str(tensor.dtype), str(tensor.device),
    )
    _TENSOR_BINDINGS[tensor] = binding
    return binding


def project_triton_artifact(
    artifact: CompiledKernel, *, operator_id: str, device: torch.device,
    launch_signature: str, grid: tuple[int, int, int],
) -> CompiledArtifactIdentity:
    """Inspect one retained kernel without compiling or resolving one."""
    if not isinstance(artifact, CompiledKernel):
        raise TypeError("artifact projection requires Triton's real CompiledKernel")
    if artifact.asm.get("cubin") != artifact.kernel:
        raise ValueError("compiled artifact cubin views disagree")
    asm = {name: artifact.asm.get(name) for name in ("source", *ARTIFACT_STAGES)}
    if any(not isinstance(value, (str, bytes)) for value in asm.values()):
        raise ValueError("compiled artifact omits a required source or compiler stage")
    metadata, target = artifact.metadata._asdict(), artifact.metadata.target
    status, driver_version = cuda_driver.cuDriverGetVersion()
    if int(status) != 0:
        raise RuntimeError(f"CUDA driver version query failed: {status!r}")
    specialization = {name: getattr(artifact.src, name) for name in
                      ("name", "language", "ext", "signature", "constants", "attrs")}
    arguments = (
        COMPILED_ARTIFACT_SCHEMA_ID, operator_id,
        _artifact_stage_hash(artifact, "hash", str(artifact.hash)),
        _artifact_stage_hash(artifact, "source", asm["source"]),
        tuple((name, _artifact_stage_hash(artifact, name, asm[name])) for name in ARTIFACT_STAGES), _json(metadata),
        _json(specialization), (str(target.backend), int(target.arch), int(target.warp_size)),
        int(driver_version), torch.cuda.get_device_capability(device),
        tuple((name, _json(metadata[name])) for name in COMPILER_FLAGS),
        (("torch", str(torch.__version__)), ("triton", str(triton.__version__)),
         ("cuda", str(torch.version.cuda))), launch_signature, grid,
    )
    # Re-read aliases, every stage, metadata, specialization and environment on
    # every projection. Memoize only the fully observed immutable identity.
    cacheable = (type(operator_id) is str and type(launch_signature) is str
                 and _positive_tuple(grid, 3))
    identities = _ARTIFACT_IDENTITIES.setdefault(artifact, {}) if cacheable else {}
    if cacheable and arguments in identities:
        return identities[arguments]
    identity = CompiledArtifactIdentity(*arguments)
    if cacheable:
        if len(identities) >= 16:
            identities.pop(next(iter(identities)))
        identities[arguments] = identity
    return identity


def project_torch_cublas_artifact(
    x: Tensor, weight: Tensor, output: Tensor, *, operator_id: str,
    launch_signature: str, grid: tuple[int, int, int],
) -> CompiledArtifactIdentity:
    """Content-address the ATen/cuBLAS dispatch closure; its selected cubin is opaque."""

    binaries = _mapped_cublas_hashes()
    status, driver_version = cuda_driver.cuDriverGetVersion()
    if int(status) != 0:
        raise RuntimeError(f"CUDA driver version query failed: {status!r}")
    flags, runtime = _torch_cublas_flags(), _torch_cublas_runtime()
    sm = torch.cuda.get_device_capability(x.device)
    properties = torch.cuda.get_device_properties(x.device)
    specialization = {
        "operation": "torch.mm(x,weight.t(),out=output)",
        "device": (str(x.device), properties.name),
        "x": (tuple(x.shape), tuple(x.stride()), str(x.dtype)),
        "weight": (tuple(weight.shape), tuple(weight.stride()), str(weight.dtype)),
        "output": (tuple(output.shape), tuple(output.stride()), str(output.dtype)),
    }
    metadata = {"dispatch": "aten.mm.out", "selected_cubin": "not exposed by vendor ABI"}
    return CompiledArtifactIdentity(
        COMPILED_ARTIFACT_SCHEMA_ID, operator_id, stable_digest((binaries, runtime)),
        _hash("torch.mm(x,weight.t(),out=output)"), (), _json(metadata),
        _json(specialization), ("cuda", 10 * sm[0] + sm[1], 32),
        int(driver_version), sm, flags, runtime,
        launch_signature, grid, "torch-cublas", binaries,
    )


def validate_torch_cublas_runtime(artifact: CompiledArtifactIdentity) -> None:
    CompiledArtifactIdentity.__post_init__(artifact)
    if artifact.backend != "torch-cublas" or (
        artifact.backend_binary_hashes, artifact.compiler_flags, artifact.runtime
    ) != (_mapped_cublas_hashes(), _torch_cublas_flags(), _torch_cublas_runtime()):
        raise ValueError("torch/cuBLAS runtime differs from the retained vendor identity")


def _torch_cublas_flags() -> tuple[tuple[str, str], ...]:
    return (
        ("allow_bf16_reduced_precision_reduction", _json(
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)),
        ("allow_tf32", _json(torch.backends.cuda.matmul.allow_tf32)),
        ("cublas_workspace_bytes", _json(torch._C._cuda_getCublasWorkspaceSize())),
        ("cublaslt_workspace_bytes", _json(torch._C._cuda_getCublasLtWorkspaceSize())),
        ("deterministic_algorithms", _json(torch.are_deterministic_algorithms_enabled())),
        ("preferred_blas_library", _json(str(torch.backends.cuda.preferred_blas_library()))),
        ("workspace_config", _json(os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""))),
    )


@cache
def _torch_cublas_runtime() -> tuple[tuple[str, str], ...]:
    return (("torch", f"{torch.__version__}@{torch.version.git_version}"),
            ("cuda", str(torch.version.cuda)), ("cublas", version("nvidia-cublas")))


@cache
def _mapped_cublas_hashes() -> tuple[tuple[str, str], ...]:
    paths: dict[str, Path] = {}
    with Path("/proc/self/maps").open(encoding="utf-8") as mappings:
        for line in mappings:
            path = Path(line.split()[-1])
            for name in VENDOR_BINARIES:
                if path.name.startswith(name + ".so"):
                    paths[name] = path.resolve()
    if tuple(sorted(paths)) != tuple(sorted(VENDOR_BINARIES)):
        raise RuntimeError("active cuBLAS and cuBLASLt binaries are not both mapped")
    values = []
    for name in VENDOR_BINARIES:
        with paths[name].open("rb") as binary:
            values.append((name, "b2s256:" + file_digest(binary, "blake2s").hexdigest()))
    return tuple(values)


def validate_launch_receipt(
    receipt: object, *, artifact_object_id: int, bindings: tuple[TensorBinding, ...],
    launch_signature: str, grid: tuple[int, int, int], stream: int, ordinal: int,
    primitive_token: object,
) -> None:
    if type(receipt) is not LaunchReceipt:
        raise ValueError("exact compiled launch receipt is required")
    LaunchReceipt.__post_init__(receipt)
    observed = (receipt.artifact_object_id, receipt.bindings, receipt.launch_signature,
                receipt.grid, receipt.stream, receipt.ordinal, receipt._primitive_token)
    if observed != (artifact_object_id, bindings, launch_signature, grid, stream, ordinal,
                    primitive_token):
        raise ValueError("compiled launch receipt does not close this primitive launch")


def receipt_payload(receipt: object) -> dict[str, object]:
    if type(receipt) is not LaunchReceipt:
        raise ValueError("exact compiled launch receipt is required")
    LaunchReceipt.__post_init__(receipt)
    return {"artifact_object_id": receipt.artifact_object_id,
            "bindings": tuple(asdict(item) for item in receipt.bindings),
            "launch_signature": receipt.launch_signature, "grid": receipt.grid,
            "stream": receipt.stream, "ordinal": receipt.ordinal}


def _hash(value: str | bytes) -> str:
    material = value.encode() if isinstance(value, str) else value
    return "b2s256:" + hashlib.blake2s(material).hexdigest()


def _canonical(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return (f"{type(value).__module__}.{type(value).__qualname__}", value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return _canonical(asdict(value))
    if hasattr(value, "_asdict"):
        return _canonical(value._asdict())
    if isinstance(value, Mapping):
        return [(_canonical(key), _canonical(item)) for key, item in
                sorted(value.items(), key=lambda pair: repr(pair[0]))]
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    return (f"{type(value).__module__}.{type(value).__qualname__}", repr(value))


def stable_digest(value: Any) -> str:
    return _hash(_json(value))
