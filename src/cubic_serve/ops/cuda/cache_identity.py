"""Persistent operator plans belong to CUDA implementations and GPU models.

Project revisions, Python control code, model names and physical device UUIDs
are not kernel identities. Invocation geometry is an entry inside that domain.
An executable's image already carries its compilation decisions; a pre-build
lookup uses the kernel sources and build recipe that will produce that image.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from functools import lru_cache
from pathlib import Path
from types import ModuleType

import torch

from cubic_serve import cache_root


def digest(value) -> str:
    return hashlib.blake2s(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()


@lru_cache(maxsize=32)
def native_headers(root: Path) -> tuple:
    """The include manifest used by native source bundles, without Python glue."""
    return tuple((str(path.relative_to(root)), hashlib.blake2s(path.read_bytes()).hexdigest())
                 for path in sorted(root.rglob("*"))
                 if path.is_file() and path.suffix in (".h", ".hpp", ".cuh"))


@lru_cache(maxsize=64)
def gpu_model(device) -> tuple:
    """Model name and its resources; only the name namespaces cached plans."""
    p = torch.cuda.get_device_properties(device)
    return (p.name, torch.cuda.get_device_capability(device), p.total_memory,
            p.multi_processor_count, p.L2_cache_size,
            p.shared_memory_per_block_optin, p.warp_size)


@lru_cache(maxsize=128)
def module_kernels(module: ModuleType) -> tuple:
    """Only kernel bodies/headers owned or imported by this operator module.

    Triton's own kernel key follows its JIT dependencies. No Python file or
    import-closure hashing is involved. Native generators supply their actual
    generated CUDA source separately when choosing a specialization.
    """
    from triton.runtime.jit import JITFunction

    kernels = [(name, value.cache_key) for name, value in vars(module).items()
               if isinstance(value, JITFunction)]
    source = inspect.getsourcefile(module)
    if source is not None:
        for path in sorted(Path(source).parent.iterdir()):
            if path.is_file() and path.suffix in (".cu", ".cuh", ".h", ".hpp"):
                kernels.append((path.name, hashlib.blake2s(path.read_bytes()).hexdigest()))
    return tuple(sorted(kernels))


@lru_cache(maxsize=8)
def cuda_libraries(names: tuple[str, ...]) -> tuple:
    """Content identities for loaded vendor libraries containing CUDA kernels."""
    paths = {}
    with Path("/proc/self/maps").open() as mappings:
        for line in mappings:
            path = Path(line.split()[-1])
            for name in names:
                if path.name.startswith(name + ".so"):
                    paths[name] = path.resolve()
    if set(paths) != set(names):
        raise ValueError("vendor CUDA image is not loaded")
    result = []
    for name in names:
        with paths[name].open("rb") as stream:
            result.append((name, hashlib.file_digest(stream, "blake2s").hexdigest()))
    return tuple(result)


def plan_path(owner: str, kernels, device, entry) -> Path:
    """The fingerprint directory contains only kernel identity and GPU model."""
    model = gpu_model(device)
    fingerprint = digest(("cubic.cuda-operator.v1", kernels, model[0]))
    # Resource partitions can change a valid choice's work without changing the
    # GPU model or kernel. Keep them inside the invocation entry, never the
    # implementation fingerprint; neither UUID nor CUDA ordinal is included.
    return cache_root() / "operators" / fingerprint / owner / (digest((model[1:], entry)) + ".json")
