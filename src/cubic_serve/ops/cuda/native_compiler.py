"""Content-addressed CUDA 13.0 baseline compilation shared by native operators."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

import torch

from cubic_serve import compiler_cache_root
from cubic_serve.ops.cuda.native import CudaModule
from cubic_serve.ops.cuda.native_bundle import installed_image


@lru_cache(maxsize=64)
def compile_native(sources_tuple: tuple[tuple[str, bytes], ...], device: torch.device,
                   namespace: str, *, minimum_isa: int = 80) -> tuple[CudaModule, str]:
    """Compile before capture; retain source, compiler and executable identities."""
    sources = dict(sources_tuple)
    if "entry.cu" not in sources or not namespace.isidentifier():
        raise ValueError("native compilation requires an entry and a cache namespace")
    if any(Path(name).is_absolute() or ".." in Path(name).parts for name in sources):
        raise ValueError("native source names must remain inside the compilation unit")
    if minimum_isa not in (80, 90):
        raise ValueError("native operator declares an unsupported minimum ISA")
    major, minor = torch.cuda.get_device_capability(device)
    if major * 10 + minor < minimum_isa:
        raise ValueError("native arithmetic instruction is unavailable on this device")
    installed = installed_image(
        namespace, sources, minimum_isa=minimum_isa, sm=major * 10 + minor,
    )
    if installed is not None:
        ptx, cubin, entry, metadata = installed
        return CudaModule(ptx, cubin=cubin, sources=sources,
                          compiler_metadata=metadata, device=device), entry
    baseline = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda-13.0")) / "bin" / "nvcc"
    nvcc = str(baseline) if baseline.is_file() else shutil.which("nvcc")
    if nvcc is None:
        raise RuntimeError("Native CUDA kernels require the CUDA 13 nvcc compiler at startup")
    compiler = subprocess.run(
        [nvcc, "--version"], check=True, capture_output=True, text=True, timeout=10
    ).stdout
    if "release 13.0" not in compiler:
        raise RuntimeError("Native CUDA compilation requires the CUDA 13.0 baseline")
    # An arithmetic instruction can require a newer ISA (e.g. the existing
    # BF16 exponential policy requires SM90). This is an instruction contract,
    # never a performance tuning selector. Other operators retain SM80 PTX.
    flags = ("--ptx", f"-arch=compute_{minimum_isa}", "-std=c++17", "-O3", "--expt-relaxed-constexpr")
    metadata = {"compiler": compiler, "flags": flags, "torch": str(torch.__version__)}
    digest = hashlib.blake2s(json.dumps(metadata, sort_keys=True).encode())
    for name, data in sorted(sources.items()):
        digest.update(name.encode() + b"\0" + data + b"\0")
    folder = compiler_cache_root() / namespace / digest.hexdigest()
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "kernel.ptx"
    with (folder / "compile.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not target.exists():
            for name, data in sources.items():
                path = folder / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            temporary = folder / "kernel.ptx.tmp"
            result = subprocess.run(
                [
                    nvcc,
                    *flags,
                    "-I",
                    str(folder),
                    "-I",
                    str(Path(torch.__file__).parent / "include"),
                    str(folder / "entry.cu"),
                    "-o",
                    str(temporary),
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode:
                raise RuntimeError("Native CUDA compilation failed:\n" + result.stderr[-6000:])
            temporary.replace(target)
        ptx = target.read_bytes()
    entries = re.findall(rb"\.visible \.entry (\S+)\(", ptx)
    if len(entries) != 1:
        raise RuntimeError("Native CUDA module must contain one explicit kernel entry")
    module = CudaModule(ptx, sources=sources, compiler_metadata=metadata, device=device)
    return module, entries[0].decode()
