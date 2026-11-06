"""Cubic Serve."""

import os
from pathlib import Path

__version__ = "0.1.1"


def cache_root() -> Path:
    """Root of all writable Cubic Serve caches."""
    configured = os.environ.get("CUBIC_SERVE_CACHE_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    base = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))).expanduser()
    return (base / "cubic-serve").resolve()


def compiler_cache_root() -> Path:
    """Writable compiler products under the common cache root."""
    return cache_root() / "compile"


def _configure_compiler_cache() -> None:
    # Resolve before importing Torch/Triton or creating a CUDA context. Spawned
    # workers inherit the same root; no model repository owns this policy.
    base = cache_root()
    os.environ["CUBIC_SERVE_CACHE_DIR"] = str(base)
    root = base / "compile"
    os.environ["TRITON_CACHE_DIR"] = str(root / "triton")
    os.environ["CUDA_CACHE_PATH"] = str(root / "cuda-driver")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(root / "inductor")
    os.environ["TORCH_EXTENSIONS_DIR"] = str(root / "torch-extensions")
    os.environ["PYTORCH_KERNEL_CACHE_PATH"] = str(root / "torch-kernels")
    os.environ["TORCH_HOME"] = str(base / "torch")


_configure_compiler_cache()
