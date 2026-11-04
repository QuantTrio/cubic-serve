"""Explicit extension registration; no directory scanning or hot replacement.

Register before constructing checkpoint/config/execution objects. Builtins only
register descriptors, so metadata inspection does not import CUDA or kernels.
"""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import entry_points

from cubic_serve.quantization.api import QuantizationExtension

_extensions: dict[str, QuantizationExtension] = {}
_loaded = False


def register(extension: QuantizationExtension) -> None:
    if not isinstance(extension, QuantizationExtension):
        raise TypeError("quantization entry point must export a QuantizationExtension descriptor")
    for current in _extensions.values():
        if (
            extension.name == current.name
            or extension.format_id == current.format_id
            or set(extension.checkpoint_methods) & set(current.checkpoint_methods)
            or set(extension.group_formats) & set(current.group_formats)
        ):
            raise ValueError(f"quantization extension registration conflicts with {current.name}")
    _extensions[extension.format_id] = extension


def extensions() -> tuple[QuantizationExtension, ...]:
    global _loaded
    if not _loaded:
        # The only builtin list. A new external package uses register() instead.
        for name in ("awq", "fp8", "nvfp4", "cubic", "int", "ggml"):
            register(
                import_module(f"cubic_serve.quantization.extensions.{name}.extension").EXTENSION
            )
        for entry in sorted(entry_points(group="cubic_serve.quantization"), key=lambda e: e.name):
            register(entry.load())
        _loaded = True
    return tuple(_extensions.values())


def get_extension(format_id: str) -> QuantizationExtension:
    extensions()
    try:
        return _extensions[format_id]
    except KeyError as error:
        raise ValueError(f"unregistered quantization encoding {format_id!r}") from error


def supports_compute_mode(encoding, operation: str, mode: str) -> bool:
    """Ask the format owner about an operator, never infer it from model identity."""
    if mode == "wna16":
        return True
    policy = get_extension(encoding.format).module("policy")
    predicate = getattr(policy, "supports_compute_mode", None)
    return bool(predicate is not None and predicate(encoding, operation, mode))


def require_compute_mode(encoding, operation: str, mode: str) -> None:
    """Validate an actual operator consumer before preparing native storage."""
    if not supports_compute_mode(encoding, operation, mode):
        raise ValueError(f"{encoding.format} {operation} does not support {mode}")


def select_compute_mode(encoding, operation: str, mode: str, *, m: int, n: int, k: int,
                        dtype, device, activation_quantized: bool = False) -> str:
    """Resolve a format's measured arithmetic plan for one physical operator."""
    require_compute_mode(encoding, operation, mode)
    policy = get_extension(encoding.format).module("policy")
    selector = getattr(policy, "select_compute_mode", None)
    selected = (mode if selector is None else selector(
        encoding, operation, mode, m=m, n=n, k=k, dtype=dtype, device=device,
        activation_quantized=activation_quantized,
    ))
    # A format may select geometry/backends within the requested arithmetic,
    # never silently change the activation precision of a packed projection.
    if selected != mode or (
        selected == "wna8" and not supports_compute_mode(encoding, operation, "wna8")
    ):
        raise ValueError("format selected an undeclared packed compute mode")
    return selected


def checkpoint_extension(method: str) -> QuantizationExtension:
    for extension in extensions():
        if method in extension.checkpoint_methods:
            return extension
    raise ValueError(f"unsupported checkpoint quantization method {method!r}")


def group_extension(format_id: str) -> QuantizationExtension:
    for extension in extensions():
        if format_id in extension.group_formats:
            return extension
    raise ValueError(f"unsupported compressed-tensors format {format_id!r}")
