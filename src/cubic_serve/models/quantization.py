"""Immutable checkpoint encoding contracts, independent of CUDA implementation."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from cubic_serve._compat import StrEnum
from functools import lru_cache


class QuantizationError(ValueError):
    """Checkpoint encoding or consumer semantics are not closed."""


class WeightFormat(StrEnum):
    AWQ = "awq-gemm-v1"
    FP8 = "fp8-e4m3-channel-v1"
    NVFP4 = "nvfp4-compressed-tensors-v1"
    CUBIC = "cubic-bitstream-v1"


class ActivationFormat(StrEnum):
    IDENTITY = "identity"
    FP8_TOKEN = "fp8-e4m3-dynamic-token"
    NVFP4_LOCAL = "nvfp4-dynamic-local"


@dataclass(frozen=True, slots=True)
class QuantizationScheme:
    format: WeightFormat | str
    bits: int
    group_in: int
    group_out: int = 1
    metadata: str = ""
    activation: ActivationFormat | str = ActivationFormat.IDENTITY

    def __post_init__(self) -> None:
        if (
            not isinstance(self.format, str)
            or not self.format
            or not isinstance(self.activation, str)
            or not self.activation
        ):
            raise QuantizationError("encoding requires named weight and activation formats")
        if any(type(v) is not int or v <= 0 for v in (self.bits, self.group_in, self.group_out)):
            raise QuantizationError("bit width and quantization groups must be positive integers")
        # Normalize builtin aliases; registered extensions may introduce other IDs.
        with suppress(ValueError):
            object.__setattr__(self, "activation", ActivationFormat(self.activation))
        from cubic_serve.quantization.registry import get_extension

        get_extension(self.format).module("checkpoint").validate_scheme(self)


@dataclass(frozen=True, slots=True)
class EncodingRule:
    name: str
    targets: tuple[str, ...]
    encoding: QuantizationScheme


@dataclass(frozen=True, slots=True)
class QuantizationConfig:
    """Exporter rules normalized once; model-owned aliases are explicit inputs."""

    method: str
    rules: tuple[EncodingRule, ...]
    ignored: tuple[str, ...]
    source_scope: str
    digest: str
    ignore_substrings: bool = False
    target_subtrees: bool = False

    @classmethod
    def from_model_config(cls, raw: Mapping[str, object]) -> QuantizationConfig | None:
        root = raw.get("quantization_config")
        text = raw.get("text_config", {})
        nested = text.get("quantization_config") if isinstance(text, Mapping) else None
        if root is not None and nested is not None and root != nested:
            raise QuantizationError("conflicting root and text quantization declarations")
        config = root if root is not None else nested
        if config is None:
            return None
        if not isinstance(config, Mapping):
            raise QuantizationError("quantization_config must be an object")
        method = config.get("quant_method")
        if not isinstance(method, str):
            raise QuantizationError("checkpoint quantization method must be a string")
        if config.get("sparsity_config") or config.get("transform_config"):
            raise QuantizationError("checkpoint sparsity and input transforms are not implemented")
        from cubic_serve.quantization.common.checkpoint import parse_groups
        from cubic_serve.quantization.registry import checkpoint_extension, group_extension

        if method == "compressed-tensors":

            def parse(
                group: Mapping[str, object], weights: Mapping[str, object]
            ) -> QuantizationScheme:
                fmt = group.get("format") or config.get("format")
                if not isinstance(fmt, str):
                    raise QuantizationError("compressed-tensors group requires a format")
                return group_extension(fmt).module("checkpoint").parse_group(group, weights)

            rules = parse_groups(config, parse)
            ignored = _strings(config.get("ignore", []))
        else:
            extension = checkpoint_extension(method)
            rules, ignored = extension.module("checkpoint").parse_config(config)
        encoded = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return cls(
            method,
            tuple(rules),
            ignored,
            "root" if root is not None else "text_config",
            "b2s256:" + hashlib.blake2s(encoded.encode()).hexdigest(),
            ignore_substrings=False
            if method == "compressed-tensors"
            else extension.ignore_substrings,
            target_subtrees=method == "compressed-tensors",
        )

    def resolve(
        self, module: str, *, aliases: tuple[str, ...] = (), operation: str = "linear"
    ) -> QuantizationScheme | None:
        if operation not in ("linear", "embedding", "routed_moe"):
            raise QuantizationError(f"unknown quantized consumer {operation!r}")
        names = (module, *aliases)
        if self.source_scope == "text_config" and module.startswith("model.language_model."):
            # Nested text exporters target the text model's namespace. Preserve
            # fused projection aliases there; do not let stale wrapper-qualified
            # rules override a rule for the actual text-model consumer.
            names = tuple(name.replace("model.language_model.", "model.") for name in names)
        if _may_match(self.ignored, names, full=False) and any(
            _ignored(target, name, awq=self.ignore_substrings)
            for target in self.ignored
            for name in names
        ):
            return None
        matches: list[tuple[tuple[int, int, int, int], QuantizationScheme]] = []
        for rule in self.rules:
            if not _may_match(rule.targets, names, full=True):
                continue
            for target in rule.targets:
                for name in names:
                    score = _match(target, name, operation, subtree=self.target_subtrees)
                    if score is not None:
                        matches.append((score, rule.encoding))
        if not matches:
            return None
        best = max(score for score, _ in matches)
        encodings = {encoding for score, encoding in matches if score == best}
        if len(encodings) != 1:
            raise QuantizationError(f"conflicting encoding rules for {module!r}")
        encoding = encodings.pop()
        if operation == "embedding" and encoding.activation is not ActivationFormat.IDENTITY:
            # Exporter groups can include embedding alongside linear modules.
            # Lookup consumes integer IDs, not a floating activation matrix.
            from dataclasses import replace

            encoding = replace(encoding, activation=ActivationFormat.IDENTITY)
        return encoding


def _strings(value: object) -> tuple[str, ...]:
    if not isinstance(value, list | tuple) or any(not isinstance(v, str) or not v for v in value):
        raise QuantizationError("targets and ignore rules must be non-empty strings")
    for entry in value:
        if entry.startswith("re:"):
            try:
                re.compile(entry[3:])
            except re.error as error:
                raise QuantizationError("invalid quantization target expression") from error
    return tuple(value)


def _positive_int(value: object) -> int:
    if type(value) is not int or value <= 0:
        raise QuantizationError("quantization group size must be a positive integer")
    return value


@lru_cache(maxsize=4096)
def _target_pattern(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern)


@lru_cache(maxsize=128)
def _target_union(targets: tuple[str, ...]) -> re.Pattern[str] | None:
    """A rejection-only prefilter; original rules still decide precedence."""
    if not targets or any(not target.startswith("re:") for target in targets):
        return None
    patterns = tuple(_target_pattern(target[3:]) for target in targets)
    # Captures/backreferences and global inline flags must retain their original
    # regex scope. Unsupported unions simply keep individual matching.
    if any(pattern.groups for pattern in patterns):
        return None
    try:
        return re.compile("|".join(f"(?:{pattern.pattern})" for pattern in patterns))
    except re.error:
        return None


def _may_match(targets: tuple[str, ...], names: tuple[str, ...], *, full: bool) -> bool:
    pattern = _target_union(targets)
    if pattern is None:
        return True
    if full:
        for name in names:
            if name.startswith("re:"):
                return True
    match = pattern.fullmatch if full else pattern.match
    for name in names:
        if match(name) is not None or match("." + name) is not None:
            return True
    return False


def _ignored(target: str, name: str, *, awq: bool) -> bool:
    if target.startswith("re:"):
        pattern = _target_pattern(target[3:])
        return pattern.match(name) is not None or pattern.match("." + name) is not None
    if awq:
        return target in name
    return target == name


def _match(
    target: str, name: str, operation: str, *, subtree: bool
) -> tuple[int, int, int, int] | None:
    if target == name:
        return (3, 0, 0, len(target))
    if subtree and name.startswith(target + "."):
        return (3, 0, 0, len(target))
    if target.startswith("re:"):
        pattern = target[3:]
        compiled = _target_pattern(pattern)
        if compiled.fullmatch(name) is not None or compiled.fullmatch("." + name) is not None:
            return (2, int(pattern.startswith("^")), -pattern.count(".*"), len(pattern))
    if (
        target
        in {
            "linear": ("Linear",),
            "embedding": ("Embedding",),
            "routed_moe": ("FusedMoE", "RoutedExperts"),
        }[operation]
    ):
        return (1, 0, 0, 0)
    return None
