"""Immutable, startup-only operator parameters; no model-specific kernel code."""

from __future__ import annotations

from typing_extensions import TypeAliasType

import hashlib
import json
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, cast

from cubic_serve.quantization.api import PolicyContract
from cubic_serve.quantization.common.parameters import (
    PackedLinearParameters as PackedLinearParameters,
)
from cubic_serve.quantization.common.parameters import (
    PackedMatmulParameters as PackedMatmulParameters,
)
from cubic_serve.quantization.common.parameters import (
    PackedMoEParameters as PackedMoEParameters,
)
from cubic_serve.quantization.registry import extensions


@dataclass(frozen=True)
class LaunchParameters:
    # Portable search/resource limits, not model or concurrency cutoffs.
    shared_expert_overlap: bool = True
    projection_group_overlap: bool = True
    branch_fork_join_launches: int = 1
    max_candidates: int = 4
    max_block_m: int = 64
    max_block_n: int = 256
    max_warps: int = 8
    max_stages: int = 4
    calibration_samples: int = 3
    micro_calibration_replays: int = 32
    register_reserve: int = 16


@dataclass(frozen=True)
class LinearParameters:
    fixed_block_m: int = 16
    fixed_max_block_m: int = 32
    fixed_block_n: int = 64
    fixed_block_k: int = 64
    fixed_warps: int = 4
    fixed_stages: int = 3
    overlap_block_m: int = 64
    overlap_block_n: int = 128
    canonical_max_rows: int = 8
    canonical_warps: int = 8
    grouped_projections: bool = True
    grouped_vector_tile_fill_percent: int = 50
    grouped_vector_min_k: int = 2048
    fixed_max_k_partitions: int = 16
    fixed_cta_supply_factor: int = 2


@dataclass(frozen=True)
class MoEParameters:
    block_m: int = 16
    block_n: int = 64
    block_k: int = 128
    compact_down_block_n: int = 128
    gemm_warps: int = 4
    gemm_stages: int = 3
    compact_min_routes_per_expert: int = 1
    # Price the extra integer deduplication against the expert-major CTA count.
    # Zero disables route-indexed sparse work; no model/device/C lookup table.
    sparse_route_grid_percent: int = 100

    def route_indexed(self, rows: int, experts: int, top_k: int, row_tile: int) -> bool:
        expert_blocks = experts * ((rows + row_tile - 1) // row_tile)
        return rows * top_k * 100 < expert_blocks * self.sparse_route_grid_percent


@dataclass(frozen=True)
class GDNParameters:
    # Independent value rows; never changes the K reduction or state boundary.
    native_value_tile: int = 64
    # Compute token/head coefficients once across independent value stripes.
    reuse_coefficients: bool = True


@dataclass(frozen=True)
class CollectiveParameters:
    # Extra exchange startup expressed as equivalent transferred bytes.
    exchange_latency_bytes: int = 1024 * 1024
    embedding_host_mirror: bool = True
    host_sum: bool = True
    host_sum_partition: bool = True
    host_sum_threads: int = 128
    transport_calibration: bool = True
    transport_calibration_bytes: int = 256 * 1024
    transport_calibration_replays: int = 10
    transport_min_gain_percent: int = 5
    transport_max_fit_error_percent: int = 20
    # Ordered route completion: cold fit budgets and acceptance margins, not
    # model/device/concurrency points. Tensor/group geometry determines traffic.
    route_completion: bool = True
    route_completion_calibration_bytes: int = 256 * 1024
    route_completion_calibration_replays: int = 10
    route_completion_min_gain_percent: int = 5
    route_completion_max_fit_error_percent: int = 20
    # Startup search limits, not concurrency/model/device thresholds.
    decode_overlap_max_parts: int = 4
    decode_overlap_samples: int = 3
    decode_overlap_min_gain_percent: int = 5
    decode_overlap_max_bytes: int = 4 * 1024 * 1024 * 1024

    def overlap_partitions(self, rows: int, row_tiles: int) -> tuple[tuple[int, ...], ...]:
        """Partition existing power-of-two capacity without extra physical rows.

        Split the largest part first. Three parts are legal (e.g. 128+64+64).
        A child cannot be smaller than the largest producer's row tile, so
        this prefilter never assumes that splitting one CTA saves work.
        Actual startup timing must still account for layout/launch changes.
        """
        if rows < 1 or rows & (rows - 1) or row_tiles < 1:
            raise ValueError("overlap requires power-of-two rows and positive tile count")
        minimum = (rows + row_tiles - 1) // row_tiles
        parts = [rows]
        candidates = []
        while len(parts) < self.decode_overlap_max_parts:
            largest = max(parts)
            if largest // 2 < minimum:
                break
            parts.remove(largest)
            parts.extend((largest // 2, largest // 2))
            parts.sort(reverse=True)
            candidates.append(tuple(parts))
        return tuple(candidates)

    def partitioned_sum(self, payload_bytes: int, ranks: int) -> bool:
        if payload_bytes <= 0 or ranks <= 0:
            raise ValueError("collective cost requires positive payload and group size")
        # All-gather: (R-1)*B. Exchange + completed-partition gather:
        # 2*(R-1)*B/R, plus one extra collective startup. Integer comparison
        # avoids a size-band cliff and preserves canonical source-rank math.
        return payload_bytes * (ranks - 1) * (ranks - 2) > self.exchange_latency_bytes * ranks


OperatorParameters = TypeAliasType("OperatorParameters", LaunchParameters
    | LinearParameters
    | MoEParameters
    | PackedLinearParameters
    | PackedMoEParameters
    | GDNParameters
    | CollectiveParameters)


_TYPES = {
    "launch": LaunchParameters,
    "linear": LinearParameters,
    "moe": MoEParameters,
    "gdn": GDNParameters,
    "collective": CollectiveParameters,
}
_SHAPES = {
    "launch": set(),
    "linear": {"n", "k"},
    "moe": {"hidden", "intermediate", "experts", "global_experts", "top_k"},
    "gdn": {"q_heads", "value_heads", "key_dim", "value_dim"},
    "collective": set(),
}
_EXTENSION_POLICIES: dict[str, tuple[PolicyContract, str]] = {}


def _load_extension_parameters() -> None:
    for extension in extensions():
        policy = extension.module("policy")
        for operation, schema in policy.PARAMETERS.items():
            family = extension.family(operation)
            _TYPES[family] = schema
            _SHAPES[family] = (
                {"n", "k"} if operation == "linear" else {"n", "k", "experts", "top_k"}
            )
            _EXTENSION_POLICIES[family] = policy, operation


_load_extension_parameters()


def _parameters(family: str, values: object) -> tuple[tuple[str, int | bool | float | str], ...]:
    if not isinstance(values, dict):
        raise ValueError(f"{family} parameters must be an object")
    _load_extension_parameters()
    if family in _EXTENSION_POLICIES:
        policy, operation = _EXTENSION_POLICIES[family]
        return policy.validate(operation, values)
    known = {f.name for f in fields(_TYPES[family])}
    for key, value in values.items():
        if (key == "grouped_projections" and family == "linear") or (
            key == "reuse_coefficients" and family == "gdn"
        ) or (
            key in ("route_completion", "transport_calibration", "host_sum", "host_sum_partition", "embedding_host_mirror")
            and family == "collective"
        ) or (
            key in ("shared_expert_overlap", "projection_group_overlap") and family == "launch"
        ):
            if type(value) is not bool:
                raise ValueError(f"{key} must be a boolean")
            continue
        if key not in known or type(value) is not int:
            raise ValueError(f"invalid {family} parameter: {key}")
        if family == "launch":
            valid = value >= 1
            if key in ("max_block_m", "max_block_n"):
                valid = value >= 16 and value & (value - 1) == 0
            elif key == "max_warps":
                valid = value in (4, 8)
            elif key == "max_stages":
                valid = 1 <= value <= 4
            elif key == "register_reserve":
                valid = value >= 0
        elif family == "collective":
            valid = value >= 0
            if key == "host_sum_threads":
                valid = value in (64, 128, 256)
            if key in (
                "decode_overlap_max_parts", "decode_overlap_samples",
                "route_completion_calibration_bytes", "route_completion_calibration_replays",
                "transport_calibration_bytes", "transport_calibration_replays",
            ):
                valid = value >= 1
            elif key in ("route_completion_min_gain_percent",
                         "route_completion_max_fit_error_percent",
                         "transport_min_gain_percent", "transport_max_fit_error_percent"):
                valid = 0 <= value < 100
        elif family == "gdn" and key == "native_value_tile":
            valid = value in (16, 32, 64, 128)
        elif key.endswith("warps"):
            valid = value in (4, 8)
        elif key in ("fixed_stages", "gemm_stages"):
            valid = 1 <= value <= 4
        elif key == "compact_min_routes_per_expert":
            valid = value >= 1
        elif family == "moe" and key == "sparse_route_grid_percent":
            valid = value >= 0
        elif key == "canonical_max_rows":
            valid = value in (1, 2, 4, 8)
        elif family == "linear" and key in ("fixed_max_k_partitions", "fixed_cta_supply_factor"):
            valid = value >= 1
        elif family == "linear" and key == "grouped_vector_tile_fill_percent":
            valid = 0 <= value <= 100
        elif family == "linear" and key == "grouped_vector_min_k":
            valid = value >= 1
        else:
            valid = value in (16, 32, 64, 128, 256)
            if key.endswith("block_m"):
                valid = value in (16, 32, 64)
        if not valid:
            raise ValueError(f"unsupported {family} parameter {key}={value}")
    return tuple(sorted(values.items()))


@dataclass(frozen=True)
class _Profile:
    family: str
    device: str | None
    model: str | None
    shape: tuple[tuple[str, int], ...]
    parameters: tuple[tuple[str, int | bool | float | str], ...]

    @property
    def priority(self) -> int:
        return int(self.device is not None) + 2 * int(self.model is not None)


@dataclass(frozen=True)
class OperatorConfig:
    profiles: tuple[_Profile, ...] = ()
    overrides: tuple[tuple[str, tuple[tuple[str, int | bool | float | str], ...]], ...] = ()
    activation_dtype: str = "auto"
    softmax_dtype: str = "fp32"
    attention_dtype: str = "auto"
    compute_mode: str = "wna16"

    def __post_init__(self) -> None:
        if self.compute_mode not in ("wna16", "wna8"):
            raise ValueError("compute_mode must be wna16 or wna8")
        if self.attention_dtype not in ("auto", "fp32"):
            raise ValueError("attention_dtype must be auto or fp32")
        if self.activation_dtype not in ("auto", "fp32"):
            raise ValueError("activation_dtype must be auto or fp32")
        if self.softmax_dtype not in ("auto", "fp32"):
            raise ValueError("softmax_dtype must be auto or fp32")

    @classmethod
    def from_dict(cls, payload: object) -> OperatorConfig:
        _load_extension_parameters()
        if not isinstance(payload, dict) or set(payload) - {
            "version",
            "profiles",
            "overrides",
            "activation_dtype",
            "softmax_dtype",
            "attention_dtype",
            "compute_mode",
        }:
            raise ValueError("operator config requires version, profiles and/or overrides")
        if type(payload.get("version")) is not int or payload["version"] != 1:
            raise ValueError("operator config version must be 1")
        profiles = payload.get("profiles", [])
        if not isinstance(profiles, list):
            raise ValueError("operator profiles must be a list")
        parsed = []
        for profile in profiles:
            if not isinstance(profile, dict) or set(profile) - {
                "operator",
                "device",
                "model",
                "shape",
                "parameters",
            }:
                raise ValueError("invalid operator profile fields")
            family = profile.get("operator")
            if not isinstance(family, str) or family not in _TYPES:
                raise ValueError(f"unknown operator family; expected one of {tuple(_TYPES)}")
            for scope in ("device", "model"):
                if scope in profile and (
                    not isinstance(profile[scope], str) or not profile[scope].strip()
                ):
                    raise ValueError(f"{scope} must be a nonempty exact identity")
            shape = profile.get("shape", {})
            if not isinstance(shape, dict) or set(shape) - _SHAPES[family]:
                raise ValueError(
                    f"invalid {family} shape; M-dependent reduction selection is forbidden"
                )
            if any(type(v) is not int or v <= 0 for v in shape.values()):
                raise ValueError("shape dimensions must be positive integers")
            parsed.append(
                _Profile(
                    family,
                    profile.get("device"),
                    profile.get("model"),
                    tuple(sorted(shape.items())),
                    _parameters(family, profile.get("parameters", {})),
                )
            )
        overrides = payload.get("overrides", {})
        if not isinstance(overrides, dict) or set(overrides) - _TYPES.keys():
            raise ValueError("invalid operator overrides")
        return cls(
            tuple(sorted(parsed, key=lambda p: p.priority)),
            tuple((key, _parameters(key, value)) for key, value in sorted(overrides.items())),
            activation_dtype=payload.get("activation_dtype", "auto"),
            softmax_dtype=payload.get("softmax_dtype", "fp32"),
            attention_dtype=payload.get("attention_dtype", "auto"),
            compute_mode=payload.get("compute_mode", "wna16"),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> OperatorConfig:
        def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate operator config key: {key}")
                result[key] = value
            return result

        return cls.from_dict(json.loads(Path(path).read_text(), object_pairs_hook=unique))

    @property
    def digest(self) -> str:
        return hashlib.blake2s(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()

    def resolve(self, family: str, *, device: str, model: str, **shape: int) -> OperatorParameters:
        _load_extension_parameters()
        if (
            family not in _TYPES
            or set(shape) != _SHAPES[family]
            or any(type(v) is not int or v <= 0 for v in shape.values())
        ):
            raise ValueError("operator binding requires its complete positive tensor shape")
        result: OperatorParameters = _TYPES[family]()
        for profile in self.profiles:
            if (
                profile.family == family
                and profile.device in (None, device)
                and profile.model in (None, model)
                and all(shape[k] == v for k, v in profile.shape)
            ):
                result = cast(
                    OperatorParameters, replace(cast(Any, result), **dict(profile.parameters))
                )
        for key, values in self.overrides:
            if key == family:
                result = cast(OperatorParameters, replace(cast(Any, result), **dict(values)))
        return result


_CONTEXT: ContextVar[tuple[OperatorConfig, str, str]] = ContextVar(
    "operator_strategies", default=(OperatorConfig(), "", "")
)


def compute_mode() -> str:
    """Arithmetic mode shared by the independent oracle's scoped forward."""
    return _CONTEXT.get()[0].compute_mode


@contextmanager
def operator_strategy_scope(config: OperatorConfig, *, device: str, model: str) -> Iterator[None]:
    """Bind once before graph construction; never read files during replay."""
    token = _CONTEXT.set((config, device, model))
    try:
        yield
    finally:
        _CONTEXT.reset(token)


def collective_thread_config(config: OperatorConfig, threads: int | None) -> OperatorConfig:
    """Copy only a calibrated launch resource; preserve all arithmetic policies."""
    if threads is None:
        return config
    if type(threads) is not int or threads not in (64, 128, 256):
        raise ValueError("host sum threads must be 64, 128 or 256")
    values = dict(config.overrides)
    collective = dict(values.get("collective", ()))
    collective["host_sum_threads"] = threads
    values["collective"] = tuple(collective.items())
    return replace(config, overrides=tuple(values.items()))


@contextmanager
def collective_thread_scope(threads: int | None) -> Iterator[None]:
    config, device, model = _CONTEXT.get()
    with operator_strategy_scope(
        collective_thread_config(config, threads), device=device, model=model,
    ):
        yield


def overlap_collective_threads() -> tuple[int | None, ...]:
    """Bounded resource candidates, selected by complete paired round cost.

    Wider CTAs cover a fixed payload with fewer blocks, potentially leaving
    more SMs for a concurrent producer. Isolated collective latency cannot
    price that interference. Explicit resource overrides remain authoritative.
    """
    policy = collective_parameters()
    if not policy.host_sum or "host_sum_threads" in parameter_overrides("collective"):
        return (None,)
    wider = next((t for t in (64, 128, 256) if t > policy.host_sum_threads), None)
    return (None, wider) if wider is not None else (None,)


def linear_row_tile(captured_m: int, minimum: int, maximum: int) -> int:
    """Reuse weights across available rows without changing the K reduction.

    The minimum remains an explicit fixed-tile override. The maximum permits
    wider row tiles only when the captured input has those rows available.
    Neither checkpoint identity nor device generation participates.
    """
    if type(captured_m) is not int or captured_m <= 0:
        raise ValueError("captured M must be a positive integer")
    if minimum not in (16, 32, 64) or maximum not in (16, 32, 64):
        raise ValueError("linear row tiles must be 16, 32 or 64")
    return max(minimum, min(maximum, 1 << (captured_m.bit_length() - 1)))


def linear_parameters(*, n: int, k: int) -> LinearParameters:
    config, device, model = _CONTEXT.get()
    result = config.resolve("linear", device=device, model=model, n=n, k=k)
    assert isinstance(result, LinearParameters)
    return result


def launch_parameters() -> LaunchParameters:
    config, device, model = _CONTEXT.get()
    result = config.resolve("launch", device=device, model=model)
    assert isinstance(result, LaunchParameters)
    return result


def parameter_overrides(family: str, **shape: int) -> frozenset[str]:
    """Keep explicit operator choices authoritative over automatic geometry."""
    config, device, model = _CONTEXT.get()
    keys = set()
    for profile in config.profiles:
        if (profile.family == family and profile.device in (None, device)
                and profile.model in (None, model)
                and all(shape[k] == v for k, v in profile.shape)):
            keys.update(dict(profile.parameters))
    for key, values in config.overrides:
        if key == family:
            keys.update(dict(values))
    return frozenset(keys)


def linear_overlap_tiles(m: int, parameters: LinearParameters) -> tuple[tuple[int, int], ...]:
    """Two bounded placements; K, warps and pipeline stages remain unchanged."""
    base = (linear_row_tile(m, parameters.fixed_block_m, parameters.fixed_max_block_m),
            parameters.fixed_block_n)
    wider = (linear_row_tile(m, parameters.fixed_block_m,
                             max(parameters.fixed_max_block_m, parameters.overlap_block_m)),
             max(parameters.fixed_block_n, parameters.overlap_block_n))
    return tuple(dict.fromkeys((base, wider)))


def moe_parameters(
    *, hidden: int, intermediate: int, experts: int, global_experts: int, top_k: int
) -> MoEParameters:
    config, device, model = _CONTEXT.get()
    result = config.resolve(
        "moe",
        device=device,
        model=model,
        hidden=hidden,
        intermediate=intermediate,
        experts=experts,
        global_experts=global_experts,
        top_k=top_k,
    )
    assert isinstance(result, MoEParameters)
    return result


def activation_precision() -> str:
    """Return the startup arithmetic policy, independent of linear/MoE tuning."""
    return _CONTEXT.get()[0].activation_dtype


def collective_parameters() -> CollectiveParameters:
    config, device, model = _CONTEXT.get()
    result = config.resolve("collective", device=device, model=model)
    assert isinstance(result, CollectiveParameters)
    return result


def gdn_parameters(
    *, q_heads: int, value_heads: int, key_dim: int, value_dim: int
) -> GDNParameters:
    config, device, model = _CONTEXT.get()
    result = config.resolve(
        "gdn",
        device=device,
        model=model,
        q_heads=q_heads,
        value_heads=value_heads,
        key_dim=key_dim,
        value_dim=value_dim,
    )
    assert isinstance(result, GDNParameters)
    return result


def softmax_precision() -> str:
    """Auto inherits activation arithmetic; fp32 overrides only softmax."""
    config = _CONTEXT.get()[0]
    return config.activation_dtype if config.softmax_dtype == "auto" else config.softmax_dtype


def attention_precision() -> str:
    """Attention/GDN operands; FP32 accumulation and softmax are separate contracts."""
    return _CONTEXT.get()[0].attention_dtype
