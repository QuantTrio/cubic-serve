from dataclasses import dataclass
from typing import ClassVar

from cubic_serve.quantization.common.marlin.parameters import MarlinParameters, validate_marlin
from cubic_serve.quantization.common.parameters import PackedLinearParameters, PackedMoEParameters


def supports_compute_mode(encoding, operation: str, mode: str) -> bool:
    if mode == "wna16":
        return True
    return (mode == "wna8" and operation in ("linear", "moe")
            and 2 <= encoding.bits <= 8)


@dataclass(frozen=True)
class IntegerLinearParameters(PackedLinearParameters, MarlinParameters):
    generic_geometry: ClassVar[bool] = False
    # Zero derives the legal search bounds from tensor/device resources.
    marlin_max_k_partitions: int = 0
    marlin_max_candidates: int = 6


@dataclass(frozen=True)
class IntegerMoEParameters(PackedMoEParameters, MarlinParameters):
    marlin_max_candidates: int = 6
    marlin_max_k_partitions: int = 0


# Format-owned output arithmetic: local stored route/shared weighting, then TP completion.
LOCAL_WEIGHTED_TP_COMPLETION = True


PARAMETERS = {"linear": IntegerLinearParameters, "moe": IntegerMoEParameters}
FORBIDDEN_OVERRIDES: dict[str, str] = {}


def validate(operation: str, values: object) -> tuple[tuple[str, int | bool | float | str], ...]:
    custom: dict[str, int | float] = {}
    if operation in ("linear", "moe") and isinstance(values, dict):
        values = dict(values)
        for key in ("marlin_max_k_partitions",):
            if key in values:
                value = values.pop(key)
                if type(value) is not int or value < 0:
                    raise ValueError("marlin_max_k_partitions must be nonnegative (zero is auto)")
                custom[key] = value
    result = validate_marlin(
        PackedLinearParameters if operation == "linear" else PackedMoEParameters,
        values,
        FORBIDDEN_OVERRIDES,
        min_block_m=8,
    )
    return tuple(sorted((*result, *custom.items())))
