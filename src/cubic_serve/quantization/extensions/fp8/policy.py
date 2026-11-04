"""FP8 retains token/weight scale semantics independently of launch geometry."""

from dataclasses import dataclass

from cubic_serve.quantization.common.marlin.parameters import MarlinParameters, validate_marlin
from cubic_serve.quantization.common.parameters import PackedLinearParameters, PackedMoEParameters
from cubic_serve.quantization.common.policy import validate_tiles


@dataclass(frozen=True)
class Fp8LinearParameters(PackedLinearParameters, MarlinParameters):
    grouped_projections: bool = True


PARAMETERS = {"linear": Fp8LinearParameters, "moe": PackedMoEParameters}
FORBIDDEN_OVERRIDES: dict[str, str] = {}


def validate(operation: str, values: object) -> tuple[tuple[str, int | bool | float | str], ...]:
    if operation == "linear":
        return validate_marlin(Fp8LinearParameters, values, FORBIDDEN_OVERRIDES)
    return validate_tiles(PARAMETERS[operation], values, FORBIDDEN_OVERRIDES)
