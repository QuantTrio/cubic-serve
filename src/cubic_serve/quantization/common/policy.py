"""Shared validation for the existing packed tile schema, not a format selector."""

from __future__ import annotations

from typing_extensions import TypeVar

from dataclasses import fields
from typing import TYPE_CHECKING, Any

from cubic_serve.quantization.common.parameters import PackedMatmulParameters

if TYPE_CHECKING:
    import torch

def validate_tiles(
    schema: type[Any], values: object, forbidden: dict[str, str]
) -> tuple[tuple[str, int | bool | float | str], ...]:
    if not isinstance(values, dict):
        raise ValueError("packed parameters must be an object")
    known = {field.name for field in fields(schema)}
    for key, value in values.items():
        if key in forbidden:
            raise ValueError(forbidden[key])
        if key == "grouped_projections" and key in known:
            if type(value) is not bool:
                raise ValueError("grouped_projections must be boolean")
            continue
        if key not in known or type(value) is not int:
            raise ValueError(f"invalid packed parameter: {key}")
        if key.endswith("warps"):
            valid = value in (4, 8)
        elif key == "fixed_stages":
            valid = 1 <= value <= 4
        elif key.endswith("block_m"):
            valid = value in (16, 32, 64, 128, 256)
        else:
            valid = value in (16, 32, 64, 128, 256)
        if not valid:
            raise ValueError(f"unsupported packed parameter {key}={value}")
    return tuple(sorted(values.items()))


Parameters = TypeVar("Parameters", bound=PackedMatmulParameters)


def placement_candidates(
    parameters: Parameters,
    *,
    family: str,
    device: torch.device,
    rows: int,
    columns: int,
    reduction: int,
    element_bytes: int,
    groups: int = 1,
    override_fields: tuple[str, ...] = (
        "fixed_block_m",
        "fixed_block_n",
        "fixed_warps",
        "fixed_stages",
    ),
    **scope: int,
) -> tuple[Parameters, ...]:
    """Reuse resource geometry; the format's K and arithmetic remain fixed.

    Imports are lazy so loading extension metadata does not initialize CUDA.
    Explicit overrides bind the same fields as ordinary linear/MoE planning.
    """
    from dataclasses import replace

    from cubic_serve.ops.cuda.launch_cost import (
        MatmulGeometry,
        device_resources,
        matmul_geometries,
    )
    from cubic_serve.ops.strategies import parameter_overrides

    explicit = parameter_overrides(family, n=columns, k=reduction, **scope)
    names = ("block_m", "block_n", "warps", "stages")
    keys = tuple(
        zip(names, ("fixed_block_m", "fixed_block_n", "fixed_warps", "fixed_stages"), strict=True)
    )
    points = matmul_geometries(
        device_resources(device),
        rows=rows,
        columns=(columns,),
        reduction=reduction,
        element_bytes=element_bytes,
        groups=groups,
        baseline=MatmulGeometry(
            parameters.fixed_block_m,
            parameters.fixed_block_n,
            parameters.fixed_block_k,
            parameters.fixed_warps,
            parameters.fixed_stages,
        ),
        locked=frozenset(
            name for name, key in zip(names, override_fields, strict=True) if key in explicit
        ),
    )
    return tuple(
        replace(parameters, **{key: getattr(p, name) for name, key in keys}) for p in points
    )
