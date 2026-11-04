"""Checkpoint declarations and validation owned by this extension."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    from cubic_serve.models.quantization import EncodingRule, QuantizationScheme

from cubic_serve.models.quantization import (
    ActivationFormat,
    EncodingRule,
    QuantizationError,
    QuantizationScheme,
    WeightFormat,
    _positive_int,
    _strings,
)


def parse_config(config: Mapping[str, object]) -> tuple[tuple[EncodingRule, ...], tuple[str, ...]]:
    ignored = _strings(config.get("modules_to_not_convert", []))
    rules = []
    if config.get("lm_head", False) is not True:
        ignored = (*ignored, "lm_head")
    version = config.get("version", "gemm")
    if config.get("bits") != 4 or not isinstance(version, str) or version.lower() != "gemm":
        raise QuantizationError("only AWQ 4-bit GEMM packing is admitted")
    if config.get("zero_point") is not True:
        raise QuantizationError("AWQ GEMM requires explicit zero points")
    rules.append(
        EncodingRule(
            "awq",
            ("Linear",),
            QuantizationScheme(WeightFormat.AWQ, 4, _positive_int(config.get("group_size"))),
        )
    )

    return tuple(rules), ignored


def validate_scheme(self: QuantizationScheme) -> None:
    if self.bits != 4 or self.group_out != 1:
        raise QuantizationError("AWQ GEMM requires 4-bit, input-axis groups")

    if self.activation not in (ActivationFormat.IDENTITY,):
        raise QuantizationError("activation encoding is incompatible with weight format")
