"""Checkpoint declarations and validation owned by this extension."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    from cubic_serve.models.quantization import QuantizationScheme

from collections.abc import Mapping

from cubic_serve.models.quantization import (
    ActivationFormat,
    QuantizationError,
    QuantizationScheme,
    WeightFormat,
)


def parse_config(config: Mapping[str, object]):
    from fnmatch import translate

    from cubic_serve.models.quantization import _strings
    from cubic_serve.quantization.common.checkpoint import parse_groups

    if config.get("quant_algo") != "NVFP4":
        raise QuantizationError("ModelOpt entry requires declared NVFP4 W4A4 semantics")

    def parse(group, weights):
        for label, declaration in (
            ("weight", weights),
            ("activation", group.get("input_activations")),
        ):
            if not isinstance(declaration, Mapping) or any(
                declaration.get(key) != value
                for key, value in (
                    ("num_bits", 4),
                    ("group_size", 16),
                    ("type", "float"),
                    ("dynamic", False),
                )
            ):
                raise QuantizationError(f"unsupported ModelOpt NVFP4 {label} declaration")
        return QuantizationScheme(
            WeightFormat.NVFP4, 4, 16, metadata="modelopt", activation=ActivationFormat.NVFP4_LOCAL
        )

    # ModelOpt uses shell-style patterns, whereas the common contract uses
    # explicit regex rules. Keep this exporter syntax inside its parser.
    ignored = tuple("re:" + translate(pattern) for pattern in _strings(config.get("ignore", [])))
    return parse_groups(config, parse), ignored


def parse_group(group: Mapping[str, object], weights: Mapping[str, object]) -> QuantizationScheme:
    activation = group.get("input_activations")
    if weights.get("dynamic") is not False or weights.get("symmetric") is not True:
        raise QuantizationError("checkpoint weights require static symmetric encoding")
    if (
        weights.get("num_bits") != 4
        or weights.get("group_size") != 16
        or weights.get("strategy") != "tensor_group"
        or weights.get("scale_dtype") != "torch.float8_e4m3fn"
    ):
        raise QuantizationError("NVFP4 packed weight declaration differs")
    mode = ActivationFormat.IDENTITY
    if activation is not None:
        if (
            not isinstance(activation, Mapping)
            or activation.get("dynamic") != "local"
            or activation.get("num_bits") != 4
            or activation.get("group_size") != 16
            or activation.get("strategy") != "tensor_group"
            or activation.get("symmetric") is not True
        ):
            raise QuantizationError("unsupported NVFP4 activation declaration")
        mode = ActivationFormat.NVFP4_LOCAL
    return QuantizationScheme(WeightFormat.NVFP4, 4, 16, activation=mode)


def validate_scheme(self: QuantizationScheme) -> None:
    if self.metadata not in ("", "modelopt"):
        raise QuantizationError("unknown NVFP4 serialization convention")
    if (self.bits, self.group_in, self.group_out) != (
        4,
        16,
        1,
    ):
        raise QuantizationError("NVFP4 encoding requires 4-bit groups of 16")

    if self.activation not in (ActivationFormat.IDENTITY, ActivationFormat.NVFP4_LOCAL):
        raise QuantizationError("activation encoding is incompatible with weight format")
