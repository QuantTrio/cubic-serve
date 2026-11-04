"""Grouped container syntax and row coordinates shared by actual extensions."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from cubic_serve.models.quantization import EncodingRule, QuantizationScheme


from collections.abc import Mapping

from cubic_serve.models.quantization import EncodingRule, QuantizationError, _strings


def parse_groups(
    config: Mapping[str, object],
    parse: Callable[[Mapping[str, object], Mapping[str, object]], QuantizationScheme],
) -> tuple[EncodingRule, ...]:
    groups = config.get("config_groups")
    if not isinstance(groups, Mapping) or not groups:
        raise QuantizationError("quantized checkpoint requires config_groups")
    rules = []
    for name, group in groups.items():
        if not isinstance(name, str) or not isinstance(group, Mapping):
            raise QuantizationError("invalid quantization group")
        targets = _strings(group.get("targets"))
        if not targets:
            raise QuantizationError("encoding rule requires targets")
        weights = group.get("weights")
        if not isinstance(weights, Mapping):
            raise QuantizationError("encoding rule requires weight semantics")
        if group.get("output_activations") is not None:
            raise QuantizationError("output activation quantization is not implemented")
        rules.append(EncodingRule(name, targets, parse(group, weights)))
    return tuple(rules)
