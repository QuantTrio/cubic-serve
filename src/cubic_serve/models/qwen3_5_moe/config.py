"""Configuration for checkpoints whose ``model_type`` is ``qwen3_5_moe``."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from cubic_serve.models.qwen3_5.model import (
    ModelIdentity,
    Qwen3_5ArchitecturePlan,
    Qwen3_5ConfigError,
    Qwen3_5GraphConfig,
    Qwen3_5GraphVariant,
    Qwen3_5TextConfig,
    _parse_json_object,
    _parse_qwen3_5_graph,
)


@dataclass(frozen=True, slots=True)
class Qwen3_5MoeConfig:
    """Qwen3.5 MoE architecture with an explicit loader key."""

    identity: ModelIdentity
    graph: Qwen3_5GraphConfig

    @property
    def text_config(self) -> Qwen3_5TextConfig:
        """Expose the original text schema without copying or changing policy."""
        return self.graph.text

    def architecture_plan(self) -> Qwen3_5ArchitecturePlan:
        return self.graph.architecture_plan()


def parse_qwen3_5_moe_config(raw: Mapping[str, object]) -> Qwen3_5MoeConfig:
    graph = _parse_qwen3_5_graph(raw, expected_variant=Qwen3_5GraphVariant.MOE)
    moe = graph.text.moe
    assert moe is not None
    if moe.dense_layer_indexes or moe.sparse_step != 1:
        raise Qwen3_5ConfigError("qwen3_5_moe execution requires routed experts in every layer")
    if not moe.normalize_topk_probability:
        raise Qwen3_5ConfigError("qwen3_5_moe router requires normalized selected probabilities")
    if not moe.shared_expert_intermediate_size:
        raise Qwen3_5ConfigError("qwen3_5_moe execution requires a shared expert")
    return Qwen3_5MoeConfig(
        identity=ModelIdentity(
            release_family="qwen3_5",
            release_profile="qwen3_5_moe",
            model_type="qwen3_5_moe",
            graph_family="qwen3_5",
            architecture=graph.architecture,
        ),
        graph=graph,
    )


def parse_qwen3_5_moe_config_json(document: str | bytes | bytearray) -> Qwen3_5MoeConfig:
    return parse_qwen3_5_moe_config(_parse_json_object(document))
