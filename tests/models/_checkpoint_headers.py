"""Test-only safetensors header reader; tensor payloads are never materialized."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import cast

from safetensors import safe_open

from cubic_serve.models.weight_schema import TensorHeader


def read_index_headers(model_root: Path) -> tuple[TensorHeader, ...]:
    index = cast(
        dict[str, object],
        json.loads((model_root / "model.safetensors.index.json").read_text()),
    )
    raw_weight_map = cast(dict[str, object], index["weight_map"])
    by_shard: dict[str, list[str]] = defaultdict(list)
    for name, shard in raw_weight_map.items():
        if not isinstance(shard, str):
            raise TypeError("fixture checkpoint index shard must be a string")
        by_shard[shard].append(name)
    headers: list[TensorHeader] = []
    for shard, names in sorted(by_shard.items()):
        with safe_open(model_root / shard, framework="pt", device="cpu") as reader:
            for name in names:
                tensor = reader.get_slice(name)
                headers.append(
                    TensorHeader(
                        name=name,
                        dtype=tensor.get_dtype(),
                        shape=tuple(tensor.get_shape()),
                    )
                )
    return tuple(sorted(headers, key=lambda header: header.name))
