"""Minimal deterministic sampling primitives."""

from __future__ import annotations

import json
import secrets
from dataclasses import dataclass
from pathlib import Path

import torch

from cubic_serve.contracts.request import SamplingMeta
from cubic_serve.generation.constraints import LogitConstraint


@dataclass(frozen=True, slots=True)
class SamplingParameters:
    temperature: float
    top_p: float
    top_k: int
    seed: int

    def __post_init__(self) -> None:
        if (
            type(self.temperature) is not float
            or not 0.0 <= self.temperature <= 2.0
            or type(self.top_p) is not float
            or not 0.0 < self.top_p <= 1.0
            or type(self.top_k) is not int
            or not 1 <= self.top_k <= 20
            or type(self.seed) is not int
            or not 0 <= self.seed < 2**63
        ):
            raise ValueError("sampling parameters are outside the qualified domain")


GREEDY_SAMPLING = SamplingParameters(0.0, 1.0, 1, 0)


def load_eos_token_ids(checkpoint: Path, *, tokenizer_eos: int) -> tuple[int, ...]:
    """Keep all declared generation terminators, including the tokenizer EOS.

    Model and generation metadata may declare either one ID or a list. The
    lexical-history sentinel remains a separate model property.
    """
    if type(tokenizer_eos) is not int or tokenizer_eos < 0:
        raise ValueError("tokenizer EOS must be a non-negative integer")
    result = {tokenizer_eos}
    for filename in ("config.json", "generation_config.json"):
        path = checkpoint / filename
        if not path.exists():
            continue
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot load EOS metadata from {path}") from error
        if not isinstance(document, dict):
            raise ValueError(f"{filename} must contain an object")
        declarations = [document]
        if filename == "config.json" and isinstance(document.get("text_config"), dict):
            declarations.append(document["text_config"])
        for declaration in declarations:
            value = declaration.get("eos_token_id")
            if value is None:
                continue
            values = [value] if type(value) is int else value
            if not isinstance(values, list) or any(type(x) is not int or x < 0 for x in values):
                raise ValueError(f"{filename} eos_token_id must be an integer or integer list")
            result.update(values)
    return tuple(sorted(result))


def resolve_sampling(
    defaults: SamplingParameters,
    overrides: SamplingMeta,
) -> SamplingParameters:
    if not isinstance(overrides, SamplingMeta):
        raise ValueError("generation metadata requires SamplingMeta")
    temperature = defaults.temperature if overrides.temperature is None else overrides.temperature
    return SamplingParameters(
        temperature,
        defaults.top_p if overrides.top_p is None else overrides.top_p,
        defaults.top_k if overrides.top_k is None else overrides.top_k,
        0
        if temperature == 0.0
        else (secrets.randbits(63) if overrides.seed is None else overrides.seed % 2**63),
    )


class GreedySampler:
    """Select the first maximum vocabulary id with no scheduler-owned RNG."""

    def sample(
        self,
        logits: torch.Tensor,
        constraint: LogitConstraint | None = None,
    ) -> int:
        if logits.ndim != 1 or logits.numel() == 0 or not logits.is_floating_point():
            raise ValueError("greedy sampling requires non-empty rank-1 floating logits")
        if not bool(torch.all(torch.isfinite(logits)).item()):
            raise ValueError("greedy sampling rejects non-finite logits")
        if constraint is None:
            return self._argmax(logits)

        constrained_logits = logits.clone()
        mask = constraint.mask()
        constraint.apply(constrained_logits, mask)
        if bool(torch.any(torch.isnan(constrained_logits) | torch.isposinf(constrained_logits))):
            raise ValueError("constraint produced invalid candidate logits")
        if not bool(torch.any(torch.isfinite(constrained_logits))):
            raise ValueError("constraint masked every candidate token")
        token_id = self._argmax(constrained_logits)
        constraint.accept(token_id, mask)
        return token_id

    @staticmethod
    def _argmax(logits: torch.Tensor) -> int:
        return int(torch.argmax(logits.to(torch.float32), dim=0).item())


def load_sampling_defaults(checkpoint: Path) -> SamplingParameters:
    """Load the checkpoint's generation policy without server-side invention."""

    path = checkpoint / "generation_config.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load checkpoint sampling defaults from {path}") from error
    if not isinstance(payload, dict) or type(payload.get("do_sample")) is not bool:
        raise ValueError("generation_config.json must declare boolean do_sample")
    if not payload["do_sample"]:
        return GREEDY_SAMPLING
    try:
        return SamplingParameters(
            float(payload["temperature"]),
            float(payload["top_p"]),
            int(payload["top_k"]),
            0,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("generation_config.json has unsupported sampling defaults") from error
