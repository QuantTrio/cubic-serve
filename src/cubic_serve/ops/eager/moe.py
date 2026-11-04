"""Stable expert selection from logits, independent of model/checkpoint ownership."""

import torch
from torch import Tensor

from cubic_serve.executor.runner import require_reference_execution


def topk_routes(logits: Tensor, *, top_k: int, normalize: bool) -> tuple[Tensor, Tensor]:
    require_reference_execution()
    if logits.ndim != 2 or not 0 < top_k <= logits.shape[-1]:
        raise ValueError("router needs [rows,experts] logits and a valid top-k")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("router logits must be finite")
    # Define ties before probability rounding, using original expert order.
    indices = torch.argsort(logits, descending=True, stable=True)[:, :top_k]
    probabilities = torch.softmax(logits.float(), dim=-1).gather(1, indices)
    if normalize:
        probabilities = probabilities / probabilities.sum(-1, keepdim=True)
    indices, order = indices.sort(dim=-1)
    return indices, probabilities.gather(1, order).to(logits.dtype)
