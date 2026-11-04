"""Composite reduction qualification for fixed-ABI BF16 paged GQA decode."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from cubic_serve.guardrail.compare import (
    stable_digest,
    tensor_binding_digest,
    tensor_digest,
)
from cubic_serve.ops.cuda.attention import paged_decode

_FP32_U = 2.0**-24
_TREE_ABI = paged_decode.PAGED_GQA_TREE_ABI
_BLOCK_N = 16
_WITNESS_TOKEN = object()
class AttentionQualificationError(ValueError):
    """Raised when composite attention evidence cannot be issued."""
@dataclass(frozen=True, slots=True, init=False)
class VerifiedAttentionReductionWitness:
    expected_binding: str
    actual_binding: str
    material_digest: str
    fingerprint_digest: str
    scalar_probe_digest: str
    bounds_digest: str
    max_allowed_error: float
    witness_digest: str
    def __init__(self, token: object) -> None:
        if token is not _WITNESS_TOKEN:
            raise TypeError("attention witnesses are issued only by the verifier")
@dataclass(frozen=True, slots=True)
class _LogicalRow:
    physical_row: int
    traversal: tuple[tuple[int, int], ...]
    query: Tensor
    key: Tensor
    value: Tensor
def qualify_paged_gqa_attention(
    expected: Tensor,
    actual: Tensor,
    *,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    page_table: Tensor,
    seq_lens: Tensor,
    active_rows: tuple[int, ...],
    layer: int,
    scale: float | None = None,
) -> VerifiedAttentionReductionWitness:
    """Issue content-bound evidence only for finite FP32 reassociation drift."""
    from cubic_serve.ops.strategies import attention_precision

    if attention_precision() != "fp32":
        raise AttentionQualificationError(
            "low-operand attention requires a P-rounding witness; "
            "the FP32 reassociation witness cannot qualify it"
        )
    rows, resolved_scale = _logical_rows(
        expected, actual, query, key, value, page_table, seq_lens, active_rows, layer, scale
    )
    canonical, scores = _canonical(rows, resolved_scale)
    expected_active = expected[list(active_rows)].contiguous()
    actual_active = actual[list(active_rows)].contiguous()
    if not torch.equal(expected_active, canonical):
        raise AttentionQualificationError("expected output is not canonical eager attention")
    if not bool(torch.isfinite(actual_active.float()).all().item()):
        raise AttentionQualificationError("candidate attention output is non-finite")
    fingerprint = stable_digest(
        paged_decode.paged_gqa_fingerprint_payload(
            query, key, page_table, layer=layer, scale=resolved_scale
        )
    )
    scalar_probe = _attest_scalar_abi(query.device)
    allowed = _reduction_envelope(rows, scores, expected_active, actual_active, resolved_scale)
    error = (actual_active.double() - expected_active.double()).abs().cpu()
    if bool((error > allowed).any().item()):
        raise AttentionQualificationError("candidate exceeds the composite reduction envelope")
    return _issue_witness(
        expected_active,
        actual_active,
        material_digest=_material_digest(rows, layer, resolved_scale),
        fingerprint_digest=fingerprint,
        scalar_probe_digest=scalar_probe,
        allowed=allowed,
    )
def validate_paged_gqa_attention_witness(
    witness: VerifiedAttentionReductionWitness,
    expected: Tensor,
    actual: Tensor,
    **arguments: Any,
) -> None:
    """Re-qualify current content and reject stale or foreign evidence."""
    if not isinstance(witness, VerifiedAttentionReductionWitness):
        raise TypeError("attention witness type is required")
    observed = qualify_paged_gqa_attention(expected, actual, **arguments)
    if observed.witness_digest != witness.witness_digest:
        raise AttentionQualificationError("attention witness is bound to different content")
def _logical_rows(
    expected: Tensor,
    actual: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    page_table: Tensor,
    seq_lens: Tensor,
    active_rows: tuple[int, ...],
    layer: int,
    scale: float | None,
) -> tuple[tuple[_LogicalRow, ...], float]:
    if (
        query.ndim != 3
        or key.ndim != 5
        or value.shape != key.shape
        or expected.shape != query.shape
        or actual.shape != query.shape
        or any(tensor.device != query.device for tensor in (key, value, page_table, seq_lens))
        or any(
            tensor.dtype is not torch.bfloat16 for tensor in (query, key, value, expected, actual)
        )
        or not all(tensor.is_contiguous() for tensor in (query, key, value, page_table, seq_lens))
    ):
        raise AttentionQualificationError("paged attention tensors violate the fixed BF16 ABI")
    capacity, query_heads, dim = query.shape
    pages, layers, page_block, kv_heads, key_dim = key.shape
    table_pages = page_table.shape[1] if page_table.ndim == 2 else -1
    if (
        page_table.shape != (capacity, table_pages)
        or seq_lens.shape != (capacity,)
        or page_table.dtype is not torch.int32
        or seq_lens.dtype is not torch.int32
        or dim != key_dim
        or query_heads % kv_heads
        or not active_rows
        or len(set(active_rows)) != len(active_rows)
        or any(type(row) is not int or not 0 <= row < capacity for row in active_rows)
        or type(layer) is not int
        or not 0 <= layer < layers
    ):
        raise AttentionQualificationError("paged attention metadata violates the fixed ABI")
    resolved_scale = dim**-0.5 if scale is None else scale
    if (
        type(resolved_scale) is not float
        or not math.isfinite(resolved_scale)
        or resolved_scale <= 0
    ):
        raise AttentionQualificationError("attention scale must be finite and positive")
    lengths = seq_lens.detach().cpu().tolist()
    tables = page_table.detach().cpu()
    logical: list[_LogicalRow] = []
    for row in active_rows:
        length = int(lengths[row])
        if not 0 < length <= table_pages * page_block:
            raise AttentionQualificationError("logical context length exceeds the page table")
        traversal = tuple(
            (int(tables[row, position // page_block]), position % page_block)
            for position in range(length)
        )
        if any(not 0 <= page < pages for page, _ in traversal):
            raise AttentionQualificationError("logical page traversal is out of range")
        page_ids = torch.tensor([item[0] for item in traversal], device=query.device)
        offsets = torch.tensor([item[1] for item in traversal], device=query.device)
        logical.append(
            _LogicalRow(
                row,
                traversal,
                query[row],
                key[page_ids, layer, offsets].contiguous(),
                value[page_ids, layer, offsets].contiguous(),
            )
        )
    return tuple(logical), resolved_scale
def _canonical(rows: tuple[_LogicalRow, ...], scale: float) -> tuple[Tensor, tuple[Tensor, ...]]:
    outputs: list[Tensor] = []
    score_rows: list[Tensor] = []
    for row in rows:
        query_heads, _ = row.query.shape
        group = query_heads // row.key.shape[1]
        heads: list[Tensor] = []
        scores: list[Tensor] = []
        for head in range(query_heads):
            kv_head = head // group
            score = (row.key[:, kv_head].float() * row.query[head].float()).sum(-1) * scale
            probability = torch.softmax(score, 0, dtype=torch.float32)
            heads.append((row.value[:, kv_head].float() * probability[:, None]).sum(0))
            scores.append(score)
        outputs.append(torch.stack(heads).to(torch.bfloat16))
        score_rows.append(torch.stack(scores))
    return torch.stack(outputs), tuple(score_rows)

def _reduction_envelope(
    rows: tuple[_LogicalRow, ...],
    scores: tuple[Tensor, ...],
    expected: Tensor,
    actual: Tensor,
    scale: float,
) -> Tensor:
    bounds: list[Tensor] = []
    for index, row in enumerate(rows):
        query = row.query.detach().cpu().double()
        keys = row.key.detach().cpu().double()
        values = row.value.detach().cpu().double()
        query_heads, dim = query.shape
        group = query_heads // keys.shape[1]
        head_map = torch.arange(query_heads) // group
        products = keys[:, head_map].permute(1, 0, 2) * query[:, None, :]
        sum_abs = products.abs().sum(-1)
        exact = torch.tensor(
            [math.fsum(item) for item in products.reshape(-1, dim).tolist()],
            dtype=torch.float64,
        ).reshape(query_heads, -1)
        gamma_dot = _gamma(dim - 1)
        candidate_dot_error = gamma_dot * sum_abs
        scale32 = torch.tensor(scale, dtype=torch.float32).item()
        scaled_exact = exact * scale32
        candidate_scale_error = abs(scale32) * candidate_dot_error + _FP32_U * (
            scaled_exact.abs() + abs(scale32) * candidate_dot_error
        )
        reference_error = (scores[index].detach().cpu().double() - scaled_exact).abs()
        score_delta = (candidate_scale_error + reference_error).amax(-1)[:, None]
        grouped_values = values[:, head_map].permute(1, 0, 2)
        value_max = grouped_values.abs().amax(1)
        score_effect = torch.clamp(2.0 * score_delta, max=2.0) * value_max

        tokens = keys.shape[0]
        tiles = math.ceil(tokens / _BLOCK_N)
        denominator_error = _gamma(tokens - 1 + 4 * tiles + 4)
        numerator_error = _gamma(tokens - 1 + 5 * tiles + 5)
        if denominator_error >= 1.0:
            raise AttentionQualificationError("online softmax reduction bound is undefined")
        ratio_error = 2.0 * (
            (numerator_error + denominator_error) / (1.0 - denominator_error)
            + _FP32_U / (1.0 - _FP32_U)
        )
        bounds.append(score_effect + ratio_error * value_max)
    allowed = torch.stack(bounds) + _half_ulp(expected) + _half_ulp(actual)
    if not bool(torch.isfinite(allowed).all().item()):
        raise AttentionQualificationError("attention reduction envelope is non-finite")
    return allowed

def _gamma(steps: int) -> float:
    product = steps * _FP32_U
    if steps < 0 or product >= 1.0:
        raise AttentionQualificationError("FP32 reduction gamma is undefined")
    return product / (1.0 - product)

def _half_ulp(tensor: Tensor) -> Tensor:
    value = tensor.detach().cpu()
    upper = torch.nextafter(value, torch.full_like(value, float("inf"))).double()
    lower = torch.nextafter(value, torch.full_like(value, float("-inf"))).double()
    center = value.double()
    distances = torch.stack(((upper - center).abs(), (center - lower).abs()))
    distances[~torch.isfinite(distances)] = 0.0
    return distances.amax(0) * 0.5

def _material_digest(rows: tuple[_LogicalRow, ...], layer: int, scale: float) -> str:
    return stable_digest(
        {
            "layer": layer,
            "scale": scale.hex(),
            "rows": [
                {
                    "physical_row": row.physical_row,
                    "traversal": row.traversal,
                    "query": tensor_digest(row.query),
                    "key": tensor_digest(row.key),
                    "value": tensor_digest(row.value),
                }
                for row in rows
            ],
        }
    )

def _attest_scalar_abi(device: torch.device) -> str:
    operands, actual = paged_decode.paged_gqa_scalar_probe(device)
    left, right, addend = operands
    product = torch.mul(left, right)
    expected = torch.stack(
        (
            torch.exp(left),
            torch.div(left, right),
            product,
            torch.add(product, addend),
            torch.sub(left, addend),
        ),
        dim=1,
    )
    if not torch.equal(expected, actual):
        raise AttentionQualificationError("attention scalar ABI probe is not bit-exact")
    return stable_digest(
        {
            "operands": [tensor_digest(item) for item in operands],
            "result": tensor_digest(actual),
            "abi": _TREE_ABI["scalar"],
        }
    )

def _issue_witness(
    expected: Tensor,
    actual: Tensor,
    *,
    material_digest: str,
    fingerprint_digest: str,
    scalar_probe_digest: str,
    allowed: Tensor,
) -> VerifiedAttentionReductionWitness:
    witness = VerifiedAttentionReductionWitness(_WITNESS_TOKEN)
    payload = {
        "expected_binding": tensor_binding_digest(expected),
        "actual_binding": tensor_binding_digest(actual),
        "material_digest": material_digest,
        "fingerprint_digest": fingerprint_digest,
        "scalar_probe_digest": scalar_probe_digest,
        "bounds_digest": tensor_digest(allowed.contiguous()),
        "max_allowed_error": float(allowed.max().item()),
    }
    for name, value in payload.items():
        object.__setattr__(witness, name, value)
    object.__setattr__(witness, "witness_digest", stable_digest(payload))
    return witness
