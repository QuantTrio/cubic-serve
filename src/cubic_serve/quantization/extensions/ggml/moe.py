"""Original expert order, common route metadata, and format-owned packed compute."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from cubic_serve.models.quantized_checkpoint import QuantizedMatrix
from cubic_serve.ops.cuda.compiled_artifact import TensorBinding, bind_tensor
from cubic_serve.ops.cuda.native import NativeLaunch
from cubic_serve.quantization.api import RoutedRows
from cubic_serve.quantization.common.binding import (
    PackedOperatorBinding,
    _PackedLaunch,
    concatenate_to_device,
)
from cubic_serve.quantization.registry import select_compute_mode

if TYPE_CHECKING:
    from cubic_serve.ops.cuda.linear.quantized import PackedLinearOperators

from .checkpoint import column_offset, kind
from .native import bind_matmul
from .policy import GGMLMoEParameters, bind_arithmetic, select_routed_matrix, select_vector_rows
from .q8_storage import Q8Storage, prepare


def _llama_row_tile(x: Tensor, weight: QuantizedMatrix, experts: int, rows: RoutedRows) -> int:
    hardware = torch.cuda.get_device_properties(x.device)
    return 8 if select_routed_matrix(
        mean_rows=rows.mean_rows_per_expert, experts=experts,
        n=weight.shape[0], k=weight.shape[1],
        row_bytes=weight.payload["weight_packed"].shape[-1],
        sms=hardware.multi_processor_count, l2_bytes=hardware.L2_cache_size,
    ) else 1


@dataclass(frozen=True)
class _Sources:
    weights: tuple[QuantizedMatrix, ...]
    bindings: tuple[TensorBinding, ...]

    def matches(self) -> bool:
        return all(binding.matches(weight.payload["weight_packed"])
                   for weight, binding in zip(self.weights, self.bindings, strict=True))


def _packed(operators: PackedLinearOperators, weights: tuple[QuantizedMatrix, ...],
             device: torch.device) -> Tensor | Q8Storage:
    first = weights[0]
    if any(w.encoding != first.encoding or w.shape != first.shape for w in weights):
        raise ValueError("GGML routed projection requires one declared local encoding")
    if operators.config.compute_mode == "wna8" and int(kind(first.encoding)) == 8:
        return prepare(operators, (weights,), device)
    key = ("ggml.original-experts.v1", device, tuple(id(w) for w in weights))
    cached = operators.prepared_weights.get(key)
    if cached is not None and isinstance(cached[0], _Sources) and cached[0].matches():
        return cached[1]
    sources = _Sources(weights, tuple(bind_tensor(w.payload["weight_packed"]) for w in weights))
    packed = concatenate_to_device(tuple(w.payload["weight_packed"] for w in weights), device)
    packed = packed.view(len(weights), *first.payload["weight_packed"].shape)
    operators.prepared_weights[key] = sources, packed
    return packed


def bind_grouped(
    operators: PackedLinearOperators, x: Tensor, gate: tuple[QuantizedMatrix, ...],
    up: tuple[QuantizedMatrix, ...], output: Tensor, rows: RoutedRows, top_k: int,
    params: GGMLMoEParameters,
) -> tuple[int, PackedOperatorBinding]:
    for weight in (gate[0], up[0]):
        select_compute_mode(weight.encoding, "moe", operators.config.compute_mode,
                            m=x.shape[0], n=weight.shape[0], k=weight.shape[1],
                            dtype=x.dtype, device=x.device)
    if operators.config.compute_mode == "wna8":
        from .llama import bind_matmul as bind_llama

        width = gate[0].shape[0]
        row_tile = max(_llama_row_tile(x, weights[0], len(weights), rows)
                       for weights in (gate, up))
        if gate[0].encoding == up[0].encoding and gate[0].shape == up[0].shape:
            # Each expert keeps its own gate rows followed by its own up rows.
            # Reuse the original-block cache; do not retain separate device copies.
            if int(kind(gate[0].encoding)) == 8:
                packed = prepare(operators, (gate, up), x.device)
            else:
                ordered = tuple(weight for pair in zip(gate, up, strict=True) for weight in pair)
                packed = _packed(operators, ordered, x.device)
                packed = packed.view(len(gate), 2 * width, packed.shape[-1])
            binding = bind_arithmetic(
                operators, x, gate[0], params,
                lambda algorithm, matrix_rows: bind_llama(
                    x, packed, int(kind(gate[0].encoding)), output,
                    rows=rows, input_top_k=top_k, algorithm=algorithm, matrix_rows=matrix_rows,
                ),
                default="matrix" if row_tile > 1 else "vector", routed=True,
                experts=len(gate), input_top_k=top_k,
                routing_top_k=rows.routing_top_k or top_k, group=(gate[0], up[0]),
            )
            tile = (binding.row_tile_size if binding._launches[-1].operator_id
                    == "cubic.cuda.ggml.llama.mmq.v1" else 1)
            return tile, binding
        bindings = tuple(bind_arithmetic(
            operators, x, weights[0], params,
            lambda algorithm, matrix_rows, weights=weights, offset=offset: bind_llama(
                x, _packed(operators, weights, x.device), int(kind(weights[0].encoding)),
                output[:, offset:offset + width], rows=rows, input_top_k=top_k,
                algorithm=algorithm, matrix_rows=matrix_rows,
            ),
            default="matrix" if row_tile > 1 else "vector", routed=True,
            experts=len(weights), input_top_k=top_k, routing_top_k=rows.routing_top_k or top_k,
            group=(gate[0], up[0]),
        ) for weights, offset in ((gate, 0), (up, width)))
        tiles = {
            b.row_tile_size if b._launches[-1].operator_id == "cubic.cuda.ggml.llama.mmq.v1" else 1
            for b in bindings
        }
        if len(tiles) != 1:
            raise ValueError("GGML shared route prefix has inconsistent arithmetic row tiles")
        row_tile = tiles.pop()
        # Gate/up consume the same activation. Reuse one Q8_1 producer and
        # private buffer; each original weight retains its own pointwise dot.
        first, second = bindings
        second.rebind_private_scratch(first.scratch[0])
        return row_tile, PackedOperatorBinding(
            output, (*first._launches, *second._launches[1:]),
            scratch=first.scratch, reusable_scratch=True,
        )
    width = gate[0].shape[0]
    vector_rows = select_vector_rows(params.vector_rows, input_rows=x.shape[0],
                                     input_top_k=top_k, routing_top_k=rows.routing_top_k)
    calls: list[NativeLaunch | _PackedLaunch] = []
    for weights, offset in ((gate, 0), (up, width)):
        binding = bind_matmul(
            x, _packed(operators, weights, x.device), int(kind(weights[0].encoding)),
            output[:, offset:offset + width], rows=rows, input_top_k=top_k,
            vector_rows=vector_rows, column_offset=column_offset(weights[0].encoding),
        )
        calls.extend(binding._launches)
    return vector_rows or 16, PackedOperatorBinding(output, tuple(calls))


def bind_routed(
    operators: PackedLinearOperators, x: Tensor, weights: tuple[QuantizedMatrix, ...],
    output: Tensor, rows: RoutedRows, params: GGMLMoEParameters,
) -> tuple[int, PackedOperatorBinding]:
    select_compute_mode(weights[0].encoding, "moe", operators.config.compute_mode,
                        m=x.shape[0], n=weights[0].shape[0], k=weights[0].shape[1],
                        dtype=x.dtype, device=x.device)
    if operators.config.compute_mode == "wna8":
        from .llama import bind_matmul as bind_llama

        binding = bind_arithmetic(
            operators, x, weights[0], params,
            lambda algorithm, matrix_rows: bind_llama(
                x, _packed(operators, weights, x.device), int(kind(weights[0].encoding)),
                output.view(-1, weights[0].shape[0]), rows=rows, algorithm=algorithm,
                matrix_rows=matrix_rows,
            ),
            routed=True, experts=len(weights), routing_top_k=rows.routing_top_k or 1,
        )
        row_tile = (binding.row_tile_size
                    if binding._launches[-1].operator_id == "cubic.cuda.ggml.llama.mmq.v1" else 1)
        return row_tile, binding
    vector_rows = select_vector_rows(params.vector_rows, input_rows=x.shape[0],
                                     routing_top_k=rows.routing_top_k)
    binding = bind_matmul(
        x, _packed(operators, weights, x.device), int(kind(weights[0].encoding)),
        output.view(-1, weights[0].shape[0]), rows=rows,
        vector_rows=vector_rows, column_offset=column_offset(weights[0].encoding),
    )
    return vector_rows or 16, binding
