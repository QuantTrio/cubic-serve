"""Native bindings through the existing packed-linear extension contract."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor

from cubic_serve.models.quantized_checkpoint import QuantizedMatrix
from cubic_serve.quantization.common.binding import PackedOperatorBinding
from cubic_serve.quantization.registry import select_compute_mode

if TYPE_CHECKING:
    from cubic_serve.ops.cuda.linear.quantized import PackedLinearOperators

from . import affine
from .checkpoint import column_offset, kind
from .embedding import bind_scaled_sharded_embedding as bind_scaled_sharded_embedding
from .embedding import bind_native_embedding as bind_native_embedding
from .embedding import (
    bind_prepared_scaled_sharded_embedding as bind_prepared_scaled_sharded_embedding,
)
from .native import bind_matmul
from .policy import GGMLLinearParameters, bind_arithmetic, select_vector_rows


def _device_weight(operators, weight, device):
    if int(kind(weight.encoding)) == 8 and operators.config.compute_mode == "wna8":
        from .q8_storage import prepare

        return prepare(operators, ((weight,),), device)
    return operators.materialize_weight(weight, device).payload["weight_packed"]


def _bind_native(
    operators: PackedLinearOperators, x: Tensor, weight: QuantizedMatrix, output: Tensor,
    params: GGMLLinearParameters, fp8_token_scales: Tensor | None = None,
) -> PackedOperatorBinding:
    if fp8_token_scales is not None:
        raise ValueError("GGML WNA16 does not consume externally quantized activations")
    mode = select_compute_mode(weight.encoding, "linear", operators.config.compute_mode,
                        m=x.shape[0], n=weight.shape[0], k=weight.shape[1],
                        dtype=x.dtype, device=x.device)
    if mode == "wna8":
        from .llama import bind_matmul as bind_llama

        return bind_llama(x, _device_weight(operators, weight, x.device),
                          int(kind(weight.encoding)), output)
    if params.vector_rows == -1 and affine.compatible(weight):
        proxy = affine.prepare(operators, weight)
        if proxy is not None:
            return operators.bind_linear(x, proxy, output)
    matrix = operators.materialize_weight(weight, x.device)
    return bind_matmul(x, matrix.payload["weight_packed"], int(kind(matrix.encoding)), output,
                       vector_rows=select_vector_rows(params.vector_rows, input_rows=x.shape[0]),
                       column_offset=column_offset(matrix.encoding))


def _planned_wna8(
    operators: PackedLinearOperators, x: Tensor, weight: QuantizedMatrix, output: Tensor,
    params: GGMLLinearParameters, original: PackedOperatorBinding, *, prefer_mma: bool,
    group: tuple[QuantizedMatrix, ...] = (),
) -> PackedOperatorBinding:
    def bind(algorithm: str, matrix_rows: int | None) -> PackedOperatorBinding:
        if algorithm == "mma" or (algorithm == "auto" and prefer_mma):
            if int(kind(weight.encoding)) != 8:
                raise ValueError("full-group Q8 arithmetic requires its original Q8 encoding")
            from .q8_mma import bind as bind_mma

            launch = bind_mma(x, (_device_weight(operators, weight, x.device),), (output,),
                              original.scratch[0])
            return PackedOperatorBinding(
                output, (original._launches[0], launch),
                scratch=original.scratch, reusable_scratch=True,
            )
        modes = {"vector": "cubic.cuda.ggml.llama.mmvq.v1",
                 "matrix": "cubic.cuda.ggml.llama.mmq.v1"}
        if (algorithm == "auto" or (matrix_rows is None
                and original._launches[-1].operator_id == modes.get(algorithm))):
            return original
        from .llama import bind_matmul as bind_llama

        return bind_llama(
            x, _device_weight(operators, weight, x.device), int(kind(weight.encoding)), output,
            algorithm=algorithm, matrix_rows=matrix_rows,
        )

    return bind_arithmetic(operators, x, weight, params, bind, group=group)


def bind_native(
    operators: PackedLinearOperators, x: Tensor, weight: QuantizedMatrix, output: Tensor,
    params: GGMLLinearParameters, fp8_token_scales: Tensor | None = None,
) -> PackedOperatorBinding:
    binding = _bind_native(operators, x, weight, output, params, fp8_token_scales)
    if operators.config.compute_mode != "wna8":
        return binding
    import torch

    from .q8_mma import selected as mma_selected

    eligible = (int(kind(weight.encoding)) == 8
                and binding._launches[-1].operator_id == "cubic.cuda.ggml.llama.mmvq.v1"
                and mma_selected(m=x.shape[0], widths=(output.shape[1],),
                    sms=torch.cuda.get_device_properties(x.device).multi_processor_count))
    binding = _planned_wna8(operators, x, weight, output, params, binding, prefer_mma=eligible)
    if (int(kind(weight.encoding)) != 8
            or binding._launches[-1].operator_id != "cubic.cuda.ggml.llama.mmvq.v1"):
        return binding

    from .q8_group import bind_single, selected

    if not selected(m=x.shape[0], widths=(output.shape[1],),
                    sms=torch.cuda.get_device_properties(x.device).multi_processor_count):
        return binding
    launch = bind_single(x, _device_weight(operators, weight, x.device), output, binding.scratch[0])
    return PackedOperatorBinding(
        output, (binding._launches[0], launch),
        scratch=binding.scratch, reusable_scratch=True,
    )


bind_linear = bind_native


def bind_grouped(
    operators: PackedLinearOperators, x: Tensor, weights: tuple[QuantizedMatrix, ...],
    outputs: tuple[Tensor, ...], params: GGMLLinearParameters,
) -> PackedOperatorBinding | None:
    if operators.config.compute_mode == "wna8":
        bindings = tuple(_bind_native(operators, x, weight, output, params)
                         for weight, output in zip(weights, outputs, strict=True))
        import torch

        from .q8_mma import bind as bind_mma
        from .q8_mma import selected as mma_selected

        eligible = (len(weights) == 2 and all(int(kind(w.encoding)) == 8 for w in weights)
                    and all(b._launches[-1].operator_id == "cubic.cuda.ggml.llama.mmvq.v1"
                            for b in bindings)
                    and mma_selected(m=x.shape[0], widths=tuple(y.shape[1] for y in outputs),
                        sms=torch.cuda.get_device_properties(x.device).multi_processor_count))
        bindings = tuple(_planned_wna8(
            operators, x, w, y, params, b, prefer_mma=eligible, group=weights,
        ) for w, y, b in zip(weights, outputs, bindings, strict=True))
        first = bindings[0]
        for binding in bindings[1:]:
            binding.rebind_private_scratch(first.scratch[0])
        if len(bindings) == 2 and all(
            b._launches[-1].operator_id == "cubic.cuda.ggml.q8_mma.v1" for b in bindings
        ):
            launch = bind_mma(x, tuple(_device_weight(operators, w, x.device) for w in weights),
                              outputs, first.scratch[0])
            return PackedOperatorBinding(
                first.output, (first._launches[0], launch),
                scratch=first.scratch, reusable_scratch=True,
            )
        if (len(bindings) == 2 and all(int(kind(w.encoding)) == 8 for w in weights)
                and all(b._launches[-1].operator_id == "cubic.cuda.ggml.llama.mmvq.v1"
                        for b in bindings)):
            import torch

            from .q8_group import bind_pair, selected

            widths = (outputs[0].shape[1], outputs[1].shape[1])
            if selected(m=x.shape[0], widths=widths,
                        sms=torch.cuda.get_device_properties(x.device).multi_processor_count):
                launch = bind_pair(
                    x, tuple(_device_weight(operators, w, x.device) for w in weights),
                    (outputs[0], outputs[1]), first.scratch[0],
                )
                return PackedOperatorBinding(
                    first.output, (first._launches[0], launch),
                    scratch=first.scratch, reusable_scratch=True,
                )
        # All members consume the same immutable floating boundary and the
        # format's same Q8_1 codes/scales. Share its producer, never another
        # format's per-token INT8 encoding or an in-flight branch's storage.
        return PackedOperatorBinding(
            first.output,
            (*first._launches, *(launch for binding in bindings[1:]
                                 for launch in binding._launches[1:])),
            scratch=first.scratch, reusable_scratch=True,
        )
    if (params.vector_rows != -1
            or not all(affine.compatible(w) for w in weights)):
        return None
    for weight in weights:
        select_compute_mode(weight.encoding, "linear", operators.config.compute_mode,
                            m=x.shape[0], n=weight.shape[0], k=weight.shape[1],
                            dtype=x.dtype, device=x.device)
    prepared = tuple(affine.prepare(operators, w) for w in weights)
    if any(w is None for w in prepared):
        return None
    return operators.bind_linear_group(x, tuple(w for w in prepared if w is not None), outputs)


def bind_embedding(
    operators: PackedLinearOperators, input_ids: Tensor, weight: QuantizedMatrix,
    output: Tensor, params: GGMLLinearParameters,
) -> PackedOperatorBinding:
    from .embedding import bind_lookup

    matrix = operators.materialize_weight(weight, input_ids.device)
    return bind_lookup(input_ids, ((0, matrix),), output)
