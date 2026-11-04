"""Graph-bound original-block lookup, including the shared pinned-host ABI.

llama.cpp 987498f4592a76897863cf53711dce38380c082b getrows.cu and vLLM
ab3e80042eac24dd362408e6d63ad98768046359 gguf.py select rows before decoding;
SGLang 8a20062652472190c7150b4022bd83048cb491d6 retains that invariant.
Mapped host allocation, placement, lifetime and rank ownership remain common.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import Tensor

from cubic_serve.models.quantized_checkpoint import QuantizedMatrix
from cubic_serve.ops.cuda.compiled_artifact import bind_tensor
from cubic_serve.ops.cuda.embedding.immutable import embedding_pointer
from cubic_serve.ops.cuda.native import CudaScalar, NativeLaunch
from cubic_serve.ops.cuda.native_compiler import compile_native
from cubic_serve.quantization.common.binding import PackedOperatorBinding

from .checkpoint import column_offset, kind
from .native import _grid
from .q8_storage import Q8Storage, prepare


class _EmbeddingBinding(PackedOperatorBinding):
    def __init__(self, output: Tensor, launches: tuple[NativeLaunch, ...], *,
                 payloads: tuple[Tensor, ...], metadata: tuple[Tensor, ...]) -> None:
        super().__init__(output, launches, scratch=(*payloads, *metadata))
        self.payloads = payloads
        self.payload_bindings = tuple(bind_tensor(t) for t in payloads)

    def validate_operands(self) -> None:
        super().validate_operands()
        if tuple(bind_tensor(t) for t in self.payloads) != self.payload_bindings:
            raise ValueError("immutable GGML embedding storage changed after binding")


def bind_lookup(
    ids: Tensor, shards: tuple[tuple[int, QuantizedMatrix], ...], output: Tensor, *,
    scale: Tensor | None = None, activity: Tensor | None = None,
    prepared: tuple[Q8Storage, ...] = (),
) -> PackedOperatorBinding:
    if (ids.device.type != "cuda" or ids.dtype not in (torch.int32, torch.int64)
            or not ids.is_contiguous() or ids.numel() <= 0 or not shards
            or output.dtype not in (torch.bfloat16, torch.float16)
            or output.device != ids.device or not output.is_contiguous()):
        raise ValueError("GGML embedding requires fixed CUDA integer IDs and FP16/BF16 output")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("GGML embedding binding must precede graph capture")
    if prepared and len(prepared) != len(shards):
        raise ValueError("prepared lookup must preserve every original shard")
    width = shards[0][1].shape[1]
    if output.numel() != ids.numel() * width:
        raise ValueError("GGML embedding output differs from its lookup geometry")
    if activity is not None and (
        ids.ndim != 2 or activity.shape != (ids.shape[0],) or activity.dtype != torch.int32
        or activity.device != ids.device or not activity.is_contiguous()
        or output.shape != (ids.shape[0], ids.shape[1] * width)
    ):
        raise ValueError("GGML sharded embedding needs explicit logical-row ownership")
    if scale is not None and (
        scale.numel() != 1 or scale.device != ids.device or not scale.is_contiguous()
        or scale.dtype not in (torch.float32, torch.float16, torch.bfloat16)
        or not 0 < float(scale.item()) < float("inf")
    ):
        raise ValueError("GGML embedding direct scale must be finite, positive and device resident")
    starts, ends, pointers, strides, types, offsets, layouts = [], [], [], [], [], [], []
    payloads = []
    previous_end = 0
    for ordinal, (start, weight) in enumerate(shards):
        if (not isinstance(weight, QuantizedMatrix) or weight.encoding.format != "ggml-block-v1"
                or type(start) is not int or start < previous_end
                or start + weight.shape[0] >= 2**63 or weight.shape[1] != width):
            raise ValueError("GGML lookup shards need disjoint original row owners and one width")
        carrier = prepared[ordinal] if prepared else None
        if carrier is not None and (
            int(kind(weight.encoding)) != 8 or column_offset(weight.encoding) != 0
            or carrier.n != weight.shape[0] or carrier.k != width or carrier.experts != 1
            or carrier.storage.device != ids.device
        ):
            raise ValueError("prepared lookup differs from its original row ownership")
        packed = carrier.storage if carrier is not None else weight.payload["weight_packed"]
        payloads.append(packed)
        starts.append(start)
        previous_end = start + weight.shape[0]
        ends.append(previous_end)
        pointers.append(embedding_pointer(packed, ids.device))
        strides.append(width // 32 * 34 if carrier is not None else packed.shape[1])
        types.append(int(kind(weight.encoding)))
        offsets.append(column_offset(weight.encoding))
        layouts.append(int(carrier is not None))
    operands = (ids, activity, scale, *payloads)
    if any(t is not None and t.untyped_storage().data_ptr() == output.untyped_storage().data_ptr()
           for t in operands):
        raise ValueError("GGML embedding output may not alias lookup inputs")
    metadata = tuple(torch.tensor(values, dtype=dtype, device=ids.device)
                     for values, dtype in (
                         (starts, torch.int64), (ends, torch.int64), (pointers, torch.int64),
                         (strides, torch.int32), (types, torch.int32), (offsets, torch.int32),
                         (layouts, torch.int32)))
    prefix = f"#define IS_BF16 {int(output.dtype == torch.bfloat16)}\n".encode()
    folder = Path(__file__).parent
    module, entry = compile_native((
        ("entry.cu", prefix + (folder / "embedding.cu").read_bytes()),
        ("decode.cuh", (folder / "decode.cuh").read_bytes()),
        ("q8_storage.cuh", (folder / "q8_storage.cuh").read_bytes()),
    ), ids.device, "ggml_embedding")
    scale_type = {torch.float32: 0, torch.float16: 1, torch.bfloat16: 30}[
        scale.dtype if scale is not None else torch.float32]
    launch = module.bind(
        entry, (ids, activity, *metadata, _grid(ids.device), scale, output,
                CudaScalar("i32", ids.numel()),
                CudaScalar("i32", ids.shape[1] if activity is not None else 1),
                CudaScalar("i32", width), CudaScalar("i32", len(shards)),
                CudaScalar("i32", ids.element_size()), CudaScalar("i32", scale_type)),
        operator_id="cubic.cuda.ggml.embedding.v1",
        grid=(ids.numel(), math.ceil(width / 256), 1), block=(256, 1, 1),
    )
    return _EmbeddingBinding(output, (launch,), payloads=tuple(payloads), metadata=metadata)


def bind_scaled_sharded_embedding(
    ids: Tensor, shards: tuple[tuple[int, QuantizedMatrix], ...], scale: Tensor,
    row_activity: Tensor, output: Tensor,
) -> PackedOperatorBinding:
    return bind_lookup(ids, shards, output, scale=scale, activity=row_activity)


def bind_native_embedding(operators, ids, weight, output):
    if operators.config.compute_mode != "wna8" or int(kind(weight.encoding)) != 8:
        return None
    if output.shape != (*ids.shape, weight.shape[1]):
        raise ValueError("prepared embedding output differs from the original lookup shape")
    carrier = prepare(operators, ((weight,),), ids.device)
    return bind_lookup(ids, ((0, weight),), output, prepared=(carrier,))


def bind_prepared_scaled_sharded_embedding(operators, ids, shards, scale, activity, output):
    if (operators.config.compute_mode != "wna8"
            or any(not isinstance(w, QuantizedMatrix) or w.encoding.format != "ggml-block-v1"
                   or int(kind(w.encoding)) != 8 for _, w in shards)):
        return None
    carriers = tuple(prepare(operators, ((w,),), ids.device) for _, w in shards)
    return bind_lookup(ids, shards, output, scale=scale, activity=activity, prepared=carriers)
