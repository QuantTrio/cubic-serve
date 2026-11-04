"""Caller-owned FP16/BF16 pointwise launches used by Qwen-style decode graphs.

Semantics were triangulated against llama.cpp c841aeeb (``qwen35.cpp`` and
``llama-graph.cpp``), vLLM 73fac1e1 (``qwen3_next.py``, ``activation.py``, and
``layernorm.py``), and SGLang ba433bb4 (``qwen3_5.py`` and
``layernorm_gated.py``).  The common graph is residual add, SwiGLU, sigmoid
attention gating, and norm-before-SiLU gated RMSNorm.  Cubic additionally fixes
the checkpoint-observed BF16 boundaries explicitly.

Instances retain only immutable launch metadata: the caller owns every tensor,
stream, graph, and state.  Construct and warm a fixed ABI before graph capture.
"""

from __future__ import annotations

import math

import torch
import triton  # type: ignore[import-untyped]
import triton.language as tl  # type: ignore[import-untyped]
from torch import Tensor
from triton.language.extra import libdevice  # type: ignore[import-untyped]

from cubic_serve.ops.cuda.activation import activation_type, sigmoid, silu
from cubic_serve.ops.cuda.linear_attention.gated_delta_decode import _gdn_coefficients
from cubic_serve.runtime.cache.quantization import slot_storage_contiguous

_TensorABI = tuple[tuple[int, ...], tuple[int, ...], torch.dtype, torch.device]


@triton.jit
def _update_resident_pages_kernel(
    updates, key, value, written, offsets, source, destination, table, activity,
    KS: tl.constexpr, WS: tl.constexpr, TS: tl.constexpr, BLOCK: tl.constexpr,
    HAS_ACTIVITY: tl.constexpr,
    VS: tl.constexpr = 0,
    KEY_STRIDE: tl.constexpr = 0, VALUE_STRIDE: tl.constexpr = 0,
    WRITTEN_STRIDE: tl.constexpr = 0,
):
    row, block = tl.program_id(0), tl.program_id(1)
    if HAS_ACTIVITY:
        if not tl.load(activity + row):
            return
    physical = tl.load(updates + row * 3).to(tl.int64)
    column = tl.load(updates + row * 3 + 1).to(tl.int64)
    page = tl.load(updates + row * 3 + 2).to(tl.int64)
    element = block * BLOCK + tl.arange(0, BLOCK)
    tl.store(key + page * (KEY_STRIDE if KEY_STRIDE else KS) + element, 0, element < KS)
    value_items: tl.constexpr = VS if VS else KS
    tl.store(value + page * (VALUE_STRIDE if VALUE_STRIDE else value_items) + element, 0, element < value_items)
    tl.store(written + page * (WRITTEN_STRIDE if WRITTEN_STRIDE else WS) + element, 0, element < WS)
    if block == 0:
        tl.store(offsets + physical, 0)
        tl.store(source + physical, page)
        tl.store(destination + physical, page)
        tl.store(table + physical * TS + column, page)


@triton.jit
def _clear_resident_page_field(
    updates, field, activity, WIDTH: tl.constexpr, BLOCK: tl.constexpr,
    HAS_ACTIVITY: tl.constexpr,
    STRIDE: tl.constexpr = 0,
):
    row, block = tl.program_id(0), tl.program_id(1)
    if HAS_ACTIVITY:
        if not tl.load(activity + row):
            return
    page = tl.load(updates + row * 3 + 2).to(tl.int64)
    element = block * BLOCK + tl.arange(0, BLOCK)
    tl.store(field + page * (STRIDE if STRIDE else WIDTH) + element, 0, element < WIDTH)


def update_resident_pages(
    updates: Tensor, key: Tensor, value: Tensor, written: Tensor,
    offsets: Tensor, source: Tensor, destination: Tensor, table: Tensor,
    *, compile_only: bool = False, page_fields: tuple[Tensor, ...] = (),
    update_activity: Tensor | None = None,
) -> None:
    """Clear newly owned pages and publish their mapping on the caller's stream.

    The resident transaction validates unique physical rows, owned pages and
    index bounds before H2D. Preserve that validation and the topology lease:
    direct stores replace deterministic indexed-write sorting only because
    destinations have one owner. Optional typed page fields share the same
    transaction and are cleared before its mapping is published.
    An optional explicit activity vector permits a fixed-address transition
    portfolio. Inactive records never read indices or write any payload/mapping;
    activity is independent of token segmentation and index values. Active
    records retain the caller's validated, uniquely owned update contract.
    Compilation uses tensor metadata only and does not read or mutate storage.
    """
    _require_cuda_contiguous((updates, offsets, source, destination, table))
    if any(t.device != updates.device or not slot_storage_contiguous(t) for t in (key, value, written)):
        raise ValueError("resident cache requires dense same-device page payloads")
    integer_types = (torch.int32, torch.int64)
    if (
        updates.ndim != 2 or updates.shape[1] != 3
        or updates.dtype not in integer_types
        or key.ndim < 2 or value.ndim < 2 or key.shape[0] != value.shape[0]
        or key.dtype not in (torch.float16, torch.bfloat16, torch.uint8)
        or value.dtype not in (torch.float16, torch.bfloat16, torch.uint8)
        or written.ndim < 2 or written.shape[0] != key.shape[0]
        or written.dtype != torch.bool
        or table.ndim != 2 or table.dtype not in integer_types
        or any(t.ndim != 1 or t.numel() != table.shape[0] or t.dtype not in integer_types
               for t in (offsets, source, destination))
        or updates.shape[0] > table.shape[0]
    ):
        raise ValueError("resident page update operands differ from the cache ABI")
    if update_activity is not None:
        _require_cuda_contiguous((updates, update_activity))
        if update_activity.shape != (updates.shape[0],) or update_activity.dtype != torch.bool:
            raise ValueError("resident page update activity differs from its record count")
    if type(page_fields) is not tuple or any(
        field.device != key.device or field.ndim < 2
        or field.shape[0] != key.shape[0] or min(field.shape) <= 0
        or not slot_storage_contiguous(field)
        or field.dtype not in (torch.float16, torch.bfloat16, torch.float32,
                               torch.int32, torch.int64, torch.bool)
        for field in page_fields
    ):
        raise ValueError("resident page fields differ from the typed page ownership ABI")
    count = updates.shape[0]
    if not count:
        return
    ks, vs, ws = math.prod(key.shape[1:]), math.prod(value.shape[1:]), math.prod(written.shape[1:])
    if min(ks, vs, ws, table.shape[1], key.shape[0]) <= 0:
        raise ValueError("resident page storage must be nonempty")
    grid = (count, triton.cdiv(max(ks, vs, ws), 256))
    args = (updates, key, value, written, offsets, source, destination, table, update_activity,
            ks, ws, table.shape[1], 256)
    launch_kwargs = dict(num_warps=4, HAS_ACTIVITY=update_activity is not None)
    # Complete all side payload resets before publishing the new main mapping.
    # Their page index comes from the same validated transaction; no new owner
    # or per-model lookup is introduced.
    for field in page_fields:
        width = math.prod(field.shape[1:])
        field_grid = (count, triton.cdiv(width, 256))
        field_args = (updates, field, update_activity, width, 256)
        if compile_only:
            artifact = _clear_resident_page_field.warmup(*field_args, grid=field_grid,
                                                         STRIDE=field.stride(0), **launch_kwargs)
            artifact._init_handles()
        else:
            _clear_resident_page_field[field_grid](*field_args, STRIDE=field.stride(0), **launch_kwargs)
    launch_kwargs.update(KEY_STRIDE=key.stride(0), VALUE_STRIDE=value.stride(0),
                         WRITTEN_STRIDE=written.stride(0))
    if compile_only:
        artifact = _update_resident_pages_kernel.warmup(*args, grid=grid, VS=vs, **launch_kwargs)
        artifact._init_handles()
    else:
        _update_resident_pages_kernel[grid](*args, VS=vs, **launch_kwargs)


@triton.jit
def _advance_resident_inputs_kernel(
    positions, offsets, lengths, sampled, local_ids, embedding_mask, ordinals,
    input_tokens, input_positions, prompt_steps, step, logical_steps,
    activity, M: tl.constexpr, AXES: tl.constexpr, STRIDE: tl.constexpr,
    FIRST: tl.constexpr, START: tl.constexpr, STOP: tl.constexpr,
    PROGRAM: tl.constexpr, STRIDED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = (row < M) & (tl.load(activity + row, row < M, other=0) != 0)
    if PROGRAM:
        index = tl.load(step)
        prompt = active & (index < tl.load(prompt_steps + row, active, other=0))
    delta = tl.full((BLOCK,), 1, tl.int32)
    if STRIDED:
        # A prefill segment owns multiple physical token rows. Its logical
        # cursor advances by that segment's length; decode still advances one.
        span = tl.load(logical_steps + row, prompt, other=1)
        delta = tl.where(prompt, span, 1)
    for axis in tl.static_range(AXES):
        address = positions + axis * STRIDE + row
        value = tl.load(address, active, other=0)
        next_position = value + 1
        if PROGRAM:
            explicit_position = tl.load(
                input_positions + (index * AXES + axis) * M + row,
                prompt, other=0,
            )
            next_position = tl.where(prompt, explicit_position, next_position)
        tl.store(address, next_position, active)
    offset = tl.load(offsets + row, active, other=0)
    length = tl.load(lengths + row, active, other=0)
    ordinal = tl.load(ordinals + row, active, other=0)
    tl.store(offsets + row, offset + delta, active)
    tl.store(lengths + row, length + delta, active)
    next_ordinal = ordinal + 1
    if PROGRAM:
        next_ordinal = tl.where(prompt, ordinal, next_ordinal)
    tl.store(ordinals + row, next_ordinal, active)
    if FIRST:
        token = tl.load(sampled + row, active, other=0)
        if PROGRAM:
            explicit_token = tl.load(input_tokens + index * M + row, prompt, other=0)
            token = tl.where(prompt, explicit_token, token)
        owned = (token >= START) & (token < STOP)
        tl.store(local_ids + row, tl.where(owned, token - START, 0), active)
        tl.store(embedding_mask + row, owned, active)


def advance_resident_inputs(
    positions: Tensor, offsets: Tensor, lengths: Tensor, sampled: Tensor,
    local_ids: Tensor, ordinals: Tensor, activity: Tensor,
    embedding_mask: Tensor | None, *, vocab_start: int = 0, vocab_stop: int = 0,
    input_tokens: Tensor | None = None, input_positions: Tensor | None = None,
    prompt_steps: Tensor | None = None, step: Tensor | None = None,
    logical_steps: Tensor | None = None,
) -> None:
    """Startup-bound integer update; page resets must execute afterwards.

    Explicit activity owns rows independently of sequence segmentation. Each
    position axis advances independently; this never reconstructs positions
    from sequence lengths. The executor captures this launch before readiness.
    Optional positive logical steps come from validated owner segments. They
    affect logical prefill progress only; explicit M-RoPE axes remain separate.
    """
    tensors = (positions, offsets, lengths, sampled, local_ids, ordinals, activity)
    _require_cuda_contiguous(tensors + (() if embedding_mask is None else (embedding_mask,)))
    m = activity.numel()
    if (
        m <= 0 or positions.ndim != 2 or positions.shape[1] != m
        or any(t.numel() != m for t in tensors[1:])
        or any(t.dtype not in (torch.int32, torch.int64) for t in tensors[:-1])
        or activity.dtype not in (torch.bool, torch.int32, torch.int64)
        or (embedding_mask is not None and (
            embedding_mask.numel() != m
            or embedding_mask.dtype not in (torch.float16, torch.bfloat16)
            or not 0 <= vocab_start < vocab_stop
        ))
    ):
        raise ValueError("invalid resident input update ABI")
    program = (input_tokens, input_positions, prompt_steps, step)
    has_program = any(value is not None for value in program)
    if has_program:
        if (input_tokens is None or input_positions is None
                or prompt_steps is None or step is None):
            raise ValueError("resident input program requires all metadata tensors")
        validated_program = (input_tokens, input_positions, prompt_steps, step)
        _require_cuda_contiguous(validated_program)
        if (
            input_tokens.ndim != 2 or input_tokens.shape[1] != m
            or input_positions.shape != (input_tokens.shape[0], positions.shape[0], m)
            or prompt_steps.shape != (m,) or step.numel() != 1
            or input_tokens.shape[0] <= 0
            or input_positions.dtype != positions.dtype
            or any(value.dtype not in (torch.int32, torch.int64) for value in validated_program)
        ):
            raise ValueError("invalid resident input program ABI")
    if logical_steps is not None:
        _require_cuda_contiguous((positions, logical_steps))
        if (not has_program or logical_steps.shape != (m,)
                or logical_steps.dtype not in (torch.int32, torch.int64)):
            raise ValueError("logical steps require a matching resident input program")
    _advance_resident_inputs_kernel[(triton.cdiv(m, 256),)](
        positions, offsets, lengths, sampled, local_ids,
        ordinals if embedding_mask is None else embedding_mask, ordinals,
        input_tokens if has_program else ordinals,
        input_positions if has_program else ordinals,
        prompt_steps if has_program else ordinals, step if has_program else ordinals,
        logical_steps if logical_steps is not None else ordinals,
        activity,
        m, positions.shape[0], positions.stride(0), embedding_mask is not None,
        vocab_start, vocab_stop, has_program, logical_steps is not None, 256, num_warps=4,
    )


@triton.jit  # type: ignore[untyped-decorator]
def _binary_bf16_kernel(  # type: ignore[no-untyped-def]
    left,
    right,
    output,
    ELEMENTS: tl.constexpr,
    BLOCK: tl.constexpr,
    KIND: tl.constexpr,
    ACT_TYPE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < ELEMENTS
    lhs = tl.load(left + offsets, mask=mask).to(tl.float32)
    rhs = tl.load(right + offsets, mask=mask).to(tl.float32)
    if KIND == 0:
        result = lhs + rhs
    elif KIND == 1:
        # Round the activated gate to the output dtype before up multiplication.
        activated = silu(lhs, ACT_TYPE).to(output.dtype.element_ty)
        result = activated * rhs.to(output.dtype.element_ty)
    else:
        # Full-attention sigmoid has its own activation-dtype rounding boundary.
        activated = sigmoid(rhs, ACT_TYPE).to(output.dtype.element_ty)
        result = lhs.to(output.dtype.element_ty) * activated
    tl.store(output + offsets, result, mask=mask)


@triton.jit  # type: ignore[untyped-decorator]
def _packed_swiglu_bf16_kernel(  # type: ignore[no-untyped-def]
    packed,
    output,
    ELEMENTS: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    ACT_TYPE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < ELEMENTS
    row = offsets // WIDTH
    column = offsets % WIDTH
    gate_offsets = row * (2 * WIDTH) + column
    gate = tl.load(packed + gate_offsets, mask=mask).to(tl.float32)
    up = tl.load(packed + gate_offsets + WIDTH, mask=mask)
    # Preserve Cubic's explicit activation-dtype boundary between SiLU and multiplication.
    activated = silu(gate, ACT_TYPE).to(output.dtype.element_ty)
    tl.store(output + offsets, activated * up, mask=mask)


@triton.jit  # type: ignore[untyped-decorator]
def _packed_gdn_projection_split_kernel(  # type: ignore[no-untyped-def]
    packed,
    a,
    b,
    z,
    ELEMENTS: tl.constexpr,
    A_WIDTH: tl.constexpr,
    B_WIDTH: tl.constexpr,
    Z_WIDTH: tl.constexpr,
    PACKED_WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < ELEMENTS
    row = offsets // PACKED_WIDTH
    column = offsets % PACKED_WIDTH
    value = tl.load(packed + offsets, mask=mask)
    a_mask = mask & (column < A_WIDTH)
    b_column = column - A_WIDTH
    b_mask = mask & (b_column >= 0) & (b_column < B_WIDTH)
    z_column = b_column - B_WIDTH
    z_mask = mask & (z_column >= 0) & (z_column < Z_WIDTH)
    tl.store(a + row * A_WIDTH + column, value, mask=a_mask)
    tl.store(b + row * B_WIDTH + b_column, value, mask=b_mask)
    tl.store(z + row * Z_WIDTH + z_column, value, mask=z_mask)


@triton.jit  # type: ignore[untyped-decorator]
def _q_gate_split_kernel(  # type: ignore[no-untyped-def]
    projected,
    query,
    gate,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    ELEMENTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < ELEMENTS
    dim = offsets % DIM
    row_head = offsets // DIM
    source = row_head * (2 * DIM) + dim
    tl.store(query + offsets, tl.load(projected + source, mask=mask), mask=mask)
    tl.store(gate + offsets, tl.load(projected + source + DIM, mask=mask), mask=mask)


@triton.jit  # type: ignore[untyped-decorator]
def _gated_rms_norm_kernel(  # type: ignore[no-untyped-def]
    inputs,
    gate,
    weight,
    output,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    EPSILON: tl.constexpr,
    WEIGHT_ACTIVATION: tl.constexpr,
    ACT_TYPE: tl.constexpr,
    SIGMOID_GATE: tl.constexpr = False,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    mask = columns < WIDTH
    offset = row * WIDTH + columns
    values = tl.load(inputs + offset, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(values * values, axis=0) / WIDTH
    normalized = (values * libdevice.rsqrt(variance + EPSILON)).to(output.dtype.element_ty)
    normalized_f32 = normalized.to(tl.float32)
    weights = tl.load(weight + columns, mask=mask, other=0.0).to(tl.float32)
    weighted = normalized_f32 * weights
    if WEIGHT_ACTIVATION:
        weighted = weighted.to(output.dtype.element_ty).to(tl.float32)
    gate_f32 = tl.load(gate + offset, mask=mask, other=0.0).to(tl.float32)
    # Preserve checkpoint-typed weighting independently of activation policy.
    activated = sigmoid(gate_f32, ACT_TYPE) if SIGMOID_GATE else silu(gate_f32, ACT_TYPE)
    tl.store(output + offset, weighted * activated, mask=mask)


def _abi(tensor: Tensor) -> _TensorABI:
    if not isinstance(tensor, Tensor):
        raise TypeError("CUDA primitive operands must be tensors")
    return tuple(tensor.shape), tensor.stride(), tensor.dtype, tensor.device


def _require_cuda_contiguous(tensors: tuple[Tensor, ...]) -> None:
    if any(t.device.type != "cuda" or not t.is_contiguous() for t in tensors):
        raise ValueError("all operands must be contiguous CUDA tensors")
    if len({tensor.device for tensor in tensors}) != 1:
        raise ValueError("all operands must share one CUDA device")


def _same_activation_dtype(tensors: tuple[Tensor, ...]) -> bool:
    return tensors[0].dtype in (torch.float16, torch.bfloat16) and all(
        t.dtype == tensors[0].dtype for t in tensors
    )


def _dtype_name(dtype: torch.dtype) -> str:
    return {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32"}[dtype]


class _BinaryPrimitive:
    def __init__(self, left: Tensor, right: Tensor, output: Tensor, *, kind: int) -> None:
        operands = (left, right, output)
        _require_cuda_contiguous(operands)
        if (
            left.ndim < 1
            or left.shape != right.shape
            or output.shape != left.shape
            or left.numel() <= 0
            or not _same_activation_dtype(operands)
        ):
            raise ValueError("expected non-empty, equal-shape CUDA FP16/BF16 operands")
        self._abis = tuple(_abi(tensor) for tensor in operands)
        self._meta = left.numel(), kind
        self._activation_type = activation_type(output.dtype, output.device)

    @property
    def launch_signature(self) -> str:
        elements, kind = self._meta
        return (
            f"elements={elements},kind={kind},{_dtype_name(self._abis[0][2])},"
            f"act={self._activation_type}"
        )

    def launch(self, left: Tensor, right: Tensor, output: Tensor) -> None:
        operands = (left, right, output)
        if tuple(_abi(tensor) for tensor in operands) != self._abis:
            raise ValueError("pointwise operands do not match the fixed launch ABI")
        elements, kind = self._meta
        block = 256
        _binary_bf16_kernel[(triton.cdiv(elements, block),)](
            *operands,
            ELEMENTS=elements,
            BLOCK=block,
            KIND=kind,
            num_warps=4,
            ACT_TYPE=self._activation_type,
        )


class QwenAttentionQGateSplitPrimitive:
    """Split checkpoint-packed ``[C,QH,2,D]`` projection into compact rows."""

    def __init__(self, projected: Tensor, query: Tensor, gate: Tensor) -> None:
        operands = projected, query, gate
        _require_cuda_contiguous(operands)
        if (
            projected.ndim != 4
            or projected.shape[2] != 2
            or min(projected.shape) <= 0
            or query.shape != (projected.shape[0], projected.shape[1], projected.shape[3])
            or gate.shape != query.shape
            or not _same_activation_dtype(operands)
        ):
            raise ValueError("expected CUDA FP16/BF16 [C,QH,2,D] and two [C,QH,D] outputs")
        self._abis = tuple(_abi(tensor) for tensor in operands)
        self._meta = query.numel(), query.shape[1], query.shape[2]

    @property
    def launch_signature(self) -> str:
        elements, heads, dim = self._meta
        return (
            f"elements={elements},qh={heads},d={dim},packed-q-gate,{_dtype_name(self._abis[0][2])}"
        )

    def launch(self, projected: Tensor, query: Tensor, gate: Tensor) -> None:
        operands = projected, query, gate
        if tuple(_abi(tensor) for tensor in operands) != self._abis:
            raise ValueError("Q/gate split operands do not match the fixed launch ABI")
        elements, heads, dim = self._meta
        block = 256
        _q_gate_split_kernel[(triton.cdiv(elements, block),)](
            *operands,
            HEADS=heads,
            DIM=dim,
            ELEMENTS=elements,
            BLOCK=block,
            num_warps=4,
        )


class ResidualAddPrimitive(_BinaryPrimitive):
    """Add two residual branches into caller-owned activation storage."""

    def __init__(self, residual: Tensor, update: Tensor, output: Tensor) -> None:
        super().__init__(residual, update, output, kind=0)


class SwiGLUPrimitive(_BinaryPrimitive):
    """Round SiLU to the activation dtype before multiplication and output rounding."""

    def __init__(self, gate: Tensor, up: Tensor, output: Tensor) -> None:
        super().__init__(gate, up, output, kind=1)


class PackedSwiGLUPrimitive:
    """Split contiguous ``[..., gate, up]`` and apply Cubic's rounded SwiGLU."""

    def __init__(self, packed: Tensor, output: Tensor) -> None:
        operands = packed, output
        _require_cuda_contiguous(operands)
        if (
            packed.ndim < 2
            or packed.shape[-1] <= 0
            or packed.shape[-1] % 2
            or output.shape != (*packed.shape[:-1], packed.shape[-1] // 2)
            or not _same_activation_dtype(operands)
            or packed.untyped_storage().data_ptr() == output.untyped_storage().data_ptr()
        ):
            raise ValueError("expected non-aliasing contiguous CUDA FP16/BF16 [...,2D] -> [...,D]")
        self._abis = tuple(_abi(tensor) for tensor in operands)
        self._meta = output.numel(), output.shape[-1]
        self._activation_type = activation_type(output.dtype, output.device)

    @property
    def launch_signature(self) -> str:
        elements, width = self._meta
        return (
            f"elements={elements},width={width},packed-gate-up,{_dtype_name(self._abis[0][2])},"
            f"act={self._activation_type}"
        )

    def launch(self, packed: Tensor, output: Tensor) -> None:
        operands = packed, output
        if tuple(_abi(tensor) for tensor in operands) != self._abis:
            raise ValueError("packed SwiGLU operands do not match the fixed launch ABI")
        elements, width = self._meta
        block = 256
        _packed_swiglu_bf16_kernel[(triton.cdiv(elements, block),)](
            *operands,
            ELEMENTS=elements,
            WIDTH=width,
            BLOCK=block,
            ACT_TYPE=self._activation_type,
            num_warps=4,
        )


@triton.jit
def _packed_gdn_split_coefficients(
    packed,
    a,
    b,
    z,
    coefficients,
    a_log,
    dt_bias,
    HEADS: tl.constexpr,
    Z_WIDTH: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    ACT_TYPE: tl.constexpr,
):
    row, tile = tl.program_id(0), tl.program_id(1)
    columns = tile * 256 + tl.arange(0, 256)
    stride: tl.constexpr = 2 * HEADS + Z_WIDTH
    value = tl.load(packed + row * stride + 2 * HEADS + columns, columns < Z_WIDTH, other=0)
    tl.store(z + row * Z_WIDTH + columns, value, columns < Z_WIDTH)
    if tile == 0:
        heads = tl.arange(0, HEAD_BLOCK)
        mask = heads < HEADS
        av = tl.load(packed + row * stride + heads, mask, other=0)
        bv = tl.load(packed + row * stride + HEADS + heads, mask, other=0)
        # Preserve raw boundaries for observers and the unfused reference path.
        tl.store(a + row * HEADS + heads, av, mask)
        tl.store(b + row * HEADS + heads, bv, mask)
        beta, decay = _gdn_coefficients(
            av, bv, tl.load(a_log + heads, mask, other=0),
            tl.load(dt_bias + heads, mask, other=0), ACT_TYPE, b.dtype.element_ty,
        )
        tl.store(coefficients + (row * HEADS + heads) * 2, beta, mask)
        tl.store(coefficients + (row * HEADS + heads) * 2 + 1, decay, mask)


class PackedGDNProjectionSplitPrimitive:
    """Copy contiguous ``[A,B,Z]`` activation segments into their semantic buffers."""

    def __init__(
        self,
        packed: Tensor,
        a: Tensor,
        b: Tensor,
        z: Tensor,
        *,
        coefficients: Tensor | None = None,
        a_log: Tensor | None = None,
        dt_bias: Tensor | None = None,
    ) -> None:
        operands = packed, a, b, z
        _require_cuda_contiguous(operands)
        if (
            any(tensor.ndim != 2 or min(tensor.shape) <= 0 for tensor in operands)
            or not _same_activation_dtype(operands)
            or any(tensor.shape[0] != packed.shape[0] for tensor in (a, b, z))
            or packed.shape[1] != a.shape[1] + b.shape[1] + z.shape[1]
            or len({tensor.untyped_storage().data_ptr() for tensor in operands}) != 4
        ):
            raise ValueError("expected non-aliasing contiguous CUDA FP16/BF16 [A,B,Z] segments")
        self._abis = tuple(_abi(tensor) for tensor in operands)
        self._meta = packed.numel(), a.shape[1], b.shape[1], z.shape[1], packed.shape[1]
        self._coefficient_abis: tuple[_TensorABI, ...] | None = None
        self._activation_type = activation_type(packed.dtype, packed.device)
        if coefficients is not None:
            if a_log is None or dt_bias is None:
                raise ValueError("coefficient preparation requires A_log and dt_bias")
            extra = coefficients, a_log, dt_bias
            _require_cuda_contiguous(extra)
            if (
                a.shape != b.shape
                or coefficients.shape != (*a.shape, 2)
                or coefficients.dtype is not torch.float32
                or a_log.shape != (a.shape[1],)
                or dt_bias.shape != a_log.shape
                or a_log.dtype not in (packed.dtype, torch.float32)
                or dt_bias.dtype != packed.dtype
                or any(t.device != packed.device for t in extra)
                or len({t.untyped_storage().data_ptr() for t in (*operands, *extra)}) != 7
            ):
                raise ValueError("invalid GDN coefficient preparation ABI")
            self._coefficient_abis = tuple(_abi(t) for t in extra)
        elif a_log is not None or dt_bias is not None:
            raise ValueError("coefficient operands require an output")

    @property
    def launch_signature(self) -> str:
        elements, a_width, b_width, z_width, _ = self._meta
        return (
            f"elements={elements},a={a_width},b={b_width},z={z_width},"
            f"packed-gdn,{_dtype_name(self._abis[0][2])},coefficients={self._coefficient_abis},act={self._activation_type}"
        )

    def launch(
        self,
        packed: Tensor,
        a: Tensor,
        b: Tensor,
        z: Tensor,
        *,
        coefficients: Tensor | None = None,
        a_log: Tensor | None = None,
        dt_bias: Tensor | None = None,
    ) -> None:
        operands = packed, a, b, z
        if tuple(_abi(tensor) for tensor in operands) != self._abis:
            raise ValueError("packed GDN split operands do not match the fixed launch ABI")
        elements, a_width, b_width, z_width, packed_width = self._meta
        if self._coefficient_abis is not None:
            if coefficients is None or a_log is None or dt_bias is None:
                raise ValueError("missing bound GDN coefficients")
            if tuple(_abi(t) for t in (coefficients, a_log, dt_bias)) != self._coefficient_abis:
                raise ValueError("GDN coefficient operands differ from bound ABI")
            _packed_gdn_split_coefficients[(packed.shape[0], triton.cdiv(z_width, 256))](
                packed,
                a,
                b,
                z,
                coefficients,
                a_log,
                dt_bias,
                HEADS=a_width,
                Z_WIDTH=z_width,
                HEAD_BLOCK=triton.next_power_of_2(a_width),
                ACT_TYPE=self._activation_type,
                num_warps=4,
            )
            return
        if any(t is not None for t in (coefficients, a_log, dt_bias)):
            raise ValueError("unbound GDN coefficient operands")
        block = 256
        _packed_gdn_projection_split_kernel[(triton.cdiv(elements, block),)](
            *operands,
            ELEMENTS=elements,
            A_WIDTH=a_width,
            B_WIDTH=b_width,
            Z_WIDTH=z_width,
            PACKED_WIDTH=packed_width,
            BLOCK=block,
            num_warps=4,
        )


class SigmoidOutputGatePrimitive(_BinaryPrimitive):
    """Round sigmoid to the activation dtype before multiplication and output rounding."""

    def __init__(self, inputs: Tensor, gate: Tensor, output: Tensor) -> None:
        super().__init__(inputs, gate, output, kind=2)


class GatedRMSNormPrimitive:
    """Last-dimension FP32 RMSNorm followed by the declared activation gate."""

    def __init__(
        self,
        inputs: Tensor,
        gate: Tensor,
        weight: Tensor,
        output: Tensor,
        *,
        epsilon: float,
        activation: str = "silu",
    ) -> None:
        operands = (inputs, gate, weight, output)
        if activation not in ("silu", "sigmoid"):
            raise ValueError("gated RMSNorm activation must be silu or sigmoid")
        self._gate = activation
        _require_cuda_contiguous(operands)
        if (
            inputs.ndim < 2
            or inputs.shape != gate.shape
            or output.shape != inputs.shape
            or weight.ndim != 1
            or weight.shape[0] != inputs.shape[-1]
            or inputs.numel() <= 0
            or not _same_activation_dtype((inputs, gate, output))
            or weight.dtype not in (inputs.dtype, torch.float32)
            or not math.isfinite(epsilon)
            or epsilon <= 0.0
        ):
            raise ValueError("operands do not match the gated RMSNorm activation/FP32 ABI")
        width = inputs.shape[-1]
        self._abis = tuple(_abi(tensor) for tensor in operands)
        self._meta = inputs.numel() // width, width, epsilon, weight.dtype is inputs.dtype
        self._block = triton.next_power_of_2(width)
        self._activation_type = activation_type(output.dtype, output.device)

    @property
    def launch_signature(self) -> str:
        rows, width, epsilon, _ = self._meta
        weight_dtype = _dtype_name(self._abis[2][2])
        signature = (
            f"rows={rows},width={width},eps={epsilon.hex()},weight={weight_dtype},"
            f"out={_dtype_name(self._abis[0][2])},act={self._activation_type}"
        )
        return signature if self._gate == "silu" else signature + ",gate=sigmoid"

    def launch(self, inputs: Tensor, gate: Tensor, weight: Tensor, output: Tensor) -> None:
        operands = (inputs, gate, weight, output)
        if tuple(_abi(tensor) for tensor in operands) != self._abis:
            raise ValueError("gated RMSNorm operands do not match the fixed launch ABI")
        rows, width, epsilon, weight_activation = self._meta
        _gated_rms_norm_kernel[(rows,)](
            *operands,
            WIDTH=width,
            BLOCK=self._block,
            EPSILON=epsilon,
            WEIGHT_ACTIVATION=weight_activation,
            ACT_TYPE=self._activation_type,
            SIGMOID_GATE=self._gate == "sigmoid",
            num_warps=min(max(self._block // 256, 1), 8),
        )


# Compatibility imports retain the BF16 baseline call sites and signatures.
BF16ResidualAddPrimitive = ResidualAddPrimitive
BF16SwiGLUPrimitive = SwiGLUPrimitive
BF16PackedSwiGLUPrimitive = PackedSwiGLUPrimitive
BF16PackedGDNProjectionSplitPrimitive = PackedGDNProjectionSplitPrimitive
BF16SigmoidOutputGatePrimitive = SigmoidOutputGatePrimitive


__all__ = (
    "BF16PackedGDNProjectionSplitPrimitive",
    "BF16PackedSwiGLUPrimitive",
    "BF16ResidualAddPrimitive",
    "BF16SigmoidOutputGatePrimitive",
    "BF16SwiGLUPrimitive",
    "GatedRMSNormPrimitive",
    "PackedGDNProjectionSplitPrimitive",
    "PackedSwiGLUPrimitive",
    "QwenAttentionQGateSplitPrimitive",
    "ResidualAddPrimitive",
    "SigmoidOutputGatePrimitive",
    "SwiGLUPrimitive",
)
