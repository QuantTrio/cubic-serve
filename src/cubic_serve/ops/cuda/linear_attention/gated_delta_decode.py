"""Caller-owned segment-aware Gated Delta Net CUDA launches.

The shared recurrence and convolution semantics were triangulated against
llama.cpp c841aeeb (``qwen35.cpp``), vLLM 0.27.1
(``qwen_gdn_linear_attn.py``, ``causal_conv1d.py``), and SGLang a9d5ca72
(``gdn_backend.py``, ``gdn_triton.py``, ``causal_conv1d_triton.py``).

These primitives take explicit source and destination indices so an arena can
perform COW for shared state and update exclusive state in place.  They own no
tensors, stream, graph, workspace, or state.  The caller must range- and
generation-check indices, keep source slots readable, lease every destination, and
ensure no destination aliases another segment's source/destination.  A positive
``segment_lengths`` entry starts one contiguous segment; its following zero entries
are continuations.  Zero entries start no work and never write state.  The sole
permitted equality is one exclusively owned segment or padding scratch slot.
"""

from __future__ import annotations

import math

import torch
import triton  # type: ignore[import-untyped]
import triton.language as tl  # type: ignore[import-untyped]
from torch import Tensor
from triton.language.extra import libdevice  # type: ignore[import-untyped]

from cubic_serve.ops.cuda.activation import activation_type, sigmoid
from cubic_serve.ops.cuda.activation import silu as activation_silu
from cubic_serve.ops.cuda.launch_cost import (
    MatmulGeometry,
    current_linear_cost,
    device_resources,
    matmul_geometries,
    select_launch,
    startup_operator_choices,
)
from cubic_serve.ops.strategies import (
    attention_precision,
    gdn_parameters,
    parameter_overrides,
)

INDEX_OWNERSHIP_CONTRACT = "cubic.arena.generation-checked-cow-or-exclusive-indices.v2"
_VALUE_TILE = 8  # Expose independent FP32 state stripes without changing reductions.


@triton.jit
def _rn_mul(left, right):  # type: ignore[no-untyped-def]
    """IEEE FP32 multiply without Triton's implicit FTZ contraction."""

    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [left, right],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _rn_add(left, right):  # type: ignore[no-untyped-def]
    """IEEE FP32 addition without flushing a gradual-underflow result."""

    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [left, right],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _rn_sub(left, right):  # type: ignore[no-untyped-def]
    return tl.inline_asm_elementwise(
        "sub.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [left, right],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _rn_div(left, right):  # type: ignore[no-untyped-def]
    return tl.inline_asm_elementwise(
        "div.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [left, right],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit  # type: ignore[untyped-decorator]
def _gated_delta_qk_normalize_kernel(  # type: ignore[no-untyped-def]
    packed_qkv, normalized_qk,
    EPSILON: tl.constexpr,
    Q_HEADS: tl.constexpr,
    K_DIM: tl.constexpr,
    PACKED_DIM: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row_head = tl.program_id(0)
    row, head = row_head // Q_HEADS, row_head % Q_HEADS
    keys = tl.arange(0, BLOCK_K)
    mask = keys < K_DIM
    packed = packed_qkv + row * PACKED_DIM
    q_pointer = packed + head * K_DIM + keys
    k_pointer = packed + Q_HEADS * K_DIM + head * K_DIM + keys
    q = tl.load(q_pointer, mask=mask, other=0.0).to(tl.float32)
    k = tl.load(k_pointer, mask=mask, other=0.0).to(tl.float32)
    q = q / tl.sqrt(tl.sum(_rn_mul(q, q)) + EPSILON)
    k = k / tl.sqrt(tl.sum(_rn_mul(k, k)) + EPSILON)
    normalized = normalized_qk + row * PACKED_DIM
    tl.store(normalized + head * K_DIM + keys, q, mask=mask)
    tl.store(normalized + Q_HEADS * K_DIM + head * K_DIM + keys, k, mask=mask)


@triton.jit
def _state_row(state, index, OWNER_STRIDE:tl.constexpr, LAYERS:tl.constexpr,
               LAYER_STRIDE:tl.constexpr):
    return state + (index // LAYERS) * OWNER_STRIDE + (index % LAYERS) * LAYER_STRIDE


@triton.jit  # type: ignore[untyped-decorator]
def _causal_conv_decode_kernel(  # type: ignore[no-untyped-def]
    x, weight, state, src_indices, dst_indices, segment_lengths, output,
    CAPACITY: tl.constexpr,
    CHANNELS: tl.constexpr, WIDTH: tl.constexpr,
    BLOCK_C: tl.constexpr, BLOCK_T: tl.constexpr,
    SILU: tl.constexpr, CHECKPOINTS: tl.constexpr,
    ACT_TYPE: tl.constexpr,
    STATE_STRIDE: tl.constexpr,
    STATE_LAYERS: tl.constexpr = 1, STATE_LAYER_STRIDE: tl.constexpr = 0,
):
    row = tl.program_id(0)
    segment_length = tl.load(segment_lengths + row).to(tl.int32)
    segment_length = tl.minimum(tl.maximum(segment_length, 0), CAPACITY - row)
    if segment_length == 0:
        return

    channels = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    times = tl.arange(0, BLOCK_T)
    channel_mask = channels < CHANNELS
    history_mask = times < WIDTH - 1
    src = tl.load(src_indices + row).to(tl.int64)
    dst = tl.load(dst_indices + row).to(tl.int64)

    state_base = _state_row(state,src,STATE_STRIDE,STATE_LAYERS,STATE_LAYER_STRIDE) + channels[:, None] * (WIDTH - 1)
    weights = tl.load(
        weight + channels[:, None] * WIDTH + times[None, :],
        mask=channel_mask[:, None] & history_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    current_weight = tl.load(
        weight + channels * WIDTH + WIDTH - 1, mask=channel_mask
    ).to(tl.float32)
    # The segment exclusively owns its write destinations, which may include
    # its source. Carry the short, already-rounded input history in registers
    # so publishing a checkpoint cannot overwrite a later step's operands.
    history = tl.load(
        state_base + times[None, :],
        mask=channel_mask[:, None] & history_mask[None, :], other=0.0,
    ).to(tl.float32)
    shift_indices = tl.broadcast_to(
        tl.minimum(times + 1, WIDTH - 2)[None, :], (BLOCK_C, BLOCK_T)
    )
    step = 0
    while step < segment_length:
        current = tl.load(
            x + (row + step) * CHANNELS + channels, mask=channel_mask, other=0.0
        ).to(tl.float32)
        result = tl.sum(history * weights, axis=1) + current * current_weight
        if SILU:
            # Activation arithmetic is bound separately from convolution reduction.
            result = activation_silu(result, ACT_TYPE).to(tl.float32)
        tl.store(
            output + (row + step) * CHANNELS + channels,
            result,
            mask=channel_mask,
        )
        shifted = tl.gather(history, shift_indices, axis=1)
        history = tl.where(
            times[None, :] == WIDTH - 2, current[:, None], shifted
        )
        history = tl.where(history_mask[None, :], history, 0.0)
        if CHECKPOINTS:
            checkpoint_dst = tl.load(dst_indices + row + step).to(tl.int64)
            checkpoint = (_state_row(state,checkpoint_dst,STATE_STRIDE,STATE_LAYERS,STATE_LAYER_STRIDE)
                          + channels[:, None] * (WIDTH - 1))
            tl.store(
                checkpoint + times[None, :], history,
                mask=channel_mask[:, None] & history_mask[None, :],
            )
        step += 1

    if not CHECKPOINTS:
        destination = _state_row(state,dst,STATE_STRIDE,STATE_LAYERS,STATE_LAYER_STRIDE) + channels[:, None] * (WIDTH - 1)
        tl.store(destination + times[None, :], history,
                 mask=channel_mask[:, None] & history_mask[None, :])


@triton.jit
def _low_fma(left, right, addend):  # type: ignore[no-untyped-def]
    # Packed BF16 FMA is available on SM80+, unlike standalone BF16 mul.
    # This implements real low-precision arithmetic across the baseline.
    if left.dtype == tl.bfloat16:
        return tl.inline_asm_elementwise(
            "fma.rn.bf16x2 $0, $1, $2, $3;", "=r,r,r,r",
            [left, right.to(left.dtype), addend.to(left.dtype)],
            dtype=left.dtype, is_pure=True, pack=2,
        )
    else:
        return tl.inline_asm_elementwise(
            "fma.rn.f16x2 $0, $1, $2, $3;", "=r,r,r,r",
            [left, right.to(left.dtype), addend.to(left.dtype)],
            dtype=left.dtype, is_pure=True, pack=2,
        )


@triton.jit
def _low_mul(left, right):  # type: ignore[no-untyped-def]
    return _low_fma(left, right, tl.full(left.shape, -0.0, left.dtype))


@triton.jit  # type: ignore[untyped-decorator]
def _native_state_step(state, q, k, v, beta, decay, SCALE: tl.constexpr):  # type: ignore[no-untyped-def]
    lanes = tl.arange(0, 16)
    low_state = state.to(q.dtype)
    decayed = _low_mul(low_state, decay.to(q.dtype))
    k_column = tl.where(lanes[None, :] == 0, k[:, None], 0).to(q.dtype)
    memory = tl.sum(tl.dot(decayed, k_column, out_dtype=tl.float32), 1)
    residual = _low_fma(memory.to(q.dtype), tl.full(v.shape, -1.0, q.dtype), v)
    residual = _low_mul(residual, beta.to(q.dtype))
    left = tl.where(lanes[None, :] == 0, residual[:, None], 0).to(q.dtype)
    right = tl.where(lanes[:, None] == 0, k[None, :], 0).to(q.dtype)
    # A single low-operand product is exact in FP32 for normal finite products.
    # MMA with a nonzero accumulator does not promise IEEE RN addition. At a
    # BF16 midpoint that difference changes durable state. Keep the product in
    # FP32 registers and explicitly RN-add, without an intermediate low cast.
    update = tl.dot(left, right, out_dtype=tl.float32)
    return _rn_add(decayed.to(tl.float32), update)


@triton.jit
def _gdn_coefficients(a, b, a_log, dt_bias, ACT_TYPE: tl.constexpr, B_TYPE: tl.constexpr):
    # Shared by inline recurrence and the packed-projection preparation stage.
    # Beta's low rounding and decay's FP32 result are mathematical boundaries.
    beta = sigmoid(b.to(tl.float32), ACT_TYPE).to(B_TYPE).to(tl.float32)
    alpha = a.to(tl.float32) + dt_bias.to(tl.float32)
    softplus = tl.where(alpha <= 20.0, libdevice.log1p(libdevice.exp(alpha)), alpha)
    decay = libdevice.exp(-libdevice.exp(a_log.to(tl.float32)) * softplus)
    return beta, decay


@triton.jit  # type: ignore[untyped-decorator]
def _gated_delta_decode_kernel(  # type: ignore[no-untyped-def]
    packed_qkv, normalized_qk, a, b, a_log, dt_bias, state, src_indices, dst_indices,
    segment_lengths, output,
    CAPACITY: tl.constexpr,
    SCALE: tl.constexpr,
    Q_HEADS: tl.constexpr, V_HEADS: tl.constexpr,
    K_DIM: tl.constexpr, V_DIM: tl.constexpr, PACKED_DIM: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr, CHECKPOINTS: tl.constexpr,
    STATE_BF16: tl.constexpr,
    NATIVE: tl.constexpr,
    ACT_TYPE: tl.constexpr,
    STATE_STRIDE: tl.constexpr,
    STATE_LAYERS: tl.constexpr = 1, STATE_LAYER_STRIDE: tl.constexpr = 0,
    coefficients=None, PRECOMPUTED: tl.constexpr = False,
):
    MATH_TYPE: tl.constexpr = packed_qkv.dtype.element_ty if NATIVE else tl.float32
    value_block = tl.program_id(0)
    row_head = tl.program_id(1)
    row, v_head = row_head // V_HEADS, row_head % V_HEADS
    segment_length = tl.load(segment_lengths + row).to(tl.int32)
    segment_length = tl.minimum(tl.maximum(segment_length, 0), CAPACITY - row)
    if segment_length == 0:
        return

    q_head = v_head // (V_HEADS // Q_HEADS)
    keys = tl.arange(0, BLOCK_K)
    values = value_block * BLOCK_V + tl.arange(0, BLOCK_V)
    key_mask, value_mask = keys < K_DIM, values < V_DIM
    src = tl.load(src_indices + row).to(tl.int64)
    dst = tl.load(dst_indices + row).to(tl.int64)

    state_offset = (v_head * V_DIM + values[:, None]) * K_DIM + keys[None, :]
    matrix_mask = value_mask[:, None] & key_mask[None, :]
    source = _state_row(state,src,STATE_STRIDE,STATE_LAYERS,STATE_LAYER_STRIDE) + state_offset
    old_state = tl.load(source, mask=matrix_mask, other=0.0).to(tl.float32)
    new_state = old_state
    step = 0
    while step < segment_length:
        packed = packed_qkv + (row + step) * PACKED_DIM
        normalized = normalized_qk + (row + step) * PACKED_DIM
        q = tl.load(
            normalized + q_head * K_DIM + keys, mask=key_mask, other=0.0
        ).to(MATH_TYPE)
        k = tl.load(
            normalized + Q_HEADS * K_DIM + q_head * K_DIM + keys,
            mask=key_mask,
            other=0.0,
        ).to(MATH_TYPE)
        v = tl.load(
            packed + 2 * Q_HEADS * K_DIM + v_head * V_DIM + values,
            mask=value_mask,
            other=0.0,
        ).to(MATH_TYPE)
        # Q/K were normalized once per key head and rounded into this activation
        # packed activation before recurrence.  Do not recompute that reduction
        # independently in every value stripe.
        if PRECOMPUTED:
            beta_value = tl.load(coefficients + ((row + step) * V_HEADS + v_head) * 2)
            decay = tl.load(coefficients + ((row + step) * V_HEADS + v_head) * 2 + 1)
        else:
            beta_value, decay = _gdn_coefficients(
                tl.load(a + (row + step) * V_HEADS + v_head),
                tl.load(b + (row + step) * V_HEADS + v_head),
                tl.load(a_log + v_head), tl.load(dt_bias + v_head),
                ACT_TYPE, b.dtype.element_ty,
            )

        if NATIVE:
            new_state = _native_state_step(new_state, q, k, v, beta_value, decay, SCALE)
        else:
            new_state = _rn_mul(new_state, decay)
            memory_terms = _rn_mul(new_state, k[None, :])
            residual = _rn_mul(_rn_sub(v, tl.sum(memory_terms, axis=1)), beta_value)
            # The reference boundary is two rounded FP32 operations.  Make that
            # contraction policy explicit instead of permitting an LLVM/NVVM FMA.
            update = _rn_mul(residual[:, None], k[None, :])
            new_state = _rn_add(new_state, update)
        # A BF16 state is a mathematical step boundary, not merely compressed
        # backing storage.  The value visible to this step's output must equal
        # the value that the next step reloads.
        if STATE_BF16:
            new_state = new_state.to(tl.bfloat16).to(tl.float32)
        if NATIVE:
            lanes = tl.arange(0, 16)
            scaled_q = _low_mul(q, tl.full(q.shape, SCALE, q.dtype))
            q_column = tl.where(lanes[None, :] == 0, scaled_q[:, None], 0).to(q.dtype)
            result = tl.sum(tl.dot(new_state.to(q.dtype), q_column,
                                   out_dtype=tl.float32), 1)
        else:
            result_terms = _rn_mul(new_state, (q * SCALE)[None, :])
            result = tl.sum(result_terms, axis=1)
        tl.store(
            output + ((row + step) * V_HEADS + v_head) * V_DIM + values,
            result,
            mask=value_mask,
        )
        if CHECKPOINTS:
            checkpoint_dst = tl.load(dst_indices + row + step).to(tl.int64)
            checkpoint = (
                _state_row(state,checkpoint_dst,STATE_STRIDE,STATE_LAYERS,STATE_LAYER_STRIDE)
                + state_offset
            )
            tl.store(checkpoint, new_state, mask=matrix_mask)
        step += 1

    if not CHECKPOINTS:
        destination = _state_row(state,dst,STATE_STRIDE,STATE_LAYERS,STATE_LAYER_STRIDE) + state_offset
        tl.store(destination, new_state, mask=matrix_mask)


_TensorABI = tuple[tuple[int, ...], tuple[int, ...], torch.dtype, torch.device]


def _abi(tensor: Tensor) -> _TensorABI:
    if not isinstance(tensor, Tensor):
        raise TypeError("CUDA primitive operands must be tensors")
    return tuple(tensor.shape), tensor.stride(), tensor.dtype, tensor.device


def _require_cuda_contiguous(tensors: tuple[Tensor, ...], *, state: Tensor | None = None) -> None:
    if any(t.device.type != "cuda" or (t is not state and not t.is_contiguous()) for t in tensors):
        raise ValueError("ordinary operands must be contiguous CUDA tensors")
    if state is not None and (state.ndim < 2 or not state.shape[0]
            or not state[0].is_contiguous()
            or state.stride(0) < math.prod(state.shape[1:])):
        raise ValueError("state requires nonoverlapping slots with contiguous inner payloads")
    if len({t.device for t in tensors}) != 1:
        raise ValueError("all operands must share one CUDA device")


class CausalConv1DDecodePrimitive:
    """Fixed-ABI segmented BF16 convolution, history COW, and optional SiLU."""

    def __init__(
        self, x: Tensor, weight: Tensor, state: Tensor,
        src_indices: Tensor, dst_indices: Tensor, segment_lengths: Tensor,
        output: Tensor, *, silu: bool = True, checkpoint_destinations: bool = False,
    ) -> None:
        operands = (x, weight, state, src_indices, dst_indices, segment_lengths, output)
        _require_cuda_contiguous(operands, state=state)
        if x.ndim != 2 or weight.ndim != 2 or state.ndim not in (3,4):
            raise ValueError("expected x[C,D], weight[D,K], and state[S,D,K-1]")
        capacity, channels = x.shape
        width = weight.shape[1]
        layers = state.shape[1] if state.ndim == 4 else 1
        slots = state.shape[0] * layers
        if (
            min(capacity, channels, width, slots) <= 0
            or width < 2
            or weight.shape[0] != channels
            or state.shape[-2:] != (channels, width - 1)
            or src_indices.shape != (capacity,)
            or dst_indices.shape != (capacity,)
            or segment_lengths.shape != (capacity,)
            or output.shape != x.shape
            or x.dtype not in (torch.float16, torch.bfloat16)
            or any(t.dtype != x.dtype for t in (weight, state, output))
            or src_indices.dtype is not torch.int32
            or dst_indices.dtype is not torch.int32
            or segment_lengths.dtype is not torch.int32
        ):
            raise ValueError("operands do not match the FP16/BF16 causal-conv decode ABI")
        if type(checkpoint_destinations) is not bool:
            raise TypeError("checkpoint_destinations must be boolean")
        self._meta = capacity, channels, width, slots, x.device, silu, checkpoint_destinations
        self._abis = tuple(_abi(t) for t in operands)
        self._activation_type = activation_type(output.dtype, output.device)
        self._block_t = triton.next_power_of_2(width - 1)

    @property
    def launch_signature(self) -> str:
        capacity, channels, width, slots, _, silu, checkpoints = self._meta
        dtype = "bf16" if self._abis[0][2] is torch.bfloat16 else "fp16"
        return (
            f"c={capacity},d={channels},k={width},s={slots},"
            f"segments=i32,{dtype},silu={silu},checkpoints={checkpoints},act={self._activation_type}"
        )

    def launch(
        self, x: Tensor, weight: Tensor, state: Tensor,
        src_indices: Tensor, dst_indices: Tensor, segment_lengths: Tensor, output: Tensor,
    ) -> None:
        operands = (x, weight, state, src_indices, dst_indices, segment_lengths, output)
        if tuple(_abi(t) for t in operands) != self._abis:
            raise ValueError("causal-conv operands do not match the fixed launch ABI")
        capacity, channels, width, _, _, silu, checkpoints = self._meta
        block_c = 256
        self._compiled_kernel = _causal_conv_decode_kernel[(capacity, triton.cdiv(channels, block_c))](
            *operands,
            CAPACITY=capacity,
            CHANNELS=channels,
            WIDTH=width,
            BLOCK_C=block_c,
            BLOCK_T=self._block_t,
            SILU=silu,
            CHECKPOINTS=checkpoints,
            ACT_TYPE=self._activation_type,
            STATE_STRIDE=state.stride(0),
            STATE_LAYERS=state.shape[1] if state.ndim == 4 else 1,
            STATE_LAYER_STRIDE=state.stride(1) if state.ndim == 4 else 0,
            num_warps=4,
        )


class GatedDeltaDecodePrimitive:
    """Segmented GDN with bound operand precision and FP32/BF16 state storage."""

    def __init__(
        self, packed_qkv: Tensor, normalized_qk: Tensor, a: Tensor, b: Tensor,
        a_log: Tensor, dt_bias: Tensor,
        state: Tensor, src_indices: Tensor, dst_indices: Tensor,
        segment_lengths: Tensor, output: Tensor, *,
        q_heads: int, scale: float, epsilon: float = 1e-6,
        checkpoint_destinations: bool = False,
        row_activity: Tensor | None = None,
    ) -> None:
        operands = (
            packed_qkv, normalized_qk, a, b, a_log, dt_bias, state,
            src_indices, dst_indices, segment_lengths, output,
        )
        _require_cuda_contiguous(operands, state=state)
        if state.ndim not in (4,5) or packed_qkv.ndim != 2:
            raise ValueError("expected packed_qkv[C,QKV] and state[S,VH,VD,KD]")
        value_heads, value_dim, key_dim = state.shape[-3:]
        layers = state.shape[1] if state.ndim == 5 else 1
        slots = state.shape[0] * layers
        capacity = packed_qkv.shape[0]
        packed_dim = 2 * q_heads * key_dim + value_heads * value_dim
        if (
            type(q_heads) is not int
            or q_heads <= 0
            or value_heads % q_heads
            or not math.isfinite(scale)
            or not math.isfinite(epsilon)
            or scale <= 0.0
            or epsilon <= 0.0
            or min(capacity, slots, value_heads, value_dim, key_dim) <= 0
            or packed_qkv.shape != (capacity, packed_dim)
            or normalized_qk.shape != packed_qkv.shape
            or a.shape != (capacity, value_heads)
            or b.shape != a.shape
            or a_log.shape != (value_heads,)
            or dt_bias.shape != a_log.shape
            or src_indices.shape != (capacity,)
            or dst_indices.shape != (capacity,)
            or segment_lengths.shape != (capacity,)
            or output.shape != (capacity, value_heads, value_dim)
            or packed_qkv.dtype not in (torch.float16, torch.bfloat16)
            or any(
                t.dtype != packed_qkv.dtype
                for t in (packed_qkv, normalized_qk, a, b, dt_bias, output)
            )
            or a_log.dtype not in (packed_qkv.dtype, torch.float32)
            or state.dtype not in (torch.float32, torch.bfloat16)
            or src_indices.dtype is not torch.int32
            or dst_indices.dtype is not torch.int32
            or segment_lengths.dtype is not torch.int32
        ):
            raise ValueError("operands do not match the gated-delta decode ABI")
        if type(checkpoint_destinations) is not bool:
            raise TypeError("checkpoint_destinations must be boolean")
        if row_activity is not None and (
            row_activity.shape != (capacity,) or row_activity.dtype is not torch.int32
            or row_activity.device != packed_qkv.device or not row_activity.is_contiguous()
        ):
            raise ValueError("GDN logical row activity differs from the captured row ABI")
        self._row_activity = None
        self._meta = (
            capacity, q_heads, value_heads, key_dim, value_dim, slots,
            packed_qkv.device, scale, epsilon, checkpoint_destinations,
        )
        self._abis = tuple(_abi(t) for t in operands)
        self._activation_type = activation_type(output.dtype, output.device)
        self._attention_precision = attention_precision()
        native = self._attention_precision == "auto"
        self._block_k = max(16 if native else 1, triton.next_power_of_2(key_dim))
        parameters = gdn_parameters(
            q_heads=q_heads, value_heads=value_heads, key_dim=key_dim, value_dim=value_dim,
        )
        self._block_v = (
            min(max(16, triton.next_power_of_2(value_dim)), parameters.native_value_tile)
            if native else min(triton.next_power_of_2(value_dim), _VALUE_TILE)
        )
        self._warps, self._stages = (4 if native else 1), 3
        if native:
            self._select_geometry(packed_qkv, state)
        self._row_activity = row_activity

    def _select_geometry(self, packed_qkv: Tensor, state: Tensor) -> None:
        """Price independent V rows; K and recurrent step boundaries stay fixed.

        Native state-vector products use a 16-column MMA fragment. Both inline
        and reused coefficients are measured, on private states with identical
        resets. The potentially large production state arena is never copied.
        """
        capacity, qh, vh, kd, vd, _, device, _, _, checkpoints = self._meta
        if "native_value_tile" in parameter_overrides(
            "gdn", q_heads=qh, value_heads=vh, key_dim=kd, value_dim=vd,
        ):
            return
        context = current_linear_cost()
        resources = context.device if context else device_resources(device)
        candidates = matmul_geometries(
            resources, rows=vd, columns=(16,), reduction=kd,
            element_bytes=packed_qkv.element_size(), groups=capacity * vh,
            baseline=MatmulGeometry(self._block_v, 16, self._block_k, self._warps, self._stages),
            locked=frozenset({"block_n"}),
        )
        if len(candidates) == 1:
            return
        key = ("gdn", device, packed_qkv.dtype, state.dtype, self._abis[4][2],
               capacity, qh, vh, kd, vd,
               checkpoints, self._activation_type, candidates)
        cache = context.operator_choices if context else startup_operator_choices(device)

        def bind(index: int) -> None:
            point = candidates[index]
            self._block_v, self._warps, self._stages = point.block_m, point.warps, point.stages

        if key in cache:
            bind(cache[key])
            return
        fixture = None

        def prepare() -> None:
            if fixture is None:
                return
            initial, states = fixture[6:8]
            for value in states:
                value.copy_(initial)

        def launch() -> None:
            nonlocal fixture
            if fixture is None:
                # Restore/default selection never launches calibration. Keep
                # its three row-sized state images out of normal startup.
                # Actual calibration retains the identical operands and proof.
                generator = torch.Generator(device=device).manual_seed(0)
                qkv = torch.randn(packed_qkv.shape, dtype=packed_qkv.dtype, device=device,
                                  generator=generator).mul_(0.01)
                normalized = torch.zeros_like(qkv)
                a = torch.zeros((capacity, vh), dtype=qkv.dtype, device=device)
                b = torch.zeros_like(a)
                a_log = torch.zeros(vh, dtype=self._abis[4][2], device=device)
                dt = torch.zeros(vh, dtype=qkv.dtype, device=device)
                initial = torch.randn((capacity, vh, vd, kd), dtype=state.dtype,
                                      device=device, generator=generator).mul_(0.01)
                states = (torch.empty_like(initial), torch.empty_like(initial))
                indices = torch.arange(capacity, dtype=torch.int32, device=device)
                lengths = torch.ones_like(indices)
                outputs = tuple(torch.empty((capacity, vh, vd), dtype=qkv.dtype, device=device)
                                for _ in states)
                # a=b=a_log=dt=0 produces beta=decay=0.5 in both policies.
                coefficients = torch.full((capacity, vh, 2), 0.5,
                                          dtype=torch.float32, device=device)
                fixture = (qkv, normalized, a, b, a_log, dt, initial, states,
                           indices, lengths, outputs, coefficients)
                prepare()
            qkv, normalized, a, b, a_log, dt, _, states, indices, lengths, outputs, coefficients = fixture
            for value, output, coeff in zip(states, outputs, (None, coefficients), strict=True):
                self._execute((qkv, normalized, a, b, a_log, dt, value,
                               indices, indices, lengths, output), coeff)

        cache[key] = select_launch(
            range(len(candidates)), device=device, bind=bind, prepare=prepare, launch=launch,
            snapshot=lambda: (*fixture[10], *fixture[7]),
            cache_key=(key, _gated_delta_decode_kernel.cache_key),
        )


    @property
    def launch_signature(self) -> str:
        c, qh, vh, kd, vd, slots, _, _, _, checkpoints = self._meta
        update_rule = ("mma-product-fp32-rn-add" if self._attention_precision == "auto"
                       else "fp32-rn-separate")
        return (
            f"c={c},qh={qh},vh={vh},kd={kd},vd={vd},s={slots},"
            f"segments=i32,activation={self._abis[0][2]},"
            f"a_log={self._abis[4][2]},state={self._abis[6][2]},"
            f"checkpoints={checkpoints},act={self._activation_type},"
            f"attention={self._attention_precision},state_update="
            f"{update_rule},block_v={self._block_v},block_k={self._block_k},"
            f"warps={self._warps},stages={self._stages}"
        )

    def launch(
        self, packed_qkv: Tensor, normalized_qk: Tensor, a: Tensor, b: Tensor,
        a_log: Tensor, dt_bias: Tensor,
        state: Tensor, src_indices: Tensor, dst_indices: Tensor,
        segment_lengths: Tensor, output: Tensor, *, coefficients: Tensor | None = None,
    ) -> None:
        operands = (
            packed_qkv, normalized_qk, a, b, a_log, dt_bias, state,
            src_indices, dst_indices, segment_lengths, output,
        )
        if tuple(_abi(t) for t in operands) != self._abis:
            raise ValueError("gated-delta operands do not match the fixed launch ABI")
        (
            capacity, q_heads, value_heads, key_dim, value_dim, _, _, scale,
            epsilon, checkpoints,
        ) = self._meta
        if coefficients is not None and (
            coefficients.shape != (capacity, value_heads, 2)
            or coefficients.dtype is not torch.float32
            or coefficients.device != packed_qkv.device
            or not coefficients.is_contiguous()
            or any(
                coefficients.untyped_storage().data_ptr() == t.untyped_storage().data_ptr()
                for t in operands
            )
        ):
            raise ValueError("invalid precomputed GDN coefficient ABI")
        self._execute(operands, coefficients)

    def _execute(self, operands: tuple[Tensor, ...], coefficients: Tensor | None) -> None:
        (packed_qkv, normalized_qk, a, b, a_log, dt_bias, state,
         src_indices, dst_indices, segment_lengths, output) = operands
        (capacity, q_heads, value_heads, key_dim, value_dim, _, _, scale,
         epsilon, checkpoints) = self._meta
        packed_dim = 2 * q_heads * key_dim + value_heads * value_dim
        self._compiled_normalize = _gated_delta_qk_normalize_kernel[(capacity * q_heads,)](
            packed_qkv,
            normalized_qk,
            EPSILON=epsilon,
            Q_HEADS=q_heads,
            K_DIM=key_dim,
            PACKED_DIM=packed_dim,
            BLOCK_K=self._block_k,
            num_warps=1,
            num_stages=1,
        )
        self._compiled_recurrence = _gated_delta_decode_kernel[(triton.cdiv(value_dim, self._block_v), capacity * value_heads)](
            *operands,
            CAPACITY=capacity,
            SCALE=scale,
            Q_HEADS=q_heads,
            V_HEADS=value_heads,
            K_DIM=key_dim,
            V_DIM=value_dim,
            PACKED_DIM=packed_dim,
            BLOCK_K=self._block_k,
            BLOCK_V=self._block_v,
            CHECKPOINTS=checkpoints,
            ACT_TYPE=self._activation_type,
            coefficients=coefficients if coefficients is not None else a,
            PRECOMPUTED=coefficients is not None,
            STATE_BF16=state.dtype is torch.bfloat16,
            STATE_STRIDE=state.stride(0),
            STATE_LAYERS=state.shape[1] if state.ndim == 5 else 1,
            STATE_LAYER_STRIDE=state.stride(1) if state.ndim == 5 else 0,
            NATIVE=self._attention_precision == "auto",
            num_warps=self._warps,
            num_stages=self._stages,
        )
