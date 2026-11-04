"""Segment-aware exact INT64 lexical hashes with caller-owned history slots."""

from __future__ import annotations

import torch
import triton  # type: ignore[import-untyped]
import triton.language as tl  # type: ignore[import-untyped]
from torch import Tensor

from cubic_serve.quantization.common.binding import PackedOperatorBinding, _PackedLaunch
from cubic_serve.runtime.cache.quantization import slot_storage_contiguous, storage_regions_overlap


@triton.jit  # type: ignore[untyped-decorator]
def _ngram_hash(  # type: ignore[no-untyped-def]
    TOKENS,
    STATES,
    SOURCES,
    DESTINATIONS,
    SEGMENTS,
    ACTIVE,
    MULTIPLIERS,
    VOCABS,
    OFFSETS,
    OUTPUT,
    ROWS: tl.constexpr,
    HISTORY: tl.constexpr,
    HEADS: tl.constexpr,
    EOS: tl.constexpr,
    HB: tl.constexpr,
    VB: tl.constexpr,
    STATE_STRIDE: tl.constexpr,
):
    start = tl.program_id(0)
    length = tl.load(SEGMENTS + start)
    owned = tl.load(ACTIVE + start) != 0
    if owned and length > 0:
        source = tl.load(SOURCES + start).to(tl.int64)
        h = tl.arange(0, HB)
        history = tl.load(STATES + source * STATE_STRIDE + h, (source >= 0) & (h < HISTORY), EOS)
        heads = tl.arange(0, VB)
        first_multiplier = tl.load(MULTIPLIERS)
        for step in range(tl.minimum(length, ROWS - start)):
            row = start + step
            # Segment continuation is a real token even when SEGMENTS[row]=0.
            active = tl.load(ACTIVE + row) != 0
            if active:
                token = tl.load(TOKENS + row)
                hashed = token * first_multiplier
                reset = tl.full((), False, tl.int1)
                for lag in tl.static_range(1, HISTORY + 1):
                    previous = tl.sum(tl.where(h == HISTORY - lag, history, 0), 0)
                    reset = reset | (previous == EOS)
                    previous = tl.where(reset, EOS, previous)
                    hashed = hashed ^ (previous * tl.load(MULTIPLIERS + lag))
                    base = (lag - 1) * HEADS
                    vocab = tl.load(VOCABS + base + heads, heads < HEADS, 1)
                    offset = tl.load(OFFSETS + base + heads, heads < HEADS, 0)
                    remainder = hashed % vocab
                    positive = tl.where(remainder < 0, remainder + vocab, remainder)
                    tl.store(
                        OUTPUT + row * (HISTORY * HEADS) + base + heads,
                        positive + offset,
                        heads < HEADS,
                    )
                history = tl.gather(history, tl.minimum(h + 1, HB - 1), 0)
                history = tl.where(h == HISTORY - 1, token, history)
                destination = tl.load(DESTINATIONS + row).to(tl.int64)
                tl.store(
                    STATES + destination * STATE_STRIDE + h, history, (destination >= 0) & (h < HISTORY)
                )


def bind_ngram_hash(
    tokens: Tensor,
    states: Tensor,
    sources: Tensor,
    destinations: Tensor,
    segment_lengths: Tensor,
    row_activity: Tensor,
    multipliers: Tensor,
    vocab_sizes: Tensor,
    offsets: Tensor,
    output: Tensor,
    *,
    eos_token_id: int,
) -> PackedOperatorBinding:
    """Bind hashes and per-token state checkpoints without host token inspection.

    A negative source initializes history to EOS. A negative destination suppresses
    the corresponding checkpoint. Live and padding ownership is explicit; the
    executor must validate slot bounds and disjoint writable owners before replay.
    Shared prefix sources are read-only unless explicitly owned as destinations.
    """
    tensors = (
        tokens,
        states,
        sources,
        destinations,
        segment_lengths,
        row_activity,
        multipliers,
        vocab_sizes,
        offsets,
        output,
    )
    if tokens.device.type != "cuda" or any(
        t.device != tokens.device or (t is not states and not t.is_contiguous()) for t in tensors
    ):
        raise ValueError("ngram operands require contiguous same-device CUDA tensors")
    if not slot_storage_contiguous(states):
        raise ValueError("ngram history requires dense, nonoverlapping slot payloads")
    if (
        tokens.ndim != 1
        or tokens.numel() <= 0
        or states.ndim != 2
        or states.shape[0] <= 0
        or states.shape[1] < 1
    ):
        raise ValueError("ngram tokens and history must have positive declared geometry")
    rows, history = tokens.numel(), states.shape[1]
    if any(
        t.dtype != torch.int64 for t in (tokens, states, multipliers, vocab_sizes, offsets, output)
    ):
        raise ValueError("ngram values and hash metadata must remain INT64")
    if any(
        t.dtype != torch.int32 or t.shape != (rows,)
        for t in (sources, destinations, segment_lengths, row_activity)
    ):
        raise ValueError("ngram ownership and segmentation require one INT32 entry per row")
    if (
        multipliers.shape != (history + 1,)
        or vocab_sizes.ndim != 1
        or vocab_sizes.numel() <= 0
        or vocab_sizes.numel() % history
        or offsets.shape != vocab_sizes.shape
        or output.shape != (rows, vocab_sizes.numel())
        or type(eos_token_id) is not int
        or not 0 <= eos_token_id < 2**63
    ):
        raise ValueError("ngram order, head metadata or output geometry differs")
    heads = vocab_sizes.numel() // history
    # Immutable checkpoint metadata is validated at binding, never at replay.
    sizes, starts = vocab_sizes.cpu().tolist(), offsets.cpu().tolist()
    if any(v <= 0 or o < 0 or o > (2**63 - v) for v, o in zip(sizes, starts, strict=True)):
        raise ValueError("ngram vocabulary ranges must be positive non-overflowing INT64")
    for writable in (states, output):
        if any(
            writable is not t
            and storage_regions_overlap(writable, t)
            for t in tensors
        ):
            raise ValueError("ngram state/output may not alias other operands")
    call = _PackedLaunch.compile(
        _ngram_hash,
        tensors,
        (
            rows,
            history,
            heads,
            eos_token_id,
            triton.next_power_of_2(history),
            triton.next_power_of_2(heads),
            states.stride(0),
        ),
        (rows,),
        {"num_warps": 1, "enable_fp_fusion": False},
    )
    return PackedOperatorBinding(output, (call,))
