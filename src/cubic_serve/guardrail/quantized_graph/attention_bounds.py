"""Independent CPU intervals for low-QKV/F32-softmax paged decode.

Named allowance: FP32 dot reductions, libdevice expf <=2 ULP, explicit
FP32 RN online operations, and the declared low or FP32 P boundary for PV.
No candidate partials or statistics are used as reference inputs.
"""

import math

import torch

from .bounds import check, product
from .state_bounds import library_interval


def F(x):
    return x.float().double()


def exp_range(lo, hi):
    lo, hi = (lo.clamp_max(0), hi.clamp_max(0))
    a, b = library_interval(torch.exp, lo, hi, 2)
    exact_zero = (lo == 0) & (hi == 0)
    return (torch.where(exact_zero, 1, a.clamp_min(0)), torch.where(exact_zero, 1, b))


def rounded_sum(lo, hi, axis, terms):
    gamma32 = terms * 2 ** (-24) / (1 - terms * 2 ** (-24))
    gamma64 = terms * 2 ** (-53) / (1 - terms * 2 ** (-53))
    gamma = (gamma32 + gamma64) / (1 - gamma64)
    radius = torch.maximum(lo.abs(), hi.abs()).sum(axis) * gamma
    radius += torch.finfo(torch.float32).tiny
    return (F(lo.sum(axis) - radius), F(hi.sum(axis) + radius))


def merge(left, right):
    if left is None:
        return right
    ml, mh, al, ah, dl, dh = left
    nl, nh, bl, bh, el, eh = right
    xl, xh = (torch.maximum(ml, nl), torch.maximum(mh, nh))
    sl, sh = exp_range(F(ml - xh), F(mh - xl))
    tl, th = exp_range(F(nl - xh), F(nh - xl))
    old = ml >= nh
    new = nl >= mh
    sl, sh = (torch.where(old, 1, sl), torch.where(old, 1, sh))
    tl, th = (torch.where(new, 1, tl), torch.where(new, 1, th))
    vl, vh = product(al, ah, sl[:, None], sh[:, None])
    wl, wh = product(bl, bh, tl[:, None], th[:, None])
    return (
        xl,
        xh,
        F(F(vl) + F(wl)),
        F(F(vh) + F(wh)),
        F(F(dl * sl) + F(el * tl)),
        F(F(dh * sh) + F(eh * th)),
    )


def _inverse_value_interval(lo, hi):
    """Propagate the declared FP32 butterfly, before native output rounding."""
    width = lo.shape[-1]
    block = width & -width
    low = lo.reshape(*lo.shape[:-1], -1, block).clone()
    high = hi.reshape_as(low).clone()
    step = 1
    while step < block:
        lower = low.reshape(*low.shape[:-1], -1, 2, step)
        upper = high.reshape_as(lower)
        al, bl = lower[..., 0, :].clone(), lower[..., 1, :].clone()
        ah, bh = upper[..., 0, :].clone(), upper[..., 1, :].clone()
        lower[..., 0, :], upper[..., 0, :] = F(al + bl), F(ah + bh)
        lower[..., 1, :], upper[..., 1, :] = F(al - bh), F(ah - bl)
        step *= 2
    normalization = float(torch.tensor(1 / math.sqrt(block), dtype=torch.float32))
    return F(low.reshape_as(lo) * normalization), F(high.reshape_as(hi) * normalization)


def enclose(q, k, v, out, *, scale, splits, block_n=16, probability_dtype=None,
            value_rotation=False):
    if q.shape[0] != 1 or k.shape[0] != 1 or v.shape != k.shape:
        raise ValueError("one request, matching paged K/V required")
    dtype = q.dtype
    probability_dtype = dtype if probability_dtype is None else probability_dtype
    if probability_dtype not in (dtype, torch.float32):
        raise ValueError("PV probabilities must use the declared QKV or FP32 dtype")
    if (
        dtype not in (torch.bfloat16, torch.float16)
        or k.dtype != dtype
        or v.dtype != dtype
        or (out.dtype != dtype)
    ):
        raise ValueError("one explicit low-precision QKV/output dtype required")
    q, k, v = (q[0].cpu().double(), k[0].cpu().double(), v[0].cpu().double())
    if not all(torch.isfinite(t).all() for t in (q, k, v)):
        raise ValueError("nonfinite input")
    heads, dim = q.shape
    n = k.shape[0]
    if n <= 0 or splits <= 0 or block_n <= 0 or heads % k.shape[1]:
        raise ValueError("invalid GQA/split geometry")
    mapping = torch.arange(heads) // (heads // k.shape[1])
    keys = k[:, mapping].permute(1, 0, 2)
    values = v[:, mapping].permute(1, 0, 2)
    terms = q[:, None, :] * keys
    slo, shi = rounded_sum(terms, terms, -1, dim)
    scale = float(torch.tensor(scale, dtype=torch.float32))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("positive scale required")
    slo, shi = (F(slo * scale), F(shi * scale))
    merged = None
    span = math.ceil(n / splits)
    blocks = 0
    for start in range(0, n, span):
        state = None
        for pos in range(start, min(start + span, n), block_n):
            end = min(pos + block_n, start + span, n)
            ml, mh = (slo[:, pos:end].amax(-1), shi[:, pos:end].amax(-1))
            if state is not None:
                ml, mh = (torch.maximum(ml, state[0]), torch.maximum(mh, state[1]))
            pl, ph = exp_range(F(slo[:, pos:end] - mh[:, None]), F(shi[:, pos:end] - ml[:, None]))
            dl, dh = rounded_sum(pl, ph, -1, block_n)
            dl = dl.clamp_min(0)
            rpl, rph = (pl.to(probability_dtype).double(), ph.to(probability_dtype).double())
            tl, th = product(rpl[..., None], rph[..., None], values[:, pos:end], values[:, pos:end])
            # Low-P products are exact in FP32. The scalar FP32-P path also
            # rounds each product before the independently bounded reduction.
            tl, th = F(tl), F(th)
            al, ah = rounded_sum(tl, th, 1, block_n)
            if state is not None:
                sl, sh = exp_range(F(state[0] - mh), F(state[1] - ml))
                wins = state[0] >= shi[:, pos:end].amax(-1)
                sl, sh = (torch.where(wins, 1, sl), torch.where(wins, 1, sh))
                vl, vh = product(state[2], state[3], sl[:, None], sh[:, None])
                al, ah = (F(F(vl) + al), F(F(vh) + ah))
                dl, dh = (F(F(state[4] * sl) + dl), F(F(state[5] * sh) + dh))
            state = (ml, mh, al, ah, dl, dh)
            blocks += 1
        merged = merge(merged, state)
    _, _, al, ah, dl, dh = merged
    if not (dl > 0).all():
        raise AssertionError("nonpositive denominator enclosure")
    lo, hi = product(al, ah, 1 / dh[:, None], 1 / dl[:, None])
    lo = torch.nextafter(lo, torch.full_like(lo, -float("inf")))
    hi = torch.nextafter(hi, torch.full_like(hi, float("inf")))
    if value_rotation:
        lo, hi = _inverse_value_interval(F(lo), F(hi))
    lo, hi = (F(lo).to(dtype).double(), F(hi).to(dtype).double())
    if not (torch.isfinite(lo).all() and torch.isfinite(hi).all() and (lo <= hi).all()):
        raise AssertionError("invalid output enclosure")
    report = check(out[0].cpu(), lo, hi)
    report.update(
        keys=n,
        kv_splits=splits,
        block_n=block_n,
        blocks=blocks,
        max_output_interval_width=float((hi - lo).max()),
        contract=(
            f"low-QKV/F32-acc; F32-expf-softmax; P={probability_dtype} before PV; "
            "explicit FP32 RN online and split merge"
            + ("; paired V inverse in FP32 before native output RN" if value_rotation else "")
        ),
    )
    return report
