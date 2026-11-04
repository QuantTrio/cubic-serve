"""BF16 state intervals on frozen actual inputs; explicit FP32 libdevice bounds."""

from functools import partial

import torch

from .bounds import R, check, gated_norm_enclosure, product


def fp32_gdn_output_range(state, normalized_query, *, dtype):
    """Enclose only the FP32 sum tree after an independently verified state.

    Both producers must already be witnessed. Scaling and each product use
    FP32 RN, matching the explicit FP32 attention contract; only summation
    order varies. The final interval includes the declared output storage cast.
    """
    state = state.detach().cpu().float()
    query = normalized_query.detach().cpu().float()
    if state.ndim != 4 or query.ndim != 3:
        raise ValueError("expected state[B,H,V,K] and normalized query[B,QH,K]")
    if (state.shape[0] != query.shape[0] or state.shape[-1] != query.shape[-1]
            or state.shape[1] % query.shape[1]):
        raise ValueError("state/query head geometry differs")
    width = state.shape[-1]
    scaled = (query * (width ** -0.5)).repeat_interleave(
        state.shape[1] // query.shape[1], dim=1
    )
    terms = (state * scaled.unsqueeze(-2)).double()
    center = terms.sum(-1)
    gamma = width * 2**-24 / (1 - width * 2**-24)
    error = terms.abs().sum(-1) * gamma + width * torch.finfo(torch.float32).tiny
    lower = torch.nextafter(center - error, torch.full_like(center, -float("inf")))
    upper = torch.nextafter(center + error, torch.full_like(center, float("inf")))
    return lower.float().to(dtype).double(), upper.float().to(dtype).double()


def sigmoid_range(lo, hi, *, dtype=torch.bfloat16):
    round_low = partial(R, dtype=dtype)
    alo = torch.where((lo <= 0) & (hi >= 0), 0, torch.minimum(lo.abs(), hi.abs()))
    ahi = torch.maximum(lo.abs(), hi.abs())
    arglo = round_low(-round_low(ahi) * round_low(torch.tensor(1.4426950408889634)))
    arghi = round_low(-round_low(alo) * round_low(torch.tensor(1.4426950408889634)))
    elo = (torch.exp2(arglo) * (1 - torch.finfo(dtype).eps) - torch.finfo(dtype).tiny).clamp_min(0)
    ehi = torch.exp2(arghi) * (1 + torch.finfo(dtype).eps) + torch.finfo(dtype).tiny
    elo, ehi = (round_low(elo), round_low(ehi))
    dlo, dhi = (round_low(1 + elo), round_low(1 + ehi))
    low = torch.where(lo >= 0, 1 / dhi, elo / dhi)
    high = torch.where(hi < 0, ehi / dlo, 1 / dlo)
    low = torch.nextafter(low, torch.full_like(low, -float("inf")))
    high = torch.nextafter(high, torch.full_like(high, float("inf")))
    return (round_low(low.float()), round_low(high.float()))


def library_interval(fn, lo, hi, ulps):
    lower = fn(lo).float()
    upper = fn(hi).float()
    for _ in range(ulps):
        lower = torch.nextafter(lower, torch.full_like(lower, -float("inf")))
        upper = torch.nextafter(upper, torch.full_like(upper, float("inf")))
    return (lower.double(), upper.double())


def fp32_conv_silu_range(x, weight, state, *, dtype):
    """Independent FP32 convolution/SiLU enclosure on frozen stored inputs.

    The convolution stays FP32 until SiLU; there is no low-precision
    preactivation boundary. Include any FP32 sum tree, libdevice exp's declared
    two-ULP error, RN denominator/division, then the final storage rounding.
    Nonmonotone negative SiLU is enclosed by all numerator/denominator corners.
    """
    terms = torch.cat((state.double(), x.double().unsqueeze(-1)), dim=-1) * weight.double()
    width = terms.shape[-1]
    gamma = width * 2**-24 / (1 - width * 2**-24)
    error = terms.abs().sum(-1) * gamma + width * torch.finfo(torch.float32).tiny
    center = terms.sum(-1)
    lower = (center - error).float().double()
    upper = (center + error).float().double()
    elo, ehi = library_interval(torch.exp, -upper, -lower, 2)
    dlo = (1 + elo.clamp_min(0)).float().double()
    dhi = (1 + ehi.clamp_min(0)).float().double()
    corners = torch.stack((lower / dlo, lower / dhi, upper / dlo, upper / dhi))
    lo = torch.nextafter(corners.amin(0), torch.full_like(lower, -float("inf")))
    hi = torch.nextafter(corners.amax(0), torch.full_like(upper, float("inf")))
    return lo.float().to(dtype).double(), hi.float().to(dtype).double()


def decay_range(a, alog, bias, *, dtype=torch.bfloat16):
    round_low = partial(R, dtype=dtype)
    alpha = (a.double() + bias.double()).float().double()
    elo, ehi = library_interval(torch.exp, alpha, alpha, 2)
    elo = elo.clamp_min(0)
    slo, shi = library_interval(torch.log1p, elo, ehi, 1)
    slo = slo.clamp_min(0)
    slo = torch.where(alpha <= 20, slo, alpha)
    shi = torch.where(alpha <= 20, shi, alpha)
    alo, ahi = library_interval(torch.exp, alog.double(), alog.double(), 2)
    alo = alo.clamp_min(0)
    lo = (-ahi * shi).float().double()
    hi = (-alo * slo).float().double()
    lo, hi = library_interval(torch.exp, lo, hi, 2)
    return (round_low(lo.clamp_min(0)), round_low(hi.clamp_min(0)))


def fp32_gdn_state_range(state, key, value, a, b, alog, bias, *, activation_dtype=torch.float32):
    """Full recurrence enclosure with FP32 RN operations and stored state.

    Inputs are frozen, producer-witnessed operands. The only allowances are
    libdevice exp (2 ULP), log1p (1 ULP), and the complete-K FP32 sum tree.
    Every intervening multiplication, subtraction and addition rounds to FP32;
    the final cast is the mathematical state boundary, including for output.
    """
    dtype, beta_dtype = state.dtype, b.dtype
    state, key, value, a, b, alog, bias = (
        x.detach().cpu().double() for x in (state, key, value, a, b, alog, bias)
    )
    dlo, dhi = decay_range(a, alog, bias, dtype=torch.float32)
    if activation_dtype is torch.float32:
        elo, ehi = library_interval(torch.exp, -b, -b, 2)
        blo = (1 / (1 + ehi.clamp_min(0)).float().double()).float()
        bhi = (1 / (1 + elo.clamp_min(0)).float().double()).float()
    else:
        blo, bhi = sigmoid_range(b, b, dtype=activation_dtype)
    # Beta's storage precision is independent of recurrent-state precision.
    blo, bhi = blo.to(beta_dtype).double(), bhi.to(beta_dtype).double()
    lo, hi = product(state, state, dlo[..., None, None], dhi[..., None, None])
    lo, hi = lo.float().double(), hi.float().double()
    tlo, thi = product(lo, hi, key[..., None, :], key[..., None, :])
    tlo, thi = tlo.float().double(), thi.float().double()
    width = key.shape[-1]
    gamma = width * 2**-24 / (1 - width * 2**-24)
    error = torch.maximum(tlo.abs(), thi.abs()).sum(-1) * gamma
    error += width * torch.finfo(torch.float32).tiny
    mlo, mhi = (tlo.sum(-1) - error).float().double(), (thi.sum(-1) + error).float().double()
    rlo, rhi = (value - mhi).float().double(), (value - mlo).float().double()
    rlo, rhi = product(rlo, rhi, blo.double()[..., None], bhi.double()[..., None])
    rlo, rhi = rlo.float().double(), rhi.float().double()
    ulo, uhi = product(rlo[..., None], rhi[..., None], key[..., None, :], key[..., None, :])
    ulo, uhi = ulo.float().double(), uhi.float().double()
    return (lo + ulo).float().to(dtype).double(), (hi + uhi).float().to(dtype).double()


def conv_silu_range(x, weight, state, *, dtype, activation_dtype):
    """Resolve the declared activation arithmetic after the FP32 convolution."""
    if activation_dtype is torch.float32:
        return fp32_conv_silu_range(x, weight, state, dtype=dtype)
    if activation_dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("unsupported convolution activation arithmetic")
    terms = torch.cat((state.double(), x.double().unsqueeze(-1)), dim=-1) * weight.double()
    width = terms.shape[-1]
    gamma = width * 2**-24 / (1 - width * 2**-24)
    center = terms.sum(-1)
    error = terms.abs().sum(-1) * gamma + width * torch.finfo(torch.float32).tiny
    lower, upper = (center - error).float().double(), (center + error).float().double()
    blo, bhi = sigmoid_range(lower, upper, dtype=activation_dtype)
    vlo, vhi = R(lower, activation_dtype), R(upper, activation_dtype)
    lo, hi = product(vlo, vhi, blo, bhi)
    lo, hi = R(lo.float(), activation_dtype), R(hi.float(), activation_dtype)
    if activation_dtype is torch.float16:
        # The independent activation contract retains the finite wide tails.
        crossing = (lower < -65504) | (upper > 65504)
        safe_lower, safe_upper = lower.clamp(-65504, 65504), upper.clamp(-65504, 65504)
        sblo, sbhi = sigmoid_range(safe_lower, safe_upper, dtype=activation_dtype)
        slo, shi = product(R(safe_lower, activation_dtype), R(safe_upper, activation_dtype), sblo, sbhi)
        slo, shi = R(slo.float(), activation_dtype), R(shi.float(), activation_dtype)
        tail_lo = torch.where(lower > 65504, lower, torch.zeros_like(lower))
        tail_hi = torch.where(upper > 65504, upper, torch.zeros_like(upper))
        lo = torch.where(crossing, torch.minimum(slo, tail_lo), lo)
        hi = torch.where(crossing, torch.maximum(shi, tail_hi), hi)
    return R(lo, dtype), R(hi, dtype)


def native_gdn_state_range(state, key, value, a, b, alog, bias, *, activation_dtype):
    """Low operand products, FP32 reductions/addition, then durable state RNE.

    Every operand comes from the same producer-witnessed forward. Gate arithmetic
    and gate storage are distinct from the recurrence operand dtype.
    """
    state_dtype, operand_dtype, beta_dtype = state.dtype, key.dtype, b.dtype
    state, key, value, a, b, alog, bias = (
        x.detach().cpu().double() for x in (state, key, value, a, b, alog, bias)
    )
    if activation_dtype is torch.float32:
        elo, ehi = library_interval(torch.exp, -b, -b, 2)
        blo = (1 / (1 + ehi.clamp_min(0)).float().double()).float().double()
        bhi = (1 / (1 + elo.clamp_min(0)).float().double()).float().double()
    else:
        blo, bhi = sigmoid_range(b, b, dtype=activation_dtype)
    blo, bhi = R(R(blo, beta_dtype), operand_dtype), R(R(bhi, beta_dtype), operand_dtype)
    dlo, dhi = decay_range(a, alog, bias, dtype=operand_dtype)
    prior = R(state, operand_dtype)
    lo, hi = product(prior, prior, dlo[..., None, None], dhi[..., None, None])
    lo, hi = R(lo.float(), operand_dtype), R(hi.float(), operand_dtype)
    tlo, thi = product(lo, hi, key[..., None, :], key[..., None, :])
    width = key.shape[-1]
    gamma = width * 2**-24 / (1 - width * 2**-24)
    error = torch.maximum(tlo.abs(), thi.abs()).sum(-1) * gamma
    error += width * torch.finfo(torch.float32).tiny
    mlo, mhi = (tlo.sum(-1) - error).float().double(), (thi.sum(-1) + error).float().double()
    rlo = R(value - R(mhi, operand_dtype), operand_dtype)
    rhi = R(value - R(mlo, operand_dtype), operand_dtype)
    rlo, rhi = product(rlo, rhi, blo[..., None], bhi[..., None])
    rlo, rhi = R(rlo.float(), operand_dtype), R(rhi.float(), operand_dtype)
    ulo, uhi = product(rlo[..., None], rhi[..., None], key[..., None, :], key[..., None, :])
    return R((lo + ulo.float().double()).float(), state_dtype), R((hi + uhi.float().double()).float(), state_dtype)


def native_gdn_output_range(state, normalized_query, *, dtype):
    """Read the witnessed rounded state using low operands and FP32 sums."""
    operand_dtype = normalized_query.dtype
    state = R(state.detach().cpu().double(), operand_dtype)
    query = normalized_query.detach().cpu().double()
    width = state.shape[-1]
    scaled = R(query * R(torch.tensor(width**-0.5), operand_dtype), operand_dtype)
    scaled = scaled.repeat_interleave(state.shape[1] // query.shape[1], dim=1)
    terms = state * scaled.unsqueeze(-2)
    center = terms.sum(-1)
    gamma = width * 2**-24 / (1 - width * 2**-24)
    error = terms.abs().sum(-1) * gamma + width * torch.finfo(torch.float32).tiny
    lower = torch.nextafter(center - error, torch.full_like(center, -float("inf")))
    upper = torch.nextafter(center + error, torch.full_like(center, float("inf")))
    return R(lower.float(), dtype), R(upper.float(), dtype)


def normalized_range(x, epsilon, *, dtype=torch.bfloat16):
    round_low = partial(R, dtype=dtype)
    width = x.shape[-1]
    sq = (x * x).sum(-1, keepdim=True)
    g = width * 2 ** (-24) / (1 - width * 2 ** (-24))
    lo = (sq * (1 - g) + float(torch.tensor(epsilon))).float().double()
    hi = (sq * (1 + g) + float(torch.tensor(epsilon))).float().double()
    lo, hi = product(
        x, x, torch.rsqrt(hi) * (1 - 3 * 2 ** (-23)), torch.rsqrt(lo) * (1 + 3 * 2 ** (-23))
    )
    return (round_low(lo.float()), round_low(hi.float()))


def evaluate(s, *, norm_epsilon=1e-6):
    dtype = s["conv"][0].dtype
    round_low = partial(R, dtype=dtype)
    report = {
        "scope": (
            "nonzero frozen BF16 state, per-stage actual-input intervals; "
            "conditional on declared libdevice accuracy, not a whole-model seal"
        ),
        "checks": {},
    }
    x, w, state, y, new = s["conv"]
    history = state.double() * w[:, :-1].double()
    terms = torch.cat((history, (x.double() * w[:, -1].double()).unsqueeze(-1)), dim=-1)
    width = terms.shape[-1]
    gamma = width * 2 ** (-24) / (1 - width * 2 ** (-24))
    center = terms.sum(-1)
    error = terms.abs().sum(-1) * gamma + torch.finfo(torch.float32).tiny
    lo, hi = ((center - error).float().double(), (center + error).float().double())
    blo, bhi = sigmoid_range(lo, hi, dtype=dtype)
    lo, hi = product(round_low(lo), round_low(hi), blo, bhi)
    report["checks"]["conv_silu"] = check(y, round_low(lo), round_low(hi))
    x, a, b, alog, bias, state, y, new, packed = s["gdn"]
    vh, vd, kd = state.shape[-3:]
    qh = (x.shape[-1] - vh * vd) // (2 * kd)
    width = qh * kd
    q, k, v = x.split((width, width, vh * vd), -1)
    q = q.reshape(-1, qh, kd).double()
    k = k.reshape_as(q).double()
    v = v.reshape(-1, vh, vd).double()
    aq, ak, _ = packed.split((width, width, vh * vd), -1)
    aq = aq.reshape_as(q).double()
    ak = ak.reshape_as(k).double()
    report["checks"]["normalized_q_fp32_bounds"] = check(
        aq, *normalized_range(q, 1e-06, dtype=dtype)
    )
    report["checks"]["normalized_k_fp32_bounds"] = check(
        ak, *normalized_range(k, 1e-06, dtype=dtype)
    )
    aq = aq.repeat_interleave(vh // qh, dim=1)
    ak = ak.repeat_interleave(vh // qh, dim=1)
    dlo, dhi = decay_range(a, alog, bias, dtype=dtype)
    blo, bhi = sigmoid_range(b.double(), b.double(), dtype=dtype)
    lo, hi = product(
        round_low(state.double()),
        round_low(state.double()),
        dlo[..., None, None],
        dhi[..., None, None],
    )
    lo, hi = (round_low(lo), round_low(hi))
    tlo, thi = product(lo, hi, ak[..., None, :], ak[..., None, :])
    gamma = kd * 2 ** (-24) / (1 - kd * 2 ** (-24))
    error = torch.maximum(tlo.abs(), thi.abs()).sum(-1) * gamma + torch.finfo(torch.float32).tiny
    mlo = (tlo.sum(-1) - error).float().double()
    mhi = (thi.sum(-1) + error).float().double()
    rlo, rhi = (round_low(v - round_low(mhi)), round_low(v - round_low(mlo)))
    rlo, rhi = product(rlo, rhi, blo[..., None], bhi[..., None])
    rlo, rhi = (round_low(rlo), round_low(rhi))
    ulo, uhi = product(rlo[..., None], rhi[..., None], ak[..., None, :], ak[..., None, :])
    report["checks"]["warm_state_interval"] = check(
        new, R((lo + ulo).float(), new.dtype), R((hi + uhi).float(), new.dtype)
    )
    scaled = round_low(aq * round_low(torch.tensor(kd ** (-0.5))))
    terms = round_low(new.double()) * scaled[..., None, :]
    center = terms.sum(-1)
    error = terms.abs().sum(-1) * gamma + torch.finfo(torch.float32).tiny
    report["checks"]["output_fp32_reduction"] = check(
        y, round_low(center - error), round_low(center + error)
    )
    report["checks"]["gated_norm_declared_math_bounds"] = gated_norm_enclosure(
        *s["norm"],
        epsilon=norm_epsilon,
    )
    report["decay_bf16_interval_heads"] = int((dlo != dhi).sum())
    report["state_nonzero"] = int(state.count_nonzero())
    return report
