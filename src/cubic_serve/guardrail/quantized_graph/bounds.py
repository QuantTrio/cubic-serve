"""Cold-state actual-input enclosures; never a full-model or warm-state seal."""

from functools import partial

import torch


def R(x, dtype=torch.bfloat16):
    return x.to(dtype).double()


def product(a, b, c, d):
    """Enclose all four corners without retaining a four-corner stack.

    The accumulators own their storage; input snapshots are never modified.
    CPU FP32/FP64 extrema preserve the original storage bits, including signed
    zero. Keep the original reduction for other arithmetic, promotion and
    autograd paths, whose NaN or signed-zero selection can differ.
    """
    if (a.device.type != "cpu" or a.dtype not in (torch.float32, torch.float64)
            or any(t.dtype != a.dtype or t.requires_grad for t in (a, b, c, d))):
        values = torch.stack([a * c, a * d, b * c, b * d])
        return values.amin(0), values.amax(0)
    lower = a * c
    upper = lower.clone()
    for left, right in ((a, d), (b, c), (b, d)):
        corner = left * right
        torch.minimum(lower, corner, out=lower)
        torch.maximum(upper, corner, out=upper)
    return lower, upper


def route_softmax_bounds(selected, dtype):
    """FP32 exp/reduction/division enclosure at the stored route boundary.

    CUDA 13 expf has a 2-ulp error against correctly rounded FP32 (up to
    2.5 ulp against real arithmetic). Three relative FP32 epsilons enclose
    that error, including binade boundaries. The positive sum uses gamma
    for at most K-1 additions; division allows two FP32 ulps. These are
    arithmetic bounds, not an FP16/BF16 output tolerance.
    """
    x = selected.detach().cpu().float()
    if not torch.isfinite(x).all():
        raise AssertionError("nonfinite router logits")
    shifted = (x - x.amax(-1, keepdim=True)).double()
    exact = shifted.exp()
    eps = torch.finfo(torch.float32).eps
    tiny = torch.finfo(torch.float32).tiny
    error = 3 * eps * exact + tiny
    lo, hi = (exact - error).clamp_min(0), exact + error
    steps = x.shape[-1] - 1
    gamma = steps * (eps / 2) / (1 - steps * (eps / 2))
    dlo = lo.sum(-1, keepdim=True) * (1 - gamma)
    dhi = hi.sum(-1, keepdim=True) * (1 + gamma)
    lower = (lo / dhi) * (1 - 2 * eps)
    upper = (hi / dlo) * (1 + 2 * eps)
    return R(lower, dtype), R(upper, dtype)


def sigmoid_bounds(x, *, dtype=torch.bfloat16):
    round_low = partial(R, dtype=dtype)
    arg = round_low(-round_low(x.abs()) * round_low(torch.tensor(1.4426950408889634)))
    exact = torch.exp2(arg)
    err = exact * torch.finfo(dtype).eps + torch.finfo(dtype).tiny
    lo = round_low((exact - err).clamp_min(0))
    hi = round_low(exact + err)
    dlo = round_low(1 + lo)
    dhi = round_low(1 + hi)
    lower = torch.where(x >= 0, 1 / dhi, lo / dhi)
    upper = torch.where(x >= 0, 1 / dlo, hi / dlo)
    lower = torch.nextafter(lower, torch.full_like(lower, -float("inf")))
    upper = torch.nextafter(upper, torch.full_like(upper, float("inf")))
    return (round_low(lower.float()), round_low(upper.float()))


def fp32_sigmoid_bounds(x):
    """CUDA expf, FP32 addition and division interval for sigmoid."""
    eps = torch.finfo(torch.float32).eps
    tiny = torch.finfo(torch.float32).tiny
    exact_exp = torch.exp(-x.double())
    error = 3 * eps * exact_exp + tiny
    exp_lo = (exact_exp - error).clamp_min(0)
    exp_hi = exact_exp + error
    denominator_lo = (1 + exp_lo) * (1 - eps)
    denominator_hi = (1 + exp_hi) * (1 + eps)
    return (1 / denominator_hi) * (1 - eps), (1 / denominator_lo) * (1 + eps)


def check(actual, lo, hi):
    if not (torch.isfinite(lo).all() and torch.isfinite(hi).all() and (lo <= hi).all()):
        raise AssertionError("invalid reference interval")
    actual = actual.double()
    bad = (actual < lo) | (actual > hi) | ~torch.isfinite(actual)
    r = {
        "elements": actual.numel(),
        "outside_enclosure": int(bad.sum()),
        "max_outside": float(torch.maximum(lo - actual, actual - hi).clamp_min(0).max()),
    }
    return r


def gated_norm_bounds(x, gate, w, *, epsilon=1e-6, activation="silu", activation_dtype=None):
    dtype = x.dtype
    round_low = partial(R, dtype=dtype)
    x = x.double()
    gate = gate.double()
    width = x.shape[-1]
    variance = (x * x).sum(-1, keepdim=True) / width
    gamma = width * 2 ** (-24) / (1 - width * 2 ** (-24))
    vlo = (variance * (1 - gamma) + float(torch.tensor(epsilon))).float().double()
    vhi = (variance * (1 + gamma) + float(torch.tensor(epsilon))).float().double()
    rlo = torch.rsqrt(vhi) * (1 - 3 * 2 ** (-23))
    rhi = torch.rsqrt(vlo) * (1 + 3 * 2 ** (-23))
    nlo, nhi = product(x, x, rlo, rhi)
    nlo, nhi = (round_low(nlo.float()), round_low(nhi.float()))
    nlo, nhi = product(nlo, nhi, w.double(), w.double())
    eps = torch.finfo(torch.float32).eps
    weight_error = torch.maximum(nlo.abs(), nhi.abs()) * eps + torch.finfo(torch.float32).tiny
    nlo, nhi = nlo - weight_error, nhi + weight_error
    nlo, nhi = (nlo.float().double(), nhi.float().double())
    if w.dtype == dtype:
        nlo, nhi = (round_low(nlo), round_low(nhi))
    resolved_activation = activation_dtype or (torch.float32 if activation == "sigmoid" else dtype)
    if resolved_activation is torch.float32:
        glo, ghi = fp32_sigmoid_bounds(gate)
    else:
        glo, ghi = sigmoid_bounds(gate, dtype=resolved_activation)
    if activation == "silu":
        gate_operand = R(gate, resolved_activation)
        glo, ghi = product(gate_operand, gate_operand, glo, ghi)
        glo, ghi = R(glo.float(), resolved_activation), R(ghi.float(), resolved_activation)
    elif activation != "sigmoid":
        raise ValueError("unknown gated norm activation")
    lo, hi = product(nlo, nhi, glo, ghi)
    output_error = torch.maximum(lo.abs(), hi.abs()) * eps + torch.finfo(torch.float32).tiny
    lo, hi = lo - output_error, hi + output_error
    return round_low(lo.float()), round_low(hi.float())


def gated_norm_enclosure(x, gate, w, y, *, epsilon=1e-6, activation="silu", activation_dtype=None):
    lo, hi = gated_norm_bounds(x, gate, w, epsilon=epsilon, activation=activation,
                              activation_dtype=activation_dtype)
    return check(y, lo, hi)


def scaled_silu_bounds(x, *, divisor, activation_dtype=torch.float32):
    """Enclose divide/store -> declared activation arithmetic -> output/store."""
    if not torch.isfinite(x).all() or divisor <= 0:
        raise ValueError("scaled SiLU requires finite inputs and positive divisor")
    dtype = x.dtype
    stored = R(x.float() / divisor, dtype)
    if activation_dtype is not torch.float32:
        gate_lo, gate_hi = sigmoid_bounds(stored, dtype=activation_dtype)
        operand = R(stored, activation_dtype)
        if activation_dtype is torch.float16:
            saturated = stored.abs() > 65504.0
            operand = torch.where(saturated, 0.0, operand)
        lo, hi = product(operand, operand, gate_lo, gate_hi)
        lo, hi = R(lo.float(), activation_dtype), R(hi.float(), activation_dtype)
        if activation_dtype is torch.float16:
            tail = torch.where(stored > 0, stored, 0.0)
            lo, hi = torch.where(saturated, tail, lo), torch.where(saturated, tail, hi)
        return R(lo, dtype), R(hi, dtype)
    eps = torch.finfo(torch.float32).eps
    tiny = torch.finfo(torch.float32).tiny
    exact_exp = torch.exp(-stored)
    exp_error = 3 * eps * exact_exp + tiny
    exp_lo = (exact_exp - exp_error).clamp_min(0)
    exp_hi = exact_exp + exp_error
    denominator_lo = (1 + exp_lo) * (1 - eps)
    denominator_hi = (1 + exp_hi) * (1 + eps)
    reciprocal_lo = (1 / denominator_hi) * (1 - eps)
    reciprocal_hi = (1 / denominator_lo) * (1 + eps)
    lo, hi = product(stored, stored, reciprocal_lo, reciprocal_hi)
    negative = torch.full_like(lo.float(), -float("inf"))
    positive = torch.full_like(hi.float(), float("inf"))
    return R(torch.nextafter(lo.float(), negative), dtype), R(
        torch.nextafter(hi.float(), positive), dtype
    )


def swiglu_bounds(gate, up, *, activation_dtype=torch.float32):
    """Enclose stored SiLU followed by stored multiplication with the up branch."""
    if gate.shape != up.shape or gate.dtype != up.dtype:
        raise ValueError("SwiGLU operands differ")
    activated_lo, activated_hi = scaled_silu_bounds(
        gate, divisor=1.0, activation_dtype=activation_dtype,
    )
    lo, hi = product(activated_lo, activated_hi, up.double(), up.double())
    return R(lo.float(), gate.dtype), R(hi.float(), gate.dtype)


class SwiGLUWitnessBatch:
    """Bounded, post-forward elementwise proofs; no result or state reuse."""

    def __init__(self, max_pending=32, max_bytes=1 << 20):
        if any(type(n) is not int or n <= 0 for n in (max_pending, max_bytes)):
            raise ValueError("SwiGLU proof batch requires positive limits")
        self.max_pending, self.max_bytes = max_pending, max_bytes
        self.pending, self.bytes, self.errors = [], 0, []

    def add(self, gate, up, actual, *, activation_dtype, consumer):
        if (gate.shape != up.shape or gate.shape != actual.shape
                or gate.dtype != up.dtype or gate.dtype != actual.dtype):
            raise ValueError("SwiGLU witness operands differ")
        size = sum(t.numel() * t.element_size() for t in (gate, up, actual))
        if self.pending and self.bytes + size > self.max_bytes:
            self.flush()
        receipt = {"kind": "swiglu", "consumer": consumer}
        values = tuple(t.detach().to(device="cpu", copy=True).reshape(-1)
                       for t in (gate, up, actual))
        self.pending.append((values, activation_dtype, receipt))
        self.bytes += size
        if len(self.pending) >= self.max_pending or self.bytes >= self.max_bytes:
            self.flush()
        return receipt

    def flush(self):
        groups = {}
        for values, dtype, receipt in self.pending:
            groups.setdefault((values[0].dtype, dtype), []).append((values, receipt))
        for (_, activation_dtype), entries in groups.items():
            try:
                lo, hi = swiglu_bounds(
                    torch.cat([values[0] for values, _ in entries]),
                    torch.cat([values[1] for values, _ in entries]),
                    activation_dtype=activation_dtype,
                )
                offset = 0
                for (_, _, actual), receipt in entries:
                    stop = offset + actual.numel()
                    proof = check(actual, lo[offset:stop], hi[offset:stop])
                    receipt.update(proof)
                    if proof["outside_enclosure"]:
                        self.errors.append((receipt["consumer"], proof))
                    offset = stop
            except Exception as exc:
                self.errors.append((tuple(r["consumer"] for _, r in entries), str(exc)))
        self.pending.clear()
        self.bytes = 0

    def finish(self):
        self.flush()
        if self.errors:
            raise AssertionError({"swiglu_witness_failures": self.errors})


def scaled_activation_bounds(x, *, divisor, multiplier=1.0, activation="silu",
                             activation_dtype=torch.float32):
    """Independent divide/store, activation/store, multiply/store boundaries."""
    if activation == "silu":
        lo, hi = scaled_silu_bounds(x, divisor=divisor, activation_dtype=activation_dtype)
    elif activation == "sigmoid":
        stored = R(x.float() / divisor, x.dtype)
        if activation_dtype is torch.float32:
            lo, hi = fp32_sigmoid_bounds(stored)
        else:
            lo, hi = sigmoid_bounds(stored, dtype=activation_dtype)
        lo, hi = R(lo.float(), x.dtype), R(hi.float(), x.dtype)
    else:
        raise ValueError("unknown scaled activation")
    lo, hi = product(lo, hi, torch.full_like(lo, multiplier), torch.full_like(hi, multiplier))
    return R(lo.float(), x.dtype), R(hi.float(), x.dtype)


def stream_combine_bounds(residual, update, logits, *, activation_dtype):
    """Witness the fused gate/store, product/store and residual-add/store rule."""
    streams = logits.shape[-1]
    if residual.shape != (update.shape[0], streams * update.shape[-1]):
        raise ValueError("residual stream geometry differs")
    gl, gh = scaled_activation_bounds(
        logits, divisor=streams, multiplier=2.0, activation="sigmoid",
        activation_dtype=activation_dtype,
    )
    operand = update.double().unsqueeze(1)
    pl, ph = product(operand, operand, gl.unsqueeze(-1), gh.unsqueeze(-1))
    pl, ph = R(pl.float(), update.dtype), R(ph.float(), update.dtype)
    original = residual.double().reshape_as(pl)
    return (R((original + pl).float(), residual.dtype).reshape_as(residual),
            R((original + ph).float(), residual.dtype).reshape_as(residual))


def moe_shared_combine_bounds(shared, logits, routed, *, activation_dtype):
    """Stored shared sigmoid and product, then stored routed addition."""
    gl, gh = scaled_activation_bounds(
        logits, divisor=1.0, activation="sigmoid", activation_dtype=activation_dtype,
    )
    sl, sh = product(shared.double(), shared.double(), gl, gh)
    sl, sh = R(sl.float(), shared.dtype), R(sh.float(), shared.dtype)
    return (R((routed.double() + sl).float(), routed.dtype),
            R((routed.double() + sh).float(), routed.dtype))


def signed_root_sigmoid_bounds(key, query, *, activation_dtype, magnitude_floor=1e-6):
    """Stored products, FP32 dot reduction, stored scale/root, declared sigmoid."""
    terms = R((key.double() * query.double()).float(), key.dtype)
    width = key.shape[-1]
    u = 2.0**-24
    gamma = (width - 1) * u / (1 - (width - 1) * u)
    error = gamma * terms.abs().sum(-1)
    exact = terms.sum(-1)
    divisor = float(torch.tensor(width**0.5, dtype=torch.float32))
    lower = R((R((exact - error).float(), key.dtype) / divisor).float(), key.dtype)
    upper = R((R((exact + error).float(), key.dtype) / divisor).float(), key.dtype)
    floor = R(torch.tensor(magnitude_floor), key.dtype)

    def transform(value):
        root = R(value.abs().clamp_min(floor).sqrt().float(), key.dtype)
        return root * value.sign()

    lo, hi = transform(lower), transform(upper)
    if activation_dtype is torch.float32:
        gl, _ = fp32_sigmoid_bounds(lo)
        _, gh = fp32_sigmoid_bounds(hi)
    else:
        from cubic_serve.guardrail.quantized_graph.state_bounds import sigmoid_range
        gl, gh = sigmoid_range(lo, hi, dtype=activation_dtype)
    return R(gl.float(), key.dtype), R(gh.float(), key.dtype)
