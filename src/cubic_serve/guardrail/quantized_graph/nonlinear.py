"""Deferred causal witnesses for FP16/BF16 activation and BF16 recurrent state."""

from contextlib import ExitStack, contextmanager
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import torch
import triton.language as tl

import cubic_serve.models.qwen3_5.model as model

from .state_bounds import evaluate


def state_rows(state, indices, flattened_ndim):
    """Own selected logical rows from flat or (owner, layer, ...) storage.

    Advanced indexing already creates an independent stream-ordered copy;
    callers must not add a second clone merely to freeze the same snapshot.
    """
    rows = indices.long()
    if state.ndim == flattened_ndim + 1:
        return state[rows // state.shape[1], rows % state.shape[1]]
    return state[rows]


class ColdRecorder:
    def __init__(self):
        self.events = {
            n: [] for n in ("conv", "gdn", "norm", "zero", "combine", "routes", "attention_gate")
        }

    @contextmanager
    def observe(self, body, linear_recorder):
        self.norm_epsilon = body.model.config.graph.text.rms_norm_epsilon
        originals = {
            n: getattr(body, n)
            for n in (
                "conv",
                "gdn",
                "gdn_norm",
                "hidden_norm",
                "mlp_gate",
                "_packed_moe_combine",
                "q_norm",
                "k_norm",
                "sigmoid_gate",
            )
            if hasattr(body, n)
        }

        def conv(x, w, state, src, dst, segments, y):
            logical_m = getattr(self, "logical_m", src.shape[0])
            before = (x.clone(), w.clone(), state_rows(state, src[:logical_m], 3))
            result = originals["conv"].launch(x, w, state, src, dst, segments, y)
            self.events["conv"].append((*before, y.clone(), state_rows(state, dst[:logical_m], 3)))
            return result

        def gdn(x, packed, a, b, alog, bias, state, src, dst, segments, y, **kw):
            logical_m = getattr(self, "logical_m", src.shape[0])
            before = (*tuple(t.clone() for t in (x, a, b, alog, bias)),
                      state_rows(state, src[:logical_m], 4))
            result = originals["gdn"].launch(
                x, packed, a, b, alog, bias, state, src, dst, segments, y, **kw
            )
            self.events["gdn"].append(
                (*before, y.clone(), state_rows(state, dst[:logical_m], 4), packed.clone())
            )
            return result

        def norm(x, gate, w, y):
            before = tuple(t.clone() for t in (x, gate, w))
            result = originals["gdn_norm"].launch(x, gate, w, y)
            self.events["norm"].append((*before, y.clone()))
            return result

        def zero(x, w, y):
            before = (x.clone(), w.clone())
            result = originals["hidden_norm"].launch(x, w, y)
            self.events["zero"].append((*before, y.clone()))
            return result

        def extra_zero(name):
            def run(x, w, y):
                before = (x.clone(), w.clone())
                result = originals[name].launch(x, w, y)
                self.events["zero"].append((*before, y.clone()))
                return result

            return run

        def attention_gate(x, gate, y):
            before = (x.clone(), gate.clone())
            result = originals["sigmoid_gate"].launch(x, gate, y)
            self.events["attention_gate"].append((*before, y.clone()))
            return result

        def swiglu(gate, up, out):
            result = originals["mlp_gate"].launch(gate, up, out)
            linear_recorder.record_activation(gate, up, out)
            return result

        def combine(shared, gate, routed, out):
            before = tuple(t.clone() for t in (shared, gate, routed))
            result = originals["_packed_moe_combine"].launch(shared, gate, routed, out)
            self.events["combine"].append((*before, out.clone()))
            return result

        from cubic_serve.ops.cuda.moe.quantized import PackedRoutedExpertsPrimitive

        route_launch = PackedRoutedExpertsPrimitive.launch

        def routes(op):
            result = route_launch(op)
            self.events["routes"].append(
                tuple(
                    t.clone() for t in (op.route_ids, op.route_weights, op.routes, op._operands[-1])
                )
            )
            return result

        for n, fn in [
            ("q_norm", extra_zero("q_norm")),
            ("k_norm", extra_zero("k_norm")),
            ("sigmoid_gate", attention_gate),
            ("mlp_gate", swiglu),
            ("_packed_moe_combine", combine),
            ("conv", conv),
            ("gdn", gdn),
            ("gdn_norm", norm),
            ("hidden_norm", zero),
        ]:
            if n in originals:
                setattr(body, n, SimpleNamespace(launch=fn))
        try:
            with patch.object(PackedRoutedExpertsPrimitive, "launch", routes):
                yield
        finally:
            for n, op in originals.items():
                setattr(body, n, op)


class ColdReplay:
    qk_proofs = ("normalized_q_fp64", "normalized_k_fp64")
    state_proof = "cold_state_beta_interval"

    def __init__(self, recorder):
        self.r = recorder
        self.indices = {k: 0 for k in recorder.events}
        self.proved = []
        self.route_reference = []

    def take(self, kind):
        i = self.indices[kind]
        self.indices[kind] += 1
        if i >= len(self.r.events[kind]):
            raise RuntimeError("nonlinear schedule exhausted " + kind)
        return (i, self.r.events[kind][i])

    def equal(self, a, b, name):
        if (
            a.dtype != b.dtype
            or a.numel() != b.numel()
            or (
                not torch.equal(
                    a.contiguous().view(torch.uint8).flatten(),
                    b.contiguous().view(torch.uint8).flatten(),
                )
            )
        ):
            raise AssertionError("unwitnessed nonlinear input " + name)

    def proof(self, i, *keys):
        for key in keys:
            r = self.r.proofs[i]["checks"][key]
            if r["outside_enclosure"]:
                raise AssertionError(f"nonlinear enclosure failed {i} {key}: {r}")
            self.proved.append((i, key))

    def conv(self, x, w, state, **kw):
        i, (cx, cw, cs, y, new) = self.take("conv")
        for a, b, n in [(x, cx, "conv x"), (w, cw, "conv weight"), (state, cs, "conv state")]:
            self.equal(a, b, n)
        expected = torch.cat((cs[:, :, 1:], cx.unsqueeze(-1)), dim=-1)
        self.equal(expected, new, "conv state update")
        self.proof(i, "conv_silu")
        return (y.reshape_as(x).clone(), new.clone())

    def gdn(self, q, k, v, a, b, alog, bias, state, **kw):
        i, (cx, ca, cb, calog, cbias, cs, y, new, packed) = self.take("gdn")
        cs = cs.to(state.device)
        x = torch.cat((q.flatten(2), k.flatten(2), v.flatten(2)), dim=-1)
        for x1, x2, n in [
            (x, cx, "GDN qkv"),
            (a, ca, "GDN a"),
            (b, cb, "GDN b"),
            (alog, calog, "GDN A_log"),
            (bias, cbias, "GDN bias"),
            (state, cs, "GDN state"),
        ]:
            self.equal(x1, x2, n)
        self.proof(i, *self.qk_proofs, self.state_proof, "output_fp32_reduction")
        return (y[:, None].clone(), new.to(state.device, copy=True))

    def norm(self, x, gate, w, **kw):
        i, (cx, cgate, cw, y) = self.take("norm")
        for a, b, n in [(x, cx, "norm input"), (gate, cgate, "norm gate"), (w, cw, "norm weight")]:
            self.equal(a, b, n)
        self.proof(i, "gated_norm_declared_math_bounds")
        return y.reshape_as(x).clone()

    def zero(self, x, w, **kw):
        i, (cx, cw, y) = self.take("zero")
        self.equal(x, cx, "RMS input")
        self.equal(w, cw, "RMS weight")
        xx = cx.cpu().double()
        ww = cw.cpu().float().add(1).double()
        n = xx.shape[-1]
        var = (xx * xx).sum(-1, keepdim=True) / n
        gamma = n * 2 ** (-24) / (1 - n * 2 ** (-24))
        eps = float(torch.tensor(kw["epsilon"], dtype=torch.float32))
        vlo = (var * (1 - gamma) + eps).float().double()
        vhi = (var * (1 + gamma) + eps).float().double()
        from .bounds import check, product

        lo, hi = product(
            xx, xx, torch.rsqrt(vhi) * (1 - 3 * 2 ** (-23)), torch.rsqrt(vlo) * (1 + 3 * 2 ** (-23))
        )
        lo, hi = product(lo.float().double(), hi.float().double(), ww, ww)
        r = check(y.cpu(), lo.float().to(y.dtype).double(), hi.float().to(y.dtype).double())
        if r["outside_enclosure"]:
            raise AssertionError(f"RMS enclosure {i}: {r}")
        self.proved.append((i, "zero_centered_rms"))
        return y.reshape_as(x).clone()

    def attention_gate(self, x, gate, **kw):
        i, (cx, cgate, y) = self.take("attention_gate")
        self.equal(x, cx, "attention output")
        self.equal(gate, cgate, "attention gate")
        from .bounds import R, check, product, sigmoid_bounds

        lo, hi = sigmoid_bounds(gate.cpu().double(), dtype=gate.dtype)
        lo, hi = product(
            x.cpu().double(), x.cpu().double(), lo.reshape(x.shape), hi.reshape(x.shape)
        )
        r = check(y.cpu().reshape(x.shape), R(lo, x.dtype), R(hi, x.dtype))
        if r["outside_enclosure"]:
            raise AssertionError("attention sigmoid enclosure " + str(r))
        self.proved.append((i, "attention_sigmoid"))
        return y.reshape_as(x).clone()

    def moe(self, original, owner, hidden, **kw):
        expected = original(owner, hidden, **kw)
        i, (shared, gate, routed, out) = self.take("combine")
        ids, weights, per_route, routed_saved = self.r.events["routes"][i]
        ref = self.route_reference[i]
        self.equal(ids.to(ref.expert_ids.dtype), ref.expert_ids, "route identities")
        self.equal(weights, ref.weights, "route probabilities")
        accum = torch.zeros_like(routed)
        values = per_route.reshape(ids.shape[0], ids.shape[1], -1)
        for j in range(ids.shape[1]):
            accum = accum + (values[:, j] * weights[:, j, None]).to(accum.dtype)
        self.equal(accum, routed_saved, "routed combine")
        self.equal(accum, routed, "shared combine routed input")
        from cubic_serve.ops.eager.activation import sigmoid_stored

        canonical = accum + (shared * sigmoid_stored(gate)).to(shared.dtype)
        self.equal(expected, canonical, "MoE canonical producer chain")
        from .bounds import R, check, product, sigmoid_bounds

        lo, hi = sigmoid_bounds(gate.cpu().double(), dtype=gate.dtype)
        lo, hi = product(shared.cpu().double(), shared.cpu().double(), lo, hi)
        lo, hi = (R(lo, out.dtype), R(hi, out.dtype))
        r = check(
            out.cpu(),
            R(routed.cpu().double() + lo, out.dtype),
            R(routed.cpu().double() + hi, out.dtype),
        )
        if r["outside_enclosure"]:
            raise AssertionError("shared sigmoid combine " + str(r))
        self.proved.append((i, "route_combine_and_shared_sigmoid"))
        return out.reshape_as(expected).clone()

    @contextmanager
    def install(self):
        import cubic_serve.models.qwen3_5_moe.model as moe_model

        original = moe_model.Qwen3_5MoeReferenceFeedForward.__call__
        route_original = moe_model.route_qwen3_5_moe

        def call(owner, hidden, **kw):
            return self.moe(original, owner, hidden, **kw)

        def route(*args, **kw):
            # This runs in deferred causal replay, after the observed forward.
            # The linear replay proves the router's actual input and logits.
            # Preserve exact expert identity while witnessing its separately
            # rounded probability boundary with FP32 arithmetic error bounds.
            from .bounds import check, route_softmax_bounds

            operators = kw["operators"]
            produced = []

            def linear(x, w):
                logits = operators.linear(x, w)
                produced.append(logits)
                return logits

            result = route_original(*args, **{**kw, "operators": SimpleNamespace(linear=linear)})
            i = len(self.route_reference)
            ids, weights, _, _ = self.r.events["routes"][i]
            assert weights.dtype == result.weights.dtype, "route storage dtype"
            self.equal(ids.to(result.expert_ids.dtype), result.expert_ids, "route identities")
            assert len(produced) == 1, "router producer coverage"
            selected = produced[0].gather(1, result.expert_ids)
            lo, hi = route_softmax_bounds(selected, weights.dtype)
            proof = check(weights.cpu(), lo, hi)
            if proof["outside_enclosure"]:
                raise AssertionError("route softmax enclosure " + str(proof))
            self.proved.append((i, "route_fp32_exp_sum_division"))
            result = type(result)(result.expert_ids, weights.clone())
            self.route_reference.append(result)
            return result

        with ExitStack() as stack:
            stack.enter_context(
                patch.object(moe_model.Qwen3_5MoeReferenceFeedForward, "__call__", call)
            )
            stack.enter_context(patch.object(moe_model, "route_qwen3_5_moe", route))
            for name, fn in [
                ("sigmoid_output_gate", self.attention_gate),
                ("depthwise_causal_conv1d", self.conv),
                ("gated_delta_net_recurrent", self.gdn),
                ("gated_rms_norm", self.norm),
                ("zero_centered_rms_norm", self.zero),
            ]:
                stack.enter_context(patch.object(model, name, fn))
            yield


class Recorder(ColdRecorder):
    def freeze(self):
        assert len(self.events["conv"]) == len(self.events["gdn"]) == len(self.events["norm"])
        # This runs after the observed forward and topology schedule check.
        # Keep the large immutable before/after states off the device while
        # reference math runs; replay restores only its current layer.
        for i, event in enumerate(self.events["gdn"]):
            values = list(event)
            for field in (5, 7):
                values[field] = values[field].to("cpu", copy=True)
            self.events["gdn"][i] = tuple(values)
        self.proofs = [
            evaluate(
                {k: tuple(t.cpu() for t in self.events[k][i]) for k in ("conv", "gdn", "norm")},
                norm_epsilon=self.norm_epsilon,
            )
            for i in range(len(self.events["conv"]))
        ]


class NonlinearReplay(ColdReplay):
    qk_proofs = ("normalized_q_fp32_bounds", "normalized_k_fp32_bounds")
    state_proof = "warm_state_interval"


@contextmanager
def observe_cache_producers(body):
    """Retain only ordered native inputs while a qualification prefix runs."""
    original = body.attentions
    records = []
    wrapped = []
    for layer, op in enumerate(original):
        window = op._window
        if window is None:
            raise ValueError("encoded prefix observation requires its native read window")

        def launch(*args, layer=layer, op=op, window=window):
            key = getattr(window, "producer_key", None)
            value = getattr(window, "producer_value", None)
            key = window.key if key is None else key
            value = window.value if value is None else value
            records.append((layer, key.clone(), value.clone()))
            return op.launch(*args)

        wrapped.append(SimpleNamespace(launch=launch))
    body.attentions = tuple(wrapped)
    try:
        yield records
    finally:
        body.attentions = original


def snapshot_cache_read(op, key, value, table, context_pages):
    """Stream-ordered immutable reads, compacted to the observed physical pages.

    Only copies/index gathers run here. Reference math, assertions, rank schedule
    checks and host inspection belong after the complete observed forward.
    """
    source, window = op._encoding, op._window
    layer = op.layer
    ids = table[:, :context_pages].reshape(-1).long()
    compact_table = torch.arange(ids.numel(), device=table.device, dtype=table.dtype).reshape(
        table.shape[0], context_pages)
    encoding = replace(
        source,
        key_scales=(None if source.key_scales is None
                    else source.key_scales[ids, layer][:, None].clone()),
        value_scales=(None if source.value_scales is None
                      else source.value_scales[ids, layer][:, None].clone()),
    )
    copied_window = replace(
        window, key=window.key.clone(), value=window.value.clone(),
        start_positions=window.start_positions.clone(), start_rows=window.start_rows.clone(),
        sources=window.sources.clone(),
        activity=None if window.activity is None else window.activity.clone(),
        producer_query=None if window.producer_query is None else window.producer_query.clone(),
        producer_key=None if window.producer_key is None else window.producer_key.clone(),
        producer_value=None if window.producer_value is None else window.producer_value.clone(),
        key_tail=(None if window.key_tail is None
                  else window.key_tail[:, layer:layer + 1].clone()),
        value_tail=(None if window.value_tail is None
                    else window.value_tail[:, layer:layer + 1].clone()),
    )
    return (encoding, copied_window, key[ids, layer][:, None].clone(),
            value[ids, layer][:, None].clone(), compact_table)


class PagedRecorder(Recorder):
    @contextmanager
    def observe(self, body, linear_recorder):
        original = body.attentions
        self.events["attention"] = []
        self.cache_reads = []
        self.attention_meta = []
        wrapped = []
        for layer, op in enumerate(original):
            self.attention_meta.append((op.meta.scale, op._grid[-1]))

            def launch(q, k, v, table, lengths, y, layer=layer, op=op):
                snapshot = (None if op._encoding is None else
                            snapshot_cache_read(op, k, v, table, self.context_pages))
                if snapshot is None:
                    before = (
                        q.clone(),
                        k[table[:, : self.context_pages].long(), layer]
                        .reshape(q.shape[0], -1, k.shape[-2], k.shape[-1]).clone(),
                        v[table[:, : self.context_pages].long(), layer]
                        .reshape(q.shape[0], -1, v.shape[-2], v.shape[-1]).clone(),
                        lengths.clone(),
                    )
                else:
                    native = snapshot[1]
                    before = (
                        q.clone(),
                        native.key if native.producer_key is None else native.producer_key,
                        native.value if native.producer_value is None else native.producer_value,
                        lengths.clone(),
                    )
                result = op.launch(q, k, v, table, lengths, y)
                self.events["attention"].append((*before, y.clone()))
                self.cache_reads.append(snapshot)
                return result

            wrapped.append(SimpleNamespace(launch=launch))
        body.attentions = tuple(wrapped)
        try:
            with super().observe(body, linear_recorder):
                yield
        finally:
            body.attentions = original


Base = PagedRecorder


class AttentionRecorder(Base):
    def __init__(self, context_pages):
        super().__init__()
        self.context_pages = context_pages

    @contextmanager
    def observe(self, body, linear_recorder):
        original = body.attentions
        for op in original:
            if op._attention_precision != "auto" or op._activation_type != tl.float32:
                raise RuntimeError("context witness requires native low QKV and FP32 softmax")
        metadata = [(op.meta.scale, op._grid[-1], op.reduction_abi["block_n"]) for op in original]
        with super().observe(body, linear_recorder):
            self.attention_meta = metadata
            yield
