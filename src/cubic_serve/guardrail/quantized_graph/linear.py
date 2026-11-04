"""Deferred graph boundary recorder and independent producer replay diagnostic."""

from contextlib import contextmanager
from unittest.mock import patch

import torch

from cubic_serve.guardrail.quantization import check_linear_enclosure
from cubic_serve.ops.cuda.linear.quantized import PackedProjectionDispatch
from cubic_serve.ops.cuda.moe.quantized import PackedRoutedExpertsPrimitive
from cubic_serve.ops.eager.quantization import QuantizedReferenceOperators


def same(a, b):
    return (
        a.dtype == b.dtype
        and a.numel() == b.numel()
        and torch.equal(
            a.contiguous().view(torch.uint8).flatten(), b.contiguous().view(torch.uint8).flatten()
        )
    )


def check_int8_boundary(codes, scales, lower, upper):
    """Witness a stored INT8 boundary from its producer's floating enclosure.

    The actual scale is first witnessed against the maximum envelope, then
    used to witness the codes. This runs only after the complete forward.
    """
    from .bounds import check

    q, s = codes.cpu(), scales.cpu().reshape(-1, 1)
    lo, hi = lower.cpu().double(), upper.cpu().double()
    crosses_zero = (lo <= 0) & (hi >= 0)
    abs_lo = torch.where(crosses_zero, 0.0, torch.minimum(lo.abs(), hi.abs()))
    abs_hi = torch.maximum(lo.abs(), hi.abs())
    s_lo = (abs_lo.amax(-1, keepdim=True) / 127.0).float()
    s_hi = (abs_hi.amax(-1, keepdim=True) / 127.0).float()
    valid = ((s >= s_lo) & (s <= s_hi)) | ((s == 1.0) & crosses_zero.all(-1, keepdim=True))
    if not bool((torch.isfinite(s) & (s > 0) & valid).all()):
        raise AssertionError("INT8 producer scale is outside its floating envelope")
    inverse = (1.0 / s.double()).float()

    def quantize(values):
        v = torch.where(
            s >= torch.finfo(torch.float32).tiny,
            values.float() * inverse,
            (values / s.double()).float(),
        )
        return v.round().clamp(-127, 127).double()

    code_lo, code_hi = quantize(lo), quantize(hi)
    proof = check(q, code_lo, code_hi)
    if proof["outside_enclosure"]:
        row, column = torch.nonzero((q < code_lo) | (q > code_hi))[0].tolist()
        raise AssertionError(
            "INT8 producer codes are outside their floating envelope: "
            + str({**proof, "first_row": row, "first_column": column,
                   "code": int(q[row, column]),
                   "code_interval": (int(code_lo[row, column]), int(code_hi[row, column])),
                   "floating_interval": (float(lo[row, column]), float(hi[row, column])),
                   "token_scale": float(s[row, 0])})
        )
    return proof


class Recorder:
    def __init__(self):
        self.linear = []
        self.experts = []
        self.embeddings = []
        self.immutable_embedding_group = None
        self.latest = {}
        self.pending = {}
        self.shared = {}
        self.int8_scales = {}
        self.encoded_inputs = {}

    def record_activation(self, gate, up, out):
        self.pending[out.data_ptr()] = (self.latest[gate.data_ptr()], self.latest[up.data_ptr()])

    def record_projection(self, weight, inputs, output, *, int8_token_scales=None):
        """Retain only immutable snapshots of one actual native projection."""
        index = len(self.linear)
        if int8_token_scales is not None:
            self.int8_scales[index] = int8_token_scales.detach().clone()
        if inputs.data_ptr() in self.pending:
            self.shared[index] = self.pending.pop(inputs.data_ptr())
        self.linear.append((weight, inputs.detach().clone(), output.detach().clone()))
        self.latest[output.data_ptr()] = index

    @contextmanager
    def observe(self):
        from cubic_serve.ops.cuda.embedding.immutable import ImmutableEmbedding
        immutable = ImmutableEmbedding.launch

        def embedding(op, x, w, y):
            result = immutable(op, x, w, y)
            self.immutable_embedding_group = op.table.group
            self.embeddings.append((w, x.detach().clone(), y.detach().clone()))
            return result

        linear = PackedProjectionDispatch.launch
        expert = PackedRoutedExpertsPrimitive.launch

        def projection(op, x, w, y):
            result = linear(op, x, w, y)
            if op.operation == "embedding":
                self.embeddings.append((w, x.detach().clone(), y.detach().clone()))
                return result
            x = op.operators.projection_input(x, w)
            encoded = op.operators.int8_input(x) if x.dtype == torch.int8 else None
            self.record_projection(w, x, y, int8_token_scales=(
                None if encoded is None else encoded.scales))
            return result

        def routed(op):
            result = expert(op)
            self.experts.append(
                (
                    op._weights,
                    op._operands[0].detach().clone(),
                    op.route_ids.detach().clone(),
                    op.packed.detach().clone(),
                    op.activated.detach().clone(),
                    op.routes.detach().clone(),
                )
            )
            return result

        with (
            patch.object(ImmutableEmbedding, "launch", embedding),
            patch.object(PackedProjectionDispatch, "launch", projection),
            patch.object(PackedRoutedExpertsPrimitive, "launch", routed),
        ):
            yield

    def freeze(self):
        self.events = {}
        self.activation_sources = {}
        self.encoded_inputs = {}
        keys = []
        for ordinal, (w, x, y) in enumerate(self.linear):
            key = (id(w), len(self.events.get(id(w), ())))
            keys.append(key)
            if ordinal in self.int8_scales:
                self.encoded_inputs[key] = self.int8_scales[ordinal]
            self.events.setdefault(id(w), []).append(
                (w, x.reshape(-1, x.shape[-1]), y.reshape(-1, y.shape[-1]))
            )
        for index, parents in self.shared.items():
            self.activation_sources[keys[index]] = tuple(
                (keys[j], self.linear[j][2]) for j in parents
            )
        for groups, x, ids, packed, activated, routes in self.experts:
            topk = ids.shape[-1]
            ids = ids.cpu().flatten().tolist()
            packed = packed.reshape(len(ids), -1)
            activated = activated.reshape(len(ids), -1)
            routes = routes.reshape(len(ids), -1)
            for row, e in enumerate(ids):
                producers = []
                for i, group in enumerate(groups):
                    w = group[e]
                    n = w.shape[0]
                    source = (
                        activated[row : row + 1] if i == 2 else x[row // topk : row // topk + 1]
                    )
                    out = (
                        routes[row : row + 1]
                        if i == 2
                        else packed[row : row + 1, i * n : (i + 1) * n]
                    )
                    key = (id(w), len(self.events.get(id(w), ())))
                    self.events.setdefault(id(w), []).append((w, source, out))
                    if i < 2:
                        producers.append((key, out))
                    else:
                        self.activation_sources[key] = tuple(producers)


class ProducerMismatch(Exception):
    pass


class Replay(QuantizedReferenceOperators):
    def __init__(self, recorder, names):
        super().__init__(128)
        self.events = recorder.events
        self.names = names
        self.reports = []
        self.used = set()
        self.failure = None
        self.activation_sources = recorder.activation_sources
        self.activation_proofs = []
        self.encoded_inputs = recorder.encoded_inputs
        self.immutable_embedding_group = recorder.immutable_embedding_group
        self.embedding_events = recorder.embeddings
        self.embedding_used = 0

    def embedding(self, ids, w, *, dtype):
        if self.embedding_used >= len(self.embedding_events):
            raise RuntimeError("embedding event schedule exhausted")
        weight, source, observed = self.embedding_events[self.embedding_used]
        if weight is not w or not same(ids, source):
            raise ProducerMismatch("unwitnessed embedding inputs")
        expected = super().embedding(ids, w, dtype=dtype)
        if not same(expected, observed):
            raise AssertionError("embedding differs from independent original rows")
        self.embedding_used += 1
        return expected

    def linear(self, x, w):
        rows = x.reshape(-1, x.shape[-1])
        events = self.events.get(id(w), [])
        if not events:
            raise RuntimeError("missing candidate weight event " + self.names.get(id(w), "unknown"))
        for i, (_, source, y) in enumerate(events):
            if (id(w), i) in self.used:
                continue
            scales = self.encoded_inputs.get((id(w), i))
            decoded = source if scales is None else source.float() * scales[:, None]
            expected_rows = rows
            if scales is not None and rows.dtype != torch.float32:
                from cubic_serve.quantization.common.reference import wna8_activation

                expected_rows = wna8_activation(rows)
            coherent = same(expected_rows, decoded)
            if not coherent and (id(w), i) in self.activation_sources:
                (gate_key, gate), (up_key, up) = self.activation_sources[id(w), i]
                if gate_key in self.used and up_key in self.used:
                    from cubic_serve.ops.eager.activation import silu_stored

                    canonical = silu_stored(gate) * up
                    if same(rows, canonical):
                        from .bounds import R, check, product, sigmoid_bounds

                        g = gate.cpu().double()
                        u = up.cpu().double()
                        lo, hi = sigmoid_bounds(g, dtype=gate.dtype)
                        lo, hi = product(g, g, lo, hi)
                        lo, hi = (R(lo, gate.dtype), R(hi, gate.dtype))
                        lo, hi = product(lo, hi, u, u)
                        if scales is None:
                            proof = check(source.cpu(), R(lo, source.dtype), R(hi, source.dtype))
                        else:
                            proof = check_int8_boundary(
                                source, scales, R(lo, gate.dtype), R(hi, gate.dtype)
                            )
                        if proof["outside_enclosure"]:
                            raise AssertionError("routed SwiGLU enclosure " + str(proof))
                        self.activation_proofs.append(
                            {"name": self.names.get(id(w), "unknown"), **proof}
                        )
                        coherent = True
            if coherent:
                report = check_linear_enclosure(source, w, y, int8_token_scales=scales)
                report["name"] = self.names.get(id(w), "unknown")
                self.reports.append(report)
                self.used.add((id(w), i))
                return y.clone().reshape(*x.shape[:-1], y.shape[-1])
        source = events[0][1]
        if (scales := self.encoded_inputs.get((id(w), 0))) is not None:
            source = source.float() * scales[:, None]
        d = (rows.float() - source.float()).abs()
        self.failure = {
            "producer_of": self.names.get(id(w), "unknown"),
            "max_abs": float(d.max()),
            "unequal": int((rows != source).sum()),
            "elements": rows.numel(),
            "proved_linear_boundaries": len(self.reports),
        }
        raise ProducerMismatch(str(self.failure))
