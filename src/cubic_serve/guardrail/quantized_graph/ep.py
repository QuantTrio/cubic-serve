"""EP-only witness extension: explicit ownership, local folds, observed EP sum."""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import patch

import torch
from torch import Tensor

import cubic_serve.models.qwen3_5_moe.model as model_module
from cubic_serve.models.qwen3_5_moe.model import (
    activation_sigmoid,
    activation_silu,
)

from . import collective as collective_base
from . import linear as linear_base
from .bounds import R, check, product, sigmoid_bounds
from .nonlinear import NonlinearReplay as TPNonlinearReplay
from .replay import Reference as TPReference
from .replay import Replay as TPReplay
from .replay import ordered_sum


def ep_reference(self, hidden: Tensor, *, prefix: str, operators) -> Tensor:
    flat = hidden.reshape(-1, hidden.shape[-1])
    moe = self.config.graph.text.moe
    if moe is None:
        raise ValueError("reference MoE requires declared expert semantics")
    routes = model_module.route_qwen3_5_moe(
        flat,
        self.weights[f"{prefix}.gate.weight"],
        top_k=moe.experts_per_token,
        operators=operators,
    )
    result = torch.zeros_like(flat)
    for row in range(flat.shape[0]):
        for slot, expert in enumerate(routes.expert_ids[row].tolist()):
            if not self._audit_owned[0] <= expert < self._audit_owned[1]:
                continue
            expert_prefix = f"{prefix}.experts.{expert}"
            x = flat[row : row + 1]
            gate = activation_silu(
                operators.linear(x, self.weights[f"{expert_prefix}.gate_proj.weight"])
            )
            up = operators.linear(x, self.weights[f"{expert_prefix}.up_proj.weight"])
            out = operators.linear(gate * up, self.weights[f"{expert_prefix}.down_proj.weight"])
            result[row] = result[row] + (out[0] * routes.weights[row, slot]).to(hidden.dtype)
    result = operators.ep_output(result, prefix)
    gate = activation_silu(
        operators.linear(flat, self.weights[f"{prefix}.shared_expert.gate_proj.weight"])
    )
    up = operators.linear(flat, self.weights[f"{prefix}.shared_expert.up_proj.weight"])
    shared = operators.linear(gate * up, self.weights[f"{prefix}.shared_expert.down_proj.weight"])
    shared_gate = activation_sigmoid(
        operators.linear(flat, self.weights[f"{prefix}.shared_expert_gate.weight"])
    )
    return (result + (shared * shared_gate).to(hidden.dtype)).reshape_as(hidden)


@contextmanager
def prepare():
    with (
        collective_base.prepare(),
        patch.object(model_module.Qwen3_5MoeReferenceFeedForward, "__call__", ep_reference),
    ):
        yield


def configure(stage):
    t = stage.topology
    assert t.ep_size > 1, "EP witness requires expert owners"
    e = stage.body.model.config.graph.text.moe.expert_count
    first = t.expert_coordinates(stage.rank)[0] * (e // t.ep_size)
    owned = (first, first + e // t.ep_size)
    stage.body.model._moe._audit_owned = owned
    return owned


class LinearRecorder(linear_base.Recorder):
    def __init__(self, owned):
        super().__init__()
        self.owned = owned

    @contextmanager
    def observe(self):
        with super().observe():
            yield
        first, end = self.owned
        self.experts = [
            (tuple({first + i: w for i, w in enumerate(group)} for group in groups), *rest)
            for groups, *rest in self.experts
        ]

    def freeze(self):
        self.events = {}
        self.activation_sources = {}
        keys = []
        for w, x, y in self.linear:
            key = (id(w), len(self.events.get(id(w), ())))
            keys.append(key)
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
                if e not in groups[0]:
                    continue
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


class Recorder(collective_base.Recorder):
    def __init__(self):
        super().__init__()
        self.ep_events = []
        self.fold_proofs = []
        self.fold_schedule = []
        self.folds = {}

    @contextmanager
    def observe(self, stage, linear_recorder=None):
        body = stage.body
        old = body._expert_output_reduce
        assert old is not None
        with super().observe(stage, linear_recorder):
            def complete(x):
                source = x.clone()
                result = old(x)
                out = x.clone()
                i = len(self.ep_events)
                self.events.append(("ep_output", source, out))
                self.event_semantics.append(("expert_output", i))
                self.ep_events.append((source, out))
                return result

            body._expert_output_reduce = complete
            try:
                yield
            finally:
                body._expert_output_reduce = old

    def attach_expert_scales(self, recorder):
        if self.routes:
            super().attach_expert_scales(recorder)
        else:
            assert not self.routed_scales, "unsplit EP expert must not have a TP scale collective"

    def map_outputs(self, recorder):
        if self.routes:
            super().map_outputs(recorder)
            return
        saved = recorder.experts
        recorder.experts = []
        try:
            super().map_outputs(recorder)
        finally:
            recorder.experts = saved

    def retain_logical(self, m, cap):
        super().retain_logical(m, cap)
        self.ep_events = [(x[:m], y[:m]) for x, y in self.ep_events]


def local_snapshots(linear, nonlinear, tp, owned):
    first, end = owned
    out = []
    for ids, weights, routes, y in nonlinear.events["routes"]:
        mask = (ids >= first) & (ids < end)
        shaped = routes.reshape(*ids.shape, -1)
        local = torch.where(mask[..., None], shaped, torch.zeros_like(shaped))
        out.append((ids, weights, local.reshape_as(routes), y))
    nonlinear.events["routes"] = out
    if tp.routes:
        assert len(tp.routes) == len(out), "ETP route completion coverage"
        before = []
        after = []
        for i, (ids, _weights, local, _y) in enumerate(out):
            mask = (ids >= first) & (ids < end)
            partial, completed = tp.routes[i]
            partial = torch.where(mask[..., None], partial, torch.zeros_like(partial))
            completed = torch.where(mask[..., None], completed, torch.zeros_like(completed))
            assert linear_base.same(completed, local), "ETP route producer differs"
            before.append(partial)
            after.append(completed)
        tp.routes = list(zip(before, after, strict=True))
        linear.experts = [(*record[:-1], before[i]) for i, record in enumerate(linear.experts)]
    else:
        tp.routes = [(t[2], t[2]) for t in out]
        linear.experts = [(*record[:-1], out[i][2]) for i, record in enumerate(linear.experts)]


class Reference(TPReference):
    def __init__(self, row_weights, span, group, *, ep_group, etp_group, expert_rows):
        super().__init__(
            row_weights, span, group,
            row_groups={weight: etp_group for weight in expert_rows} if etp_group else {},
        )
        self.ep_group = ep_group

    def ep_output(self, x, prefix):
        return ordered_sum(x, self.ep_group)


class Replay(TPReplay):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.ep_used = 0

    def ep_output(self, x, prefix):
        before, after = self.tp.ep_events[self.ep_used]
        self.ep_used += 1
        if not linear_base.same(x.reshape_as(before), before):
            raise AssertionError("EP output producer differs: " + prefix)
        return after.clone().reshape_as(x)


class NonlinearReplay(TPNonlinearReplay):
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
        self.equal(accum, routed_saved, "local EP route fold")
        before, after = self.ep_events[i]
        self.equal(accum, before, "EP sum input")
        self.equal(after, routed, "shared combine EP input")
        from cubic_serve.ops.eager.activation import sigmoid_stored

        canonical = after + (shared * sigmoid_stored(gate)).to(shared.dtype)
        self.equal(expected, canonical, "EP canonical producer chain")
        lo, hi = sigmoid_bounds(gate.cpu().double(), dtype=gate.dtype)
        lo, hi = product(shared.cpu().double(), shared.cpu().double(), lo, hi)
        lo, hi = (R(lo, out.dtype), R(hi, out.dtype))
        result = check(
            out.cpu(), R(routed.cpu().double() + lo, out.dtype),
            R(routed.cpu().double() + hi, out.dtype),
        )
        if result["outside_enclosure"]:
            raise AssertionError("shared sigmoid combine " + str(result))
        self.proved.append((i, "local_route_fold_EP_sum_shared_sigmoid"))
        return out.reshape_as(expected).clone()
