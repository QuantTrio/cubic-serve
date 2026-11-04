"""Original calls plus immutable snapshots; all proofs run after forward."""

from contextlib import contextmanager
from unittest.mock import patch

import torch
import torch.distributed as dist

from cubic_serve.executor.quantized_model_runner import _RankOperators, _ScaleCollective

from .linear import same

bind = _RankOperators.bind_linear
routed = _RankOperators.bind_routed_fp8_input


def linear(op, x, w, y, **kw):
    result = bind(op, x, w, y, **kw)
    if isinstance(result, _ScaleCollective):
        result._audit_input = x
        result._audit_weight = w
    return result


def input_binding(op, x, **kw):
    result = routed(op, x, **kw)
    binding = result[2]
    if isinstance(binding, _ScaleCollective):
        binding._audit_input = x
        binding._audit_weight = None
    return result


@contextmanager
def prepare():
    with (
        patch.object(_RankOperators, "bind_linear", linear),
        patch.object(_RankOperators, "bind_routed_fp8_input", input_binding),
    ):
        yield


class Recorder:
    def __init__(self):
        self.events = []
        self.routes = []
        self.weight_scales = {}
        self.proofs = []
        self.routed_scales = {}
        self.expert_scales = {}
        self.origins = []
        self.global_outputs = {}
        self.event_semantics = []

    @contextmanager
    def observe(self, stage, linear_recorder=None):
        body = stage.body
        source_owners = {op.activated.data_ptr(): i for i, op in enumerate(body._packed_experts)}
        old = body._tp_reduce
        route_old = []
        scale_old = _ScaleCollective.launch

        def ordinary(x):
            source = x.clone()
            origin = (
                linear_recorder.latest.get(x.data_ptr()) if linear_recorder is not None else None
            )
            result = old(x)
            self.events.append(("ordinary", source, x.clone()))
            self.origins.append(origin)
            self.event_semantics.append(
                ("embedding",)
                if origin is None
                else ("linear", self.names[id(linear_recorder.linear[origin][0])], origin)
            )
            return result

        if old is not None:
            body._tp_reduce = ordinary
        for layer, op in enumerate(body._packed_experts):
            original = op._reduce_routes
            route_old.append((op, original))
            if original is None:
                continue

            def route(x, original=original, layer=layer):
                source = x.clone()
                result = original(x)
                out = x.clone()
                self.events.append(("route", source, out))
                self.routes.append((source, out))
                self.event_semantics.append(("expert_routes", layer))
                return result

            op._reduce_routes = route

        def scale(op):
            source = op._audit_input.clone()
            result = scale_old(op)
            out = op.scales.clone()
            self.events.append(("scale", source, out))
            self.event_semantics.append(
                ("scale", self.names[id(op._audit_weight)])
                if op._audit_weight is not None
                else ("expert_scale", source_owners.get(op._audit_input.data_ptr()))
            )
            if op._audit_weight is not None:
                self.weight_scales[id(op._audit_weight)] = out
            else:
                self.routed_scales[source_owners.get(op._audit_input.data_ptr())] = out
            return result

        try:
            with patch.object(_ScaleCollective, "launch", scale):
                yield
        finally:
            body._tp_reduce = old
            for op, original in route_old:
                op._reduce_routes = original

    def attach_expert_scales(self, recorder):
        if None in self.routed_scales:
            raise RuntimeError("unmapped routed FP8 scale input")
        counts = {}
        for w, _x, _y in recorder.linear:
            counts[id(w)] = counts.get(id(w), 0) + 1
        for layer, (groups, _x, ids, _packed, _activated, _routes) in enumerate(recorder.experts):
            for row, e in enumerate(ids.cpu().flatten().tolist()):
                for i, group in enumerate(groups):
                    if isinstance(group, dict) and e not in group:
                        continue
                    wid = id(group[e])
                    key = (wid, counts.get(wid, 0))
                    counts[wid] = key[1] + 1
                    if i == 2 and layer in self.routed_scales:
                        self.expert_scales[key] = self.routed_scales[layer][row : row + 1]

    def map_outputs(self, recorder):
        keys = []
        counts = {}
        for w, _x, _y in recorder.linear:
            key = (id(w), counts.get(id(w), 0))
            counts[id(w)] = key[1] + 1
            keys.append(key)
        ordinary = [event for event in self.events if event[0] == "ordinary"]
        if len(ordinary) != len(self.origins):
            raise RuntimeError("ordinary origin coverage")
        self.embedding_collective = None
        for index, ((_kind, source, out), origin) in enumerate(
            zip(ordinary, self.origins, strict=True)
        ):
            if origin is None:
                if index != 0 or self.embedding_collective is not None:
                    raise RuntimeError("unmapped collective producer")
                self.embedding_collective = (source, out)
            else:
                if not same(recorder.linear[origin][2], source):
                    raise AssertionError("collective source differs from preceding linear")
                self.global_outputs[keys[origin]] = out
        for layer, (groups, _x, ids, _packed, _activated, _routes) in enumerate(recorder.experts):
            global_routes = self.routes[layer][1].reshape(ids.numel(), -1)
            for row, e in enumerate(ids.cpu().flatten().tolist()):
                for i, group in enumerate(groups):
                    if isinstance(group, dict) and e not in group:
                        continue
                    wid = id(group[e])
                    key = (wid, counts.get(wid, 0))
                    counts[wid] = key[1] + 1
                    if i == 2:
                        self.global_outputs[key] = global_routes[row : row + 1]

    def retain_logical(self, m, cap):
        self.events = [
            (
                kind,
                x[: m * (x.shape[0] // cap)] if semantic[0] == "expert_scale" else x[:m],
                y[: m * (y.shape[0] // cap)] if semantic[0] == "expert_scale" else y[:m],
            )
            for semantic, (kind, x, y) in zip(self.event_semantics, self.events, strict=True)
        ]
        self.routes = [(x[:m], y[:m]) for x, y in self.routes]
        self.weight_scales = {k: v[:m] for k, v in self.weight_scales.items()}
        self.routed_scales = {
            i: v[: m * self.routes[i][0].shape[1]] for i, v in self.routed_scales.items()
        }

    def validate(self, group, logical_m, physical_m, *, expert_groups=None):
        if len(self.event_semantics) != len(self.events):
            raise RuntimeError("semantic collective coverage")
        schedule = [
            (semantic, kind, tuple(x.shape), str(x.dtype), tuple(y.shape), str(y.dtype))
            for semantic, (kind, x, y) in zip(self.event_semantics, self.events, strict=True)
        ]
        schedule = (logical_m, physical_m, schedule)
        self.schedule = schedule
        size = dist.get_world_size(group)
        schedules = [None] * size
        dist.all_gather_object(schedules, schedule, group=group)
        if any(s != schedule for s in schedules):
            raise RuntimeError("distributed observation schedule differs")
        for index, (kind, x, y) in enumerate(self.events):
            semantic = self.event_semantics[index][0]
            event_group = group
            if expert_groups is not None:
                if semantic in ("expert_routes", "expert_scale"):
                    event_group = expert_groups.get("etp", group)
                elif semantic == "expert_output":
                    event_group = expert_groups.get("ep", group)
            event_size = dist.get_world_size(event_group)
            physical_elements = y.numel()
            factor = (
                x.shape[0] // physical_m if self.event_semantics[index][0] == "expert_scale" else 1
            )
            x = x[: logical_m * factor]
            y = y[: logical_m * factor]
            gathered = [torch.empty_like(x) for _ in range(event_size)]
            dist.all_gather(gathered, x.contiguous(), group=event_group)
            if kind == "scale":
                maxima = torch.stack([v.cpu().double().abs().amax(-1) for v in gathered]).amax(0)
                expected = (
                    (maxima / 448)
                    .float()
                    .clamp_min(float(torch.tensor(1 / (448 * 512), dtype=torch.float32)))
                )
            else:
                expected = torch.zeros_like(x, device="cpu", dtype=torch.float32)
                for value in gathered:
                    expected = expected + value.cpu().float()
                expected = expected.to(y.dtype)
            if not same(expected, y.cpu()):
                raise AssertionError(f"collective boundary differs {index} {kind}")
            self.proofs.append(
                {
                    "index": index,
                    "kind": kind,
                    "elements": y.numel(),
                    "physical_elements": physical_elements,
                    "rank_order": list(dist.get_process_group_ranks(event_group)),
                    "storage_exact": True,
                }
            )
