"""Deferred row-sharded route completion witness; observations clone only."""

from contextlib import contextmanager
from unittest.mock import patch

import torch
import torch.distributed as dist

from cubic_serve.ops.cuda.moe.route_completion import RowPartitionedRouteCompletion

from .collective import Recorder as Base
from .collective import prepare as prepare
from .linear import same


class FoldRecorder(Base):
    def __init__(self):
        super().__init__()
        self.folds = {}
        self.fold_proofs = []

    def validate(self, group, logical_m, physical_m):
        schedule = [
            (semantic, kind, tuple(x.shape), str(x.dtype), tuple(y.shape), str(y.dtype))
            for semantic, (kind, x, y) in zip(self.event_semantics, self.events, strict=True)
        ]
        schedule = (
            logical_m,
            physical_m,
            schedule,
            [
                (i, tuple(v[2].shape), tuple(v[3].shape), str(v[4].dtype), v[5:])
                for i, v in self.folds.items()
            ],
        )
        size = dist.get_world_size(group)
        all_schedules = [None] * size
        dist.all_gather_object(all_schedules, schedule, group=group)
        if any(s != schedule for s in all_schedules):
            raise RuntimeError("row-fold observation schedule differs")
        self.fold_schedule = schedule
        for index, (
            route_index,
            part,
            _out,
            _ids,
            _weights,
            _first,
            _experts,
            _local,
        ) in self.folds.items():
            shards = [torch.empty_like(part) for _ in range(size)]
            dist.all_gather(shards, part, group=group)
            observed = torch.cat(shards, dim=0)[:physical_m]
            kind, source, _ = self.events[index]
            self.events[index] = (kind, source, observed)
            self.routes[route_index] = (source, observed)
        super().validate(group, logical_m, physical_m)
        for index, (
            route_index,
            _part,
            out,
            ids,
            weights,
            first,
            experts,
            local,
        ) in self.folds.items():
            values = self.routes[route_index][1][:logical_m].cpu()
            ids = ids[:logical_m].cpu()
            weights = weights[:logical_m].cpu()
            expected = torch.zeros_like(out[:logical_m], device="cpu")
            for route in range(ids.shape[1]):
                owned = (ids[:, route] >= first) & (ids[:, route] < first + experts)
                weighted = (values[:, route].float() * weights[:, route, None].float()).to(
                    expected.dtype
                )
                expected = expected + torch.where(owned[:, None], weighted, 0)
            if not same(expected, out[:logical_m].cpu()):
                raise AssertionError("row-local fold/gather changes stored route arithmetic")
            self.fold_proofs.append(
                dict(
                    event=index,
                    layer=route_index,
                    physical_m=physical_m,
                    logical_m=logical_m,
                    local_rows=local,
                    rank_sum_exact=True,
                    fold_and_gather_exact=True,
                )
            )


"Source-bound route primitive observer: GPU snapshots only during forward."


class Recorder(FoldRecorder):
    @contextmanager
    def observe(self, stage, linear_recorder=None):
        owners = {
            id(op.combine): i
            for i, op in enumerate(stage.body._packed_experts)
            if isinstance(op.combine, RowPartitionedRouteCompletion)
        }
        original = RowPartitionedRouteCompletion.launch

        def launch(op):
            source = op.routes.clone()
            result = original(op)
            w = op.workspace
            part = w.partition.clone()
            out = op.output.clone()
            ids = op.route_ids.clone()
            weights = op.route_weights.clone()
            index = len(self.events)
            route_index = len(self.routes)
            local = w.partition.shape[0]
            ranks = w.exchanged.shape[0]
            self.events.append(("route", source, part))
            self.routes.append((source, part))
            self.event_semantics.append(("expert_routes", owners[id(op)], "row_fold", local, ranks))
            self.folds[index] = (
                route_index,
                part,
                out,
                ids,
                weights,
                op.first_expert,
                op.experts,
                local,
            )
            return result

        with Base.observe(self, stage, linear_recorder):
            for op in stage.body._packed_experts:
                if id(op.combine) in owners:
                    op._reduce_routes = None
            with patch.object(RowPartitionedRouteCompletion, "launch", launch):
                yield
