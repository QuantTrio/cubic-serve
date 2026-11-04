"""Fixed-address integer budget operands for the existing sampling transaction."""

from __future__ import annotations

from pathlib import Path

import torch

from cubic_serve.ops.cuda.native import CudaScalar
from cubic_serve.ops.cuda.native_compiler import compile_native


class ReasoningBudgetProgram:
    """Lease-private state; the CPU matcher remains the independent rule witness.

    Marker geometry comes from the model codec before capture. No vocabulary
    mask or logits allocation is duplicated: outputs reuse the common arena.
    Bindings are constructed before capture; requests only fill pinned inputs.
    """

    def __init__(self, boundaries, mask, activity, *, stop_token_ids=()):
        if (len(boundaries) < 2 or any(not marker for marker in boundaries)
                or any(type(t) is not int or not 0 <= t < mask.shape[1] * 32
                       for marker in boundaries for t in marker)):
            raise ValueError("reasoning budget requires declared token boundaries")
        self.boundaries = tuple(tuple(m) for m in boundaries)
        self.capacity, self.words = mask.shape
        self.tail_capacity = max(map(len, self.boundaries))
        self.width = 5 + self.tail_capacity
        self.stop_token_ids = tuple(stop_token_ids)
        self.mask, self.activity = mask, activity
        self.state = torch.zeros((self.capacity, self.width), dtype=torch.int64, device=mask.device)
        self.state[:, 0] = -1
        self.host = torch.zeros_like(self.state, device="cpu", pin_memory=True)
        self.host[:, 0] = -1
        self.host_array = self.host.numpy()
        self.scratch = torch.empty_like(self.state)
        offsets = [0]
        for marker in self.boundaries:
            offsets.append(offsets[-1] + len(marker))
        offsets.append(offsets[-1] + len(self.stop_token_ids))
        self.offsets = torch.tensor(offsets, dtype=torch.int32, device=mask.device)
        self.markers = torch.tensor([*[t for m in self.boundaries for t in m],
                                      *self.stop_token_ids],
                                    dtype=torch.int32, device=mask.device)
        source = Path(__file__).with_suffix(".cu").read_bytes()
        self.module, _ = compile_native((("entry.cu", source),), mask.device,
                                        "reasoning_budget_control")
        self.qualified = False

    def qualify(self, vocabulary):
        """Witness actual codec boundaries against the independent CPU parser.

        This pre-capture check concerns integer state only. Model mathematics,
        captured sampling and the complete speculative transaction have their
        own guards and independent acceptance tests.
        """
        from cubic_serve.contracts.request import ReasoningControl
        from cubic_serve.generation.constraints import ConstraintSpec, create_logit_constraint

        ctx = torch.zeros(4, dtype=torch.int32, device=self.mask.device)
        ctx[0] = 1
        decisions = torch.zeros((1, 2), dtype=torch.int64, device=self.mask.device)
        prefix = self.prefix(ctx, decisions, owners=1, depth=0,
                             vocabulary=vocabulary, head_step=0)
        commit = self.commit(ctx, decisions, owners=1, depth=0, vocabulary=vocabulary)
        marker_tokens = set(self.markers.cpu().tolist())
        ordinary = next((t for t in range(vocabulary) if t not in marker_tokens), 0)
        pattern = (*self.boundaries[0], ordinary, *self.boundaries[1], ordinary,
                   *self.boundaries[0], ordinary, *self.boundaries[-1], ordinary)
        for active, quota in ((False, 0), (True, 0), (True, 2)):
            control = ReasoningControl(self.boundaries[0], self.boundaries[1], quota, active,
                                       implicit_ends=self.boundaries[2:])
            oracle = create_logit_constraint(spec=ConstraintSpec.identity(), request_id="budget",
                                             reasoning_control=control)
            self.bind([oracle._matcher.integer_snapshot()])
            self.stage()
            for desired in pattern:
                prefix.launch()
                actual = self.mask[0].cpu()
                mask = oracle.mask()
                logits = torch.zeros(vocabulary)
                oracle.apply(logits, mask)
                allowed = torch.isfinite(logits)
                bit_allowed = ((actual.long()[torch.arange(vocabulary) // 32]
                                >> (torch.arange(vocabulary) % 32)) & 1).bool()
                if not torch.equal(allowed, bit_allowed):
                    raise RuntimeError("reasoning budget mask failed independent qualification")
                token = desired if allowed[desired] else int(allowed.nonzero()[0])
                oracle.accept(token, mask)
                decisions[0, 1] = token
                commit.launch()
                _, phase, used, tail = oracle._matcher.integer_snapshot()
                state = self.state[0].cpu()
                if (tuple(state[:4].tolist()) != (phase, used, quota, len(tail))
                        or tuple(state[5:5 + len(tail)].tolist()) != tail):
                    raise RuntimeError("reasoning budget state failed independent qualification")
        self.host_array.fill(0)
        self.host_array[:, 0] = -1
        self.stage()
        self.mask.fill_(-1)
        self.activity.zero_()
        self.qualified = True

    def stage(self):
        self.state.copy_(self.host, non_blocking=True)

    def bind(self, snapshots, *, stop_enabled=None, terminated=None):
        if not 0 < len(snapshots) <= self.capacity:
            raise ValueError("reasoning budget exceeds the leased sampling capacity")
        self.host_array.fill(0)
        self.host_array[:, 0] = -1
        for row, snapshot in enumerate(snapshots):
            if snapshot is None:
                continue
            control, phase, used, tail = snapshot
            if ((control.start, control.end, *control.implicit_ends) != self.boundaries
                    or phase not in (0, 1, 2) or used < 0 or len(tail) > self.tail_capacity):
                raise ValueError("reasoning budget differs from the captured codec")
            self.host_array[row, :4] = (3 if terminated and terminated[row] else phase,
                                        used, min(control.token_budget, 2**63 - 1),
                                        len(tail))
            self.host_array[row, 4] = bool(stop_enabled and stop_enabled[row])
            self.host_array[row, 5:5 + len(tail)] = tail

    def prefix(self, ctx, proposals, *, owners, depth, vocabulary, head_step=-1):
        rows = (owners * (depth + 1) if head_step < 0 else owners)
        # A review capacity may have additional permanent ownerless graph rows.
        if rows > self.capacity:
            raise ValueError("reasoning review exceeds the common mask arena")
        scalar = lambda value: CudaScalar("i32", value)
        return self.module.bind(
            "reasoning_budget",
            (self.state, self.scratch, ctx, proposals, None, self.mask, self.activity,
             self.markers, self.offsets, scalar(rows), scalar(owners),
             scalar((ctx.numel() - 1) // owners), scalar(depth), scalar(head_step),
             scalar(self.width), scalar(len(self.boundaries)), scalar(self.words),
             scalar(vocabulary), scalar(0)),
            operator_id="sampling.reasoning_budget.prefix", grid=(rows, 1, 1),
            block=(128, 1, 1),
        )

    def commit(self, ctx, decisions, *, owners, depth, vocabulary):
        scalar = lambda value: CudaScalar("i32", value)
        return self.module.bind(
            "reasoning_budget", (self.state, self.scratch, ctx, None, decisions,
                                 self.mask, self.activity, self.markers, self.offsets,
                                 scalar(owners), scalar(owners),
                                 scalar((ctx.numel() - 1) // owners), scalar(depth), scalar(0),
                                 scalar(self.width), scalar(len(self.boundaries)),
                                 scalar(self.words), scalar(vocabulary), scalar(1)),
            operator_id="sampling.reasoning_budget.commit", grid=(owners, 1, 1),
            block=(128, 1, 1),
        )
