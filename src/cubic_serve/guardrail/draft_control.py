"""Offline binding and observations of sealed numerical body graphs.

These prerequisites grant no qualification and replace no independent oracle.
Observations use private executables and preserve the original executable.
"""
from __future__ import annotations

import time
import hashlib
import struct

import torch
from cuda.bindings import runtime as rt

from cubic_serve.executor.resident_window import check
from cubic_serve.guardrail.graph_observer import (
    GraphObserver, SnapshotPoint, graph_nodes, kernel_identity,
    prove_child_graph_clones, prove_graph_precedes,
)


def _bound_native(call):
    return (int(call.function), *call.grid, *call.block, call.shared_bytes,
            tuple(call._argument_values))


def _actual_native(node):
    actual = kernel_identity(node)
    return (*actual[:-1], tuple(value for _, value in actual[-1]))


def _compact_copy(copy):
    def count(proof):
        return len(proof['nodes']) + sum(count(child) for child in proof['nested'])
    proof = copy['clone']
    return dict(node=copy['node'], graph=copy['graph'], nodes=count(proof),
                identity_digest=hashlib.blake2s(repr(proof).encode()).hexdigest())


def prove_window_bodies(round_, window):
    """Bind both parent compositions to the same target/review child DAGs.

    Direct origin links are proved individually through the conditional body's
    embedded copy. Every node, argument byte, launch geometry and dependency
    inside the shared numerical bodies is checked, including nested providers.
    Resident controller phases must use their exact immutable bound operands.
    Surrounding transport, other control nodes, leases and actual data/state
    provenance remain separate obligations and are explicitly not certified.
    """
    started = time.perf_counter()
    if window.round is not round_ or window.exec is None or window.inflight:
        raise ValueError("resident body proof requires its idle live owner")
    if not getattr(round_, '_retain_audit_graph', False):
        raise ValueError("resident body proof requires explicit offline graph retention")
    if len(window.children) != 2 or len(window.calls) != 7:
        raise ValueError("resident body proof inventory differs")
    candidates = not window.requires_greedy
    budget = window.requires_budget
    key = (candidates, budget, False, False, False, False)
    review = round_._review_graphs.get(key)
    if review is None:
        raise ValueError("resident body proof lacks its sealed review policy")
    ordinary = getattr(round_, ('budget_' if budget else '') +
                       ('candidate_graph' if candidates else 'graph'))
    target = round_.target.body.captured_body_graph()
    ingress, body = window.children
    parents = {}
    # This raw, uninstantiated witness has exactly the order returned to the
    # ordinary replay loop. It allocates no second executable. Numerical state,
    # host ingress and the exclusive stream lease retain their independent proofs.
    from cubic_serve.executor.resident_window import append_captured_graph
    phases = round_.ordinary_phases(ordinary, budget=budget)
    composed = torch.cuda.CUDAGraph(keep_graph=True)
    try:
        with torch.cuda.graph(composed):
            for phase in phases:
                append_captured_graph(phase)
        for name, graph in (('ordinary', composed), ('resident_body', body)):
            raw = graph.raw_cuda_graph()
            if name == 'ordinary' and len(phases) == 2:
                target_parent, review_parent = (phase.raw_cuda_graph() for phase in phases)
                # Numerical children are proved once at their real source phase.
                # The witness's two outer child nodes bind the same phase order.
                phase_nodes = []
                for phase in phases:
                    original = graph_nodes(phase.raw_cuda_graph())[0]
                    matches = [node for node in graph_nodes(raw) if
                        check(rt.cudaGraphNodeGetType(node))[0] == rt.cudaGraphNodeType.cudaGraphNodeTypeGraph
                        and int(rt.cudaGraphNodeFindInClone(original,
                            check(rt.cudaGraphChildGraphNodeGetGraph(node))[0])[0]) == 0]
                    if len(matches) != 1:
                        raise RuntimeError('ordinary phase origin inventory differs')
                    phase_nodes.append(matches[0])
                order = prove_graph_precedes(raw, *phase_nodes)
            else:
                target_parent = review_parent = phases[0].raw_cuda_graph() if name == 'ordinary' else raw
                order = None
            target_copy = _compact_copy(prove_child_graph_clones(target.raw_cuda_graph(), target_parent)[0])
            review_copy = _compact_copy(prove_child_graph_clones(review.raw_cuda_graph(), review_parent)[0])
            parents[name] = dict(target=target_copy, review=review_copy,
                order=order or prove_graph_precedes(target_parent, target_copy['node'], review_copy['node']))
        parents['ordinary']['sealed_phase_count'] = len(phases)
    finally:
        composed.reset()
    ingress_copy = _compact_copy(prove_child_graph_clones(ingress.raw_cuda_graph(), window.graph)[0])
    conditional = [node for node in graph_nodes(window.graph) if
        check(rt.cudaGraphNodeGetType(node))[0] == rt.cudaGraphNodeType.cudaGraphNodeTypeConditional]
    if len(conditional) != 1:
        raise RuntimeError("resident parent conditional inventory differs")
    node_params = check(rt.cudaGraphNodeGetParams(conditional[0]))[0]
    params = node_params.conditional
    if params.size != 1 or params.type != rt.cudaGraphConditionalNodeType.cudaGraphCondTypeWhile:
        raise RuntimeError("resident parent conditional rule differs")
    embedded = _compact_copy(prove_child_graph_clones(
        body.raw_cuda_graph(), params.phGraph_out[0])[0])
    order = prove_graph_precedes(window.graph, ingress_copy['node'], conditional[0])
    expected = tuple(_bound_native(call) for call in window.calls)
    for raw, phases in ((ingress.raw_cuda_graph(), (6,)), (body.raw_cuda_graph(), tuple(range(6)))):
        actual = [_actual_native(node) for node in graph_nodes(raw) if
            check(rt.cudaGraphNodeGetType(node))[0] == rt.cudaGraphNodeType.cudaGraphNodeTypeKernel]
        for phase in phases:
            if actual.count(expected[phase]) != 1:
                raise RuntimeError("resident controller phase binding differs")
    return dict(scope='sealed numerical-body capture identity prerequisite only',
        numerical_state_oracle=False, surrounding_transport_control_qualified=False,
        target_capacity=round_.M, provider_capacity=round_.C, depth=round_.depth,
        candidates=candidates, budget=budget, parents=parents, ingress=ingress_copy,
        conditional_body=embedded, ingress_before_loop=order,
        native_controller_phases=tuple(range(7)), seconds=time.perf_counter() - started)


class DraftRoundLogits:
    """Observe actual ordinary-round p/q values; supply no oracle or verdict.

    The caller owns pre-readiness exclusivity and teardown on every outcome.
    Only a private executable observes the borrowed raw DAG. Its original
    executable remains intact. Resident forwards are deliberately unobserved;
    their existing independent composition checks remain the caller's duty.
    CPU snapshots and device observation storage have separate equal bounds.
    """

    def __init__(self, round_, *, forward_capacity, candidates=False, byte_budget=64 << 20):
        if not getattr(round_, '_retain_audit_graph', False):
            raise ValueError('round observations require offline graph retention')
        if type(forward_capacity) is not int or forward_capacity <= 0:
            raise ValueError('round observation count must be bounded')
        if type(candidates) is not bool:
            raise ValueError('round sampling policy must be explicit')
        self.round = round_
        self.graph = round_.candidate_graph if candidates else round_.graph
        if 'replay' in self.graph.__dict__:
            raise ValueError('round graph already has a replay borrower')
        self.stream = round_.target.slot.stream
        self.stream.synchronize()
        self.private, self.observer = None, None
        self.forward_capacity, self._events = forward_capacity, []
        self._closed = False
        review = round_._review_graphs[(candidates, False, False, False, False, False)]
        copied = prove_child_graph_clones(review.raw_cuda_graph(), self.graph.raw_cuda_graph())[0]
        points = [SnapshotPoint('p', (('logits', round_.target.body.logits),), 1)]
        points.extend(SnapshotPoint('q' + str(d), (('logits', round_.head.body.logits),), 1)
                      for d in range(round_.depth))
        width = sum(t.numel() * t.element_size() for p in points for _, t in p.tensors)
        if type(byte_budget) is not int or width * (forward_capacity + 2) > byte_budget:
            raise MemoryError('round CPU snapshot budget is insufficient')
        self.observer = GraphObserver(points, byte_budget=byte_budget,
                                      device=round_.target.body.hidden.device)
        self.bindings = self._bindings()
        try:
            self.observer.attach(self.graph.raw_cuda_graph(), rt.cudaGraphNode_t(copied['node']), 'p')
            raw = rt.cudaGraph_t(copied['graph'])
            programs = []
            for depth in range(round_.depth):
                pointers = {struct.pack('=Q', t.data_ptr()) for t in self._operands(depth)}
                matches = []
                for node in graph_nodes(raw):
                    if check(rt.cudaGraphNodeGetType(node))[0] != rt.cudaGraphNodeType.cudaGraphNodeTypeKernel:
                        continue
                    identity = kernel_identity(node)
                    if pointers.issubset({value for _, value in identity[-1]}):
                        matches.append((node, identity))
                if len(matches) != 1:
                    raise RuntimeError('carried proposal copy binding is unavailable')
                node, identity = matches[0]
                # Alignment can select distinct programs; bind each actual one.
                programs.append(identity)
                self.observer.attach(raw, node, 'q' + str(depth))
            self.proposal_bindings = tuple(programs)
            self.private = check(rt.cudaGraphInstantiateWithFlags(self.graph.raw_cuda_graph(), 0))[0]
            self.graph.replay = lambda: check(rt.cudaGraphLaunch(
                self.private, torch.cuda.current_stream(round_.target.body.hidden.device).cuda_stream))
        except BaseException:
            self.close()
            raise

    def _operands(self, depth):
        r, w = self.round, self.round.head._sampling.workspace
        return (r.ctx, w.gathered_scores, w.gathered_token_ids, r.head._token_ids,
                r.pool_scores[depth], r.pool_ids[depth], r.pool_tokens[depth])

    def _bindings(self):
        tensors = (self.round.target.body.logits, self.round.head.body.logits,
                   *(t for d in range(self.round.depth) for t in self._operands(d)))
        return tuple(GraphObserver._binding(t) for t in tensors)

    def before_forward(self, kind):
        if self._closed:
            raise RuntimeError('round observation is closed')
        if kind == 'resident':
            return
        if kind != 'ordinary':
            raise ValueError('unknown round observation kind')
        if len(self._events) >= self.forward_capacity:
            raise MemoryError('round CPU snapshot count is exhausted')
        if self._bindings() != self.bindings:
            raise RuntimeError('round observation operands were rebound')
        self.observer.begin()

    def after_forward(self, kind):
        if kind == 'resident':
            return
        if kind != 'ordinary':
            raise ValueError('unknown round observation kind')
        def gather(value):
            if not torch.distributed.is_initialized():
                return [value]
            values = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(values, value)
            return values
        events = self.observer.finish(gather_schedules=gather)
        if tuple(name for name, _ in events) != tuple(p.name for p in self.observer.points):
            raise RuntimeError('actual p/q event order differs')
        with torch.inference_mode(False), torch.no_grad():
            values = {name: fields[0][1].view(point.tensors[0][1].dtype)
                      .reshape(point.tensors[0][1].shape).clone()
                      for (name, fields), point in zip(events, self.observer.points, strict=True)}
        self._events.append((values, tuple(t._version for t in values.values())))

    @property
    def events(self):
        from types import MappingProxyType
        for values, versions in self._events:
            if tuple(t._version for t in values.values()) != versions:
                raise RuntimeError('immutable round observation was modified')
        return tuple(MappingProxyType(values) for values, _ in self._events)

    def close(self):
        if self._closed:
            return
        self.stream.synchronize()
        if 'replay' in self.graph.__dict__ and self.private is not None:
            del self.graph.replay
        if self.observer is not None and self.observer.active:
            self.observer.discard(stream=self.stream)
        if self.private is not None:
            check(rt.cudaGraphExecDestroy(self.private))
            self.private = None
        if self.observer is not None:
            self.observer.detach()
        self._closed = True
