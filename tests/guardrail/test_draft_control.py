"""Actual CUDA binding checks; synthetic math grants no model qualification."""
import ctypes as ct
from types import SimpleNamespace as S

import pytest
import torch
from cuda.bindings import runtime as rt

from cubic_serve.executor.resident_window import append_captured_graph, check
from cubic_serve.guardrail.draft_control import prove_window_bodies
from cubic_serve.guardrail.graph_observer import graph_nodes
from cubic_serve.ops.cuda.native import CudaScalar
from cubic_serve.ops.cuda.native_compiler import compile_native
from cubic_serve.ops.cuda.launch_cost import _driver, _check, _KernelNodeParams

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('split', [False, True])
def test_conditional_model_origin_controller_arguments_and_phase_dependencies(split):
    device = torch.device('cuda:0')
    value = torch.zeros(8, device=device)
    counter = torch.zeros(7, dtype=torch.int32, device=device)
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    parent = check(rt.cudaGraphCreate(0))[0]
    handle = check(rt.cudaGraphConditionalHandleCreate(parent, 1, 1))[0]
    source = b'''#include <cuda_runtime.h>
    extern "C" __global__ void phase(int* counts,int phase,unsigned long long handle){
      if(!threadIdx.x){counts[phase]++;if(phase==5)cudaGraphSetConditional((cudaGraphConditionalHandle)handle,0);}}
    '''
    module, entry = compile_native((('entry.cu', source),), device, 'audit_control_binding_test')
    calls = tuple(module.bind(entry, (counter, CudaScalar('i32', i), CudaScalar('i64', int(handle))),
                  operator_id='audit.test.control', grid=(1, 1, 1), block=(32, 1, 1))
                  for i in range(7))
    target, review, ordinary, ingress, body, forward = (
        torch.cuda.CUDAGraph(keep_graph=True) for _ in range(6))
    executable = None
    try:
        with torch.cuda.graph(target, stream=stream):value.add_(1)
        with torch.cuda.graph(review, stream=stream):value.mul_(2)
        with torch.cuda.graph(forward, stream=stream):append_captured_graph(target)
        with torch.cuda.graph(ordinary, stream=stream):
            if not split:append_captured_graph(target)
            append_captured_graph(review)
        with torch.cuda.graph(ingress, stream=stream):calls[6].launch()
        with torch.cuda.graph(body, stream=stream):
            for call in calls[:3]:call.launch()
            append_captured_graph(target)
            append_captured_graph(review)
            for call in calls[3:6]:call.launch()
        stream.synchronize()
        begin = check(rt.cudaGraphAddChildGraphNode(parent, [], 0, ingress.raw_cuda_graph()))[0]
        params = rt.cudaGraphNodeParams()
        params.type = rt.cudaGraphNodeType.cudaGraphNodeTypeConditional
        params.conditional.handle = handle
        params.conditional.type = rt.cudaGraphConditionalNodeType.cudaGraphCondTypeWhile
        params.conditional.size = 1
        check(rt.cudaGraphAddNode(parent, [begin], None, 1, params))
        check(rt.cudaGraphAddChildGraphNode(params.conditional.phGraph_out[0], [], 0,
                                           body.raw_cuda_graph()))
        executable = check(rt.cudaGraphInstantiateWithFlags(parent, 0))[0]
        round_ = S(_retain_audit_graph=True, _review_graphs={(False,) * 6: review},
                   target=S(body=S(captured_body_graph=lambda: target)), graph=ordinary,
                   ordinary_phases=lambda *a, **kw: (forward, ordinary) if split else (ordinary,),
                   M=2, C=1, depth=1)
        window = S(round=round_, exec=executable, inflight=False, children=(ingress, body),
                   calls=calls, graph=parent, requires_greedy=True, requires_budget=False)
        proof = prove_window_bodies(round_, window)
        assert not proof['numerical_state_oracle']
        assert not proof['surrounding_transport_control_qualified']
        assert proof['native_controller_phases'] == tuple(range(7))
        assert proof['parents']['ordinary']['order']['dependency_path']
        assert proof['parents']['ordinary']['sealed_phase_count'] == (2 if split else 1)
        assert proof['conditional_body']['nodes'] >= 8
        assert proof['seconds'] >= 0
        # Mutating the source body after parent embedding must invalidate the
        # origin/argument proof, rather than just matching its kernel function.
        node = graph_nodes(target.raw_cuda_graph())[0]
        driver = _driver()
        driver.cuGraphKernelNodeSetParams_v2.argtypes = [ct.c_void_p, ct.POINTER(_KernelNodeParams)]
        params = _KernelNodeParams()
        _check(driver.cuGraphKernelNodeGetParams_v2(int(node), ct.byref(params)))
        params.gx += 1
        _check(driver.cuGraphKernelNodeSetParams_v2(int(node), ct.byref(params)))
        with pytest.raises(RuntimeError, match='parameters differ'):
            prove_window_bodies(round_, window)
    finally:
        if executable is not None:check(rt.cudaGraphExecDestroy(executable))
        check(rt.cudaGraphDestroy(parent))
        for graph in (body, ingress, ordinary, forward, review, target):graph.reset()


def test_actual_proposal_copy_observer_snapshots_and_failure_restoration():
    from cubic_serve.executor.draft_graph import pending_copy
    from cubic_serve.guardrail.draft_control import DraftRoundLogits
    from cubic_serve.guardrail.graph_observer import graph_identity

    device = torch.device('cuda:0')
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    depth, owners, candidates, context_stride = 3, 1, 2, 6
    context = torch.zeros(1 + context_stride, dtype=torch.int32, device=device)
    context[0] = owners
    scores = torch.ones((1, owners, candidates), device=device)
    ids = torch.arange(candidates, device=device).reshape(1, owners, candidates)
    token = torch.zeros(owners, dtype=torch.int64, device=device)
    pool_scores = torch.zeros((depth, 1, owners, candidates), device=device)
    pool_ids = torch.zeros_like(pool_scores, dtype=torch.int64)
    pool_tokens = torch.zeros((depth, owners), dtype=torch.int64, device=device)
    p, q = torch.zeros((4, 11), device=device), torch.zeros((1, 11), device=device)

    def copy(index):
        pending_copy[(owners,)](
            context, scores, ids, token, pool_scores[index], pool_ids[index],
            pool_tokens[index], context_stride, owners, owners, 1, candidates, 32, False)

    with torch.cuda.stream(stream):
        copy(0)
    stream.synchronize()
    review, ordinary = (torch.cuda.CUDAGraph(keep_graph=True) for _ in range(2))
    observer = None
    try:
        with torch.cuda.graph(review, stream=stream):
            for index in range(depth):
                q.fill_(10 + index)
                token.fill_(100 + index)
                copy(index)
        with torch.cuda.graph(ordinary, stream=stream):
            p.add_(2)
            append_captured_graph(review)
        stream.synchronize()
        round_ = S(_retain_audit_graph=True, graph=ordinary,
                   _review_graphs={(False,) * 6: review}, depth=depth,
                   target=S(slot=S(stream=stream), body=S(hidden=p, logits=p)),
                   head=S(body=S(logits=q), _token_ids=token,
                          _sampling=S(workspace=S(gathered_scores=scores,
                                                   gathered_token_ids=ids))),
                   ctx=context, pool_scores=pool_scores, pool_ids=pool_ids,
                   pool_tokens=pool_tokens)
        before = graph_identity(ordinary.raw_cuda_graph())
        with pytest.raises(MemoryError, match='budget'):
            DraftRoundLogits(round_, forward_capacity=1, byte_budget=1)
        observer = DraftRoundLogits(round_, forward_capacity=1)
        observer.before_forward('ordinary')
        with torch.cuda.stream(stream):
            ordinary.replay()
        stream.synchronize()
        observer.after_forward('ordinary')
        event = observer.events[0]
        assert torch.equal(event['p'], torch.full_like(p, 2, device='cpu'))
        for index in range(depth):
            assert torch.equal(event['q' + str(index)], torch.full_like(q, 10 + index, device='cpu'))
        with pytest.raises(MemoryError, match='count'):
            observer.before_forward('ordinary')
        event['q1'].add_(1)
        with pytest.raises(RuntimeError, match='immutable'):
            _ = observer.events
        observer.close()
        observer.close()
        assert 'replay' not in ordinary.__dict__
        assert graph_identity(ordinary.raw_cuda_graph()) == before
        with torch.cuda.stream(stream):
            ordinary.replay()
        stream.synchronize()
        assert torch.equal(p, torch.full_like(p, 4))
        assert pool_tokens[:, 0].tolist() == [100, 101, 102]

        round_.pool_tokens = torch.empty_like(pool_tokens)
        with pytest.raises(RuntimeError, match='binding'):
            DraftRoundLogits(round_, forward_capacity=1)
        assert graph_identity(ordinary.raw_cuda_graph()) == before
        round_.pool_tokens = pool_tokens
        observer = DraftRoundLogits(round_, forward_capacity=1)
        round_.pool_tokens = torch.empty_like(pool_tokens)
        with pytest.raises(RuntimeError, match='rebound'):
            observer.before_forward('ordinary')
        observer.close()
        assert graph_identity(ordinary.raw_cuda_graph()) == before
    finally:
        if observer is not None:
            observer.close()
        ordinary.reset()
        review.reset()
