"""CUDA DAG transaction tests; these are not model numerical qualification."""
import ctypes as ct

import pytest
import torch
from cuda.bindings import runtime as rt

from cubic_serve.executor.resident_window import check
from cubic_serve.guardrail.graph_observer import (
    GraphObserver, SnapshotPoint, graph_identity, graph_nodes, prove_graph_clone,
    prove_child_graph_clones, prove_graph_precedes,
)
from cubic_serve.ops.cuda.launch_cost import _driver, _check, _KernelNodeParams

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_direct_nested_clone_origin_arguments_edges_and_extra_nodes():
    value = torch.zeros(8, device='cuda:0')
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    leaf = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(leaf, stream=stream):
        value.add_(1)
        value.mul_(2)
    stream.synchronize()
    parent = check(rt.cudaGraphCreate(0))[0]
    clone = None
    try:
        check(rt.cudaGraphAddChildGraphNode(parent, [], 0, leaf.raw_cuda_graph()))
        clone = check(rt.cudaGraphClone(parent))[0]
        proof = prove_graph_clone(parent, clone)
        assert len(proof['nodes']) == 1 and len(proof['nested']) == 1
        assert len(proof['nested'][0]['nodes']) == 2
        child_node = graph_nodes(clone)[0]
        child = check(rt.cudaGraphChildGraphNodeGetGraph(child_node))[0]
        node = graph_nodes(child)[0]
        params = _KernelNodeParams()
        driver = _driver()
        driver.cuGraphKernelNodeSetParams_v2.argtypes = [ct.c_void_p, ct.POINTER(_KernelNodeParams)]
        _check(driver.cuGraphKernelNodeGetParams_v2(int(node), ct.byref(params)))
        params.gx += 1
        _check(driver.cuGraphKernelNodeSetParams_v2(int(node), ct.byref(params)))
        with pytest.raises(RuntimeError, match='parameters differ'):
            prove_graph_clone(parent, clone)
        params.gx -= 1
        _check(driver.cuGraphKernelNodeSetParams_v2(int(node), ct.byref(params)))
        prove_graph_clone(parent, clone)
        check(rt.cudaGraphAddEmptyNode(child, [], 0))
        with pytest.raises(RuntimeError, match='inventory differs'):
            prove_graph_clone(parent, clone)
    finally:
        if clone is not None:
            check(rt.cudaGraphDestroy(clone))
        check(rt.cudaGraphDestroy(parent))
        leaf.reset()


def fixture(visits=2, budget=1 << 20):
    device = torch.device("cuda:0")
    base = torch.arange(48, dtype=torch.int64, device=device).reshape(4, 12)
    source = base[:, ::3]
    output = torch.zeros_like(source)
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph, stream=stream):
        output.copy_(source)
        output.add_(1)
    stream.synchronize()
    observer = GraphObserver((SnapshotPoint("input", (("x", source),), visits),),
                             byte_budget=budget, device=device)
    raw = graph.raw_cuda_graph()
    edges = graph_identity(raw)[1]
    roots = set(int(n) for n in graph_nodes(raw)) - {b for _, b, *_ in edges}
    assert len(roots) == 1
    node = rt.cudaGraphNode_t(roots.pop())
    return base, source, output, stream, graph, raw, node, observer


def execute(graph, stream):
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        graph.replay()
    stream.synchronize()


def test_readonly_strided_bytes_and_exact_dag_restoration():
    base, source, output, stream, graph, raw, node, observer = fixture()
    before = graph_identity(raw)
    observer.attach(raw, node, "input")
    graph.instantiate()
    observer.begin()
    first = source.cpu().clone()
    execute(graph, stream)
    base.add_(100)
    second = source.cpu().clone()
    execute(graph, stream)
    base.add_(1000)  # retained snapshots may not alias their producing inputs.
    events = observer.finish(gather_schedules=lambda schedule: (schedule,))
    assert len(events) == 2
    for event, expected in zip(events, (first, second), strict=True):
        assert event[0] == "input"
        assert torch.equal(event[1][0][1].view(source.dtype).reshape(source.shape), expected)
    assert torch.equal(output.cpu(), second + 1)
    observer.detach()
    assert graph_identity(raw) == before
    graph.instantiate()
    execute(graph, stream)
    assert torch.equal(output, source + 1)


def test_overflow_is_a_harness_failure_and_detaches():
    *_, stream, graph, raw, node, observer = fixture(visits=1)
    before = graph_identity(raw)
    observer.attach(raw, node, "input")
    graph.instantiate()
    observer.begin()
    execute(graph, stream)
    execute(graph, stream)
    gathered = []
    with pytest.raises(RuntimeError, match="overflow"):
        observer.finish(gather_schedules=lambda s: gathered.append(s) or (s,))
    assert gathered[0][0] is True
    observer.detach()
    assert graph_identity(raw) == before
    graph.instantiate()


def test_rank_schedule_mismatch_precedes_payload_reads(monkeypatch):
    *_, stream, graph, raw, node, observer = fixture()
    observer.attach(raw, node, "input")
    graph.instantiate()
    observer.begin()
    execute(graph, stream)
    # The only CPU reads before consensus are count, overflow and event headers.
    output = observer.records[0][1]
    original = torch.Tensor.cpu
    def cpu(tensor, *args, **kwargs):
        assert tensor.untyped_storage().data_ptr() != output.untyped_storage().data_ptr()
        return original(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "cpu", cpu)
    with pytest.raises(RuntimeError, match="schedule mismatch"):
        observer.finish(gather_schedules=lambda s: (s, (False, ((999, 0),))))
    observer.detach()
    graph.instantiate()


def test_budget_rejection_precedes_native_compilation(monkeypatch):
    from cubic_serve.guardrail import graph_observer as module
    def forbidden(*args, **kwargs):
        raise AssertionError("budget must be checked before compilation")
    monkeypatch.setattr(module, "compile_native", forbidden)
    x = torch.zeros(128, device="cuda:0")
    with pytest.raises(MemoryError, match="budget"):
        GraphObserver((SnapshotPoint("x", (("x", x),), 3),),
                      byte_budget=32, device="cuda:0")


def test_mutated_original_kernel_is_rejected():
    *_, stream, graph, raw, node, observer = fixture()
    observer.attach(raw, node, "input")
    driver = _driver()
    driver.cuGraphKernelNodeSetParams_v2.argtypes = [ct.c_void_p, ct.POINTER(_KernelNodeParams)]
    params = _KernelNodeParams()
    _check(driver.cuGraphKernelNodeGetParams_v2(int(node), ct.byref(params)))
    # Retain the actual argument bytes independently before mutation invalidates
    # CUDA's node-owned argument array.
    from cubic_serve.guardrail.graph_observer import kernel_identity
    identity = kernel_identity(node)
    values = [ct.create_string_buffer(value, len(value)) for _, value in identity[-1]]
    pointers = (ct.c_void_p * len(values))(*(ct.addressof(v) for v in values))
    params.parameters = ct.addressof(pointers)
    original_grid = params.gx
    params.gx += 1
    _check(driver.cuGraphKernelNodeSetParams_v2(int(node), ct.byref(params)))
    with pytest.raises(RuntimeError, match="original graph"):
        observer.detach()
    params.gx = original_grid
    params.parameters = ct.addressof(pointers)
    _check(driver.cuGraphKernelNodeSetParams_v2(int(node), ct.byref(params)))
    graph.instantiate()


def test_active_transaction_cannot_detach():
    *_, stream, graph, raw, node, observer = fixture()
    observer.attach(raw, node, "input")
    graph.instantiate()
    observer.begin()
    with pytest.raises(RuntimeError, match="active"):
        observer.detach()
    execute(graph, stream)
    observer.finish(gather_schedules=lambda s: (s,))
    observer.detach()
    graph.instantiate()


def test_source_storage_rebinding_is_rejected_before_replay():
    _, source, *_, observer = fixture()
    original = source.data
    source.data = torch.zeros_like(source)
    with pytest.raises(RuntimeError, match="storage binding"):
        observer.begin()
    source.data = original


def test_memcpy_and_event_graph_identity_restores():
    device = torch.device('cuda:0')
    host = torch.arange(16, pin_memory=True)
    target = torch.zeros(16, dtype=host.dtype, device=device)
    stream = torch.cuda.Stream(device=device)
    event = torch.cuda.Event()
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph, stream=stream):
        target.copy_(host, non_blocking=True)
        event.record()
        target.add_(1)
    stream.synchronize()
    raw = graph.raw_cuda_graph()
    before = graph_identity(raw)
    edges = before[1]
    leaves = set(int(n) for n in graph_nodes(raw)) - {a for a, *_ in edges}
    assert len(leaves) == 1
    observer = GraphObserver((SnapshotPoint('copied', (('x', target),), 1),),
                             byte_budget=4096, device=device)
    observer.attach(raw, rt.cudaGraphNode_t(leaves.pop()), 'copied')
    graph.instantiate()
    observer.begin()
    execute(graph, stream)
    events = observer.finish(gather_schedules=lambda s: (s,))
    assert torch.equal(events[0][1][0][1].view(host.dtype), host)
    observer.detach()
    assert graph_identity(raw) == before
    graph.instantiate()


def test_nested_conditional_graph_observes_every_iteration_and_restores():
    from cubic_serve.ops.cuda.native import CudaScalar
    from cubic_serve.ops.cuda.native_compiler import compile_native
    source = b'''#include <cuda_runtime.h>
    #include <stdint.h>
    extern "C" __global__ void step(int64_t* state,int* count,uint64_t handle){
      if(!threadIdx.x){int n=count[0];state[0]+=n+1;count[0]=n+1;
      cudaGraphSetConditional((cudaGraphConditionalHandle)handle,n+1<3);}}
    '''
    device = torch.device('cuda:0')
    state = torch.zeros(1, dtype=torch.int64, device=device)
    count = torch.zeros(1, dtype=torch.int32, device=device)
    parent = check(rt.cudaGraphCreate(0))[0]
    handle = check(rt.cudaGraphConditionalHandleCreate(parent, 1, 1))[0]
    module, entry = compile_native((('entry.cu', source),), device, 'test_observer_conditional')
    call = module.bind(entry, (state, count, CudaScalar('i64', int(handle))),
                       operator_id='test.observer_step', grid=(1, 1, 1), block=(32, 1, 1))
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream())
    body = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(body, stream=stream):
        call.launch()
    stream.synchronize()
    params = rt.cudaGraphNodeParams()
    params.type = rt.cudaGraphNodeType.cudaGraphNodeTypeConditional
    params.conditional.handle = handle
    params.conditional.type = rt.cudaGraphConditionalNodeType.cudaGraphCondTypeWhile
    params.conditional.size = 1
    check(rt.cudaGraphAddNode(parent, [], None, 0, params))
    node = check(rt.cudaGraphAddChildGraphNode(params.conditional.phGraph_out[0], [], 0,
                                             body.raw_cuda_graph()))[0]
    embedded = check(rt.cudaGraphChildGraphNodeGetGraph(node))[0]
    before = graph_identity(embedded)
    observer = GraphObserver((SnapshotPoint('step', (('state', state), ('count', count)), 3),),
                             byte_budget=4096, device=device)
    executable = None
    try:
        observer.attach(embedded, graph_nodes(embedded)[0], 'step')
        executable = check(rt.cudaGraphInstantiateWithFlags(parent, 0))[0]
        observer.begin()
        stream.wait_stream(torch.cuda.current_stream())
        check(rt.cudaGraphLaunch(executable, stream.cuda_stream))
        stream.synchronize()
        events = observer.finish(gather_schedules=lambda s: (s,))
        actual = [(int(fields[0][1].view(torch.int64).item()),
                   int(fields[1][1].view(torch.int32).item())) for _, fields in events]
        assert actual == [(0, 0), (1, 1), (3, 2)]
        assert state.item() == 6 and count.item() == 3
        check(rt.cudaGraphExecDestroy(executable))
        executable = None
        observer.detach()
        assert graph_identity(embedded) == before
        executable = check(rt.cudaGraphInstantiateWithFlags(parent, 0))[0]
        state.zero_()
        count.zero_()
        stream.wait_stream(torch.cuda.current_stream())
        check(rt.cudaGraphLaunch(executable, stream.cuda_stream))
        stream.synchronize()
        assert state.item() == 6 and count.item() == 3
    finally:
        if executable is not None:
            check(rt.cudaGraphExecDestroy(executable))
        check(rt.cudaGraphDestroy(parent))
        body.reset()


@pytest.mark.parametrize('dtype', (torch.uint8, torch.bfloat16, torch.int64))
def test_all_storage_bits_with_broadcast_and_noncontiguous_inputs(dtype):
    device = torch.device('cuda:0')
    bits = torch.arange(256, dtype=torch.uint8, device=device)
    source = bits.view(dtype)[::2].unsqueeze(0).expand(3, -1)
    target = torch.empty_like(source)
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph, stream=stream):
        target.copy_(source)
    stream.synchronize()
    observer = GraphObserver((SnapshotPoint('bits', (('x', source),), 1),),
                             byte_budget=4096, device=device)
    raw = graph.raw_cuda_graph()
    observer.attach(raw, graph_nodes(raw)[0], 'bits')
    graph.instantiate()
    observer.begin()
    execute(graph, stream)
    events = observer.finish(gather_schedules=lambda s: (s,))
    assert torch.equal(events[0][1][0][1], source.contiguous().view(torch.uint8).reshape(-1).cpu())
    observer.detach()
    graph.instantiate()


def test_packed_launch_arguments_match_explicit_values_and_detect_pointer_change():
    from cubic_serve.ops.cuda.native_compiler import compile_native
    from cubic_serve.guardrail.graph_observer import kernel_identity
    source = b'''#include <stdint.h>
    extern "C" __global__ void copy(const int64_t* src,int64_t* dst){
    if(!threadIdx.x)dst[0]=src[0];}'''
    device = torch.device('cuda:0')
    x = torch.tensor([17], dtype=torch.int64, device=device)
    y = torch.zeros_like(x)
    module, entry = compile_native((('entry.cu', source),), device, 'test_observer_packed')
    call = module.bind(entry, (x, y), operator_id='test.packed_copy',
                       grid=(1, 1, 1), block=(32, 1, 1))
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph, stream=stream):
        call.launch()
    stream.synchronize()
    node = graph_nodes(graph.raw_cuda_graph())[0]
    original = kernel_identity(node)
    driver = _driver()
    driver.cuGraphKernelNodeSetParams_v2.argtypes = [ct.c_void_p, ct.POINTER(_KernelNodeParams)]
    params = _KernelNodeParams()
    _check(driver.cuGraphKernelNodeGetParams_v2(int(node), ct.byref(params)))
    size = ct.c_size_t(max(offset + len(value) for offset, value in original[-1]))
    packed = ct.create_string_buffer(size.value)
    for offset, value in original[-1]:
        ct.memmove(ct.addressof(packed) + offset, value, len(value))
    extra = (ct.c_void_p * 5)(1, ct.addressof(packed), 2, ct.addressof(size), 0)
    params.parameters = None
    params.extra = ct.addressof(extra)
    _check(driver.cuGraphKernelNodeSetParams_v2(int(node), ct.byref(params)))
    packed_identity = kernel_identity(node)
    assert packed_identity[:-1] == original[:-1]
    assert packed_identity[-1] == ((-1, packed.raw),)
    graph.instantiate()
    execute(graph, stream)
    assert y.item() == 17
    alternate = torch.tensor([29], dtype=x.dtype, device=device)
    pointer = ct.c_void_p(alternate.data_ptr())
    ct.memmove(ct.addressof(packed) + original[-1][0][0], ct.byref(pointer), ct.sizeof(pointer))
    _check(driver.cuGraphKernelNodeSetParams_v2(int(node), ct.byref(params)))
    assert kernel_identity(node) != packed_identity
    graph.instantiate()
    execute(graph, stream)
    assert y.item() == 29


def test_variable_packed_prefix_binds_all_supplied_bytes_without_reading_tail():
    from cuda.bindings import driver as cd
    from cubic_serve.ops.cuda.native_compiler import compile_native
    from cubic_serve.guardrail.graph_observer import kernel_identity
    source = b'''#include <stdint.h>
    struct Args {const int64_t* src;int64_t* dst;char unused[4080];};
    extern "C" __global__ void copy(Args args){if(!threadIdx.x)args.dst[0]=args.src[0];}'''
    device = torch.device('cuda:0')
    x = torch.tensor([37], dtype=torch.int64, device=device)
    y = torch.zeros_like(x)
    module, entry = compile_native((('entry.cu', source),), device, 'test_observer_prefix')
    function = check(cd.cuModuleGetFunction(module.handle, entry.encode()))[0]
    packed = ct.create_string_buffer(32)
    values = (ct.c_void_p * 2)(x.data_ptr(), y.data_ptr())
    ct.memmove(packed, values, ct.sizeof(values))
    size = ct.c_size_t(len(packed))
    # CUDA may return these tags in either order.
    extra = (ct.c_void_p * 5)(2, ct.addressof(size), 1, ct.addressof(packed), 0)
    params = cd.CUDA_KERNEL_NODE_PARAMS()
    params.func = function
    params.gridDimX = params.gridDimY = params.gridDimZ = 1
    params.blockDimX = 32
    params.blockDimY = params.blockDimZ = 1
    params.extra = ct.addressof(extra)
    raw = check(rt.cudaGraphCreate(0))[0]
    executable = None
    try:
        node = check(cd.cuGraphAddKernelNode(cd.CUgraph(int(raw)), [], 0, params))[0]
        identity = kernel_identity(node)
        driver = _driver()
        driver.cuFuncGetParamInfo.argtypes = [
            ct.c_void_p, ct.c_size_t, ct.POINTER(ct.c_size_t), ct.POINTER(ct.c_size_t)]
        offset, declared_size = ct.c_size_t(), ct.c_size_t()
        _check(driver.cuFuncGetParamInfo(
            int(function), 0, ct.byref(offset), ct.byref(declared_size)))
        assert offset.value == 0 and declared_size.value == 4096
        assert identity[-1] == ((-1, packed.raw),)
        observer = GraphObserver((SnapshotPoint('input', (('x', x),), 1),),
                                 byte_budget=4096, device=device)
        before = graph_identity(raw)
        observer.attach(raw, rt.cudaGraphNode_t(int(node)), 'input')
        executable = check(rt.cudaGraphInstantiateWithFlags(raw, 0))[0]
        observer.begin()
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream())
        check(rt.cudaGraphLaunch(executable, stream.cuda_stream))
        stream.synchronize()
        events = observer.finish(gather_schedules=lambda s: (s,))
        assert events[0][1][0][1].view(torch.int64).item() == 37
        assert y.item() == 37
        check(rt.cudaGraphExecDestroy(executable))
        executable = None
        observer.detach()
        assert graph_identity(raw) == before
        packed[31] = b'\x7f'
        check(cd.cuGraphKernelNodeSetParams(cd.CUgraphNode(int(node)), params))
        assert kernel_identity(node) != identity  # includes unused supplied bytes.
    finally:
        if executable is not None:
            check(rt.cudaGraphExecDestroy(executable))
        check(rt.cudaGraphDestroy(raw))


def test_embedded_sealed_children_require_origin_count_full_arguments_and_phase_order():
    value = torch.zeros(8, device='cuda:0')
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    first, second = (torch.cuda.CUDAGraph(keep_graph=True) for _ in range(2))
    with torch.cuda.graph(first, stream=stream):value.add_(1)
    with torch.cuda.graph(second, stream=stream):value.mul_(2)
    stream.synchronize()
    parent = check(rt.cudaGraphCreate(0))[0]
    try:
        a = check(rt.cudaGraphAddChildGraphNode(parent, [], 0, first.raw_cuda_graph()))[0]
        b = check(rt.cudaGraphAddChildGraphNode(parent, [a], 1, second.raw_cuda_graph()))[0]
        assert prove_child_graph_clones(first.raw_cuda_graph(), parent)[0]['node'] == int(a)
        assert prove_child_graph_clones(second.raw_cuda_graph(), parent)[0]['node'] == int(b)
        assert prove_graph_precedes(parent, a, b)['dependency_path']
        with pytest.raises(RuntimeError, match='dependency is missing'):
            prove_graph_precedes(parent, b, a)
        with pytest.raises(ValueError, match='distinct owned'):
            prove_graph_precedes(parent, a, a)
        child = check(rt.cudaGraphChildGraphNodeGetGraph(a))[0]
        check(rt.cudaGraphAddEmptyNode(child, [], 0))
        with pytest.raises(RuntimeError, match='inventory differs'):
            prove_child_graph_clones(first.raw_cuda_graph(), parent)
    finally:
        check(rt.cudaGraphDestroy(parent))
        first.reset();second.reset()


def test_embedded_clone_copy_count_does_not_accept_a_partial_inventory():
    value = torch.zeros(8, device='cuda:0')
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    leaf = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(leaf, stream=stream):value.add_(1)
    stream.synchronize()
    parent = check(rt.cudaGraphCreate(0))[0]
    try:
        check(rt.cudaGraphAddChildGraphNode(parent, [], 0, leaf.raw_cuda_graph()))
        check(rt.cudaGraphAddChildGraphNode(parent, [], 0, leaf.raw_cuda_graph()))
        with pytest.raises(RuntimeError, match='copy inventory differs'):
            prove_child_graph_clones(leaf.raw_cuda_graph(), parent)
        assert len(prove_child_graph_clones(leaf.raw_cuda_graph(), parent, expected_count=2)) == 2
    finally:
        check(rt.cudaGraphDestroy(parent));leaf.reset()


def test_header_only_completion_returns_no_data_proof(monkeypatch):
    _, source, _, stream, graph, raw, node, observer = fixture()
    observer.attach(raw, node, 'input')
    graph.instantiate();observer.begin();execute(graph, stream)
    outputs = {record[1].untyped_storage().data_ptr() for record in observer.records}
    original_cpu = torch.Tensor.cpu
    def cpu(tensor, *args, **kwargs):
        if tensor.untyped_storage().data_ptr() in outputs:
            raise AssertionError('header-only completion copied observed payload')
        return original_cpu(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, 'cpu', cpu)
    assert observer.finish(gather_schedules=lambda s: (s,), read_payload=False) is None
    assert not observer.active
    observer.detach()


def test_cancelled_observation_yields_no_verdict_and_restores_graph():
    _, source, _, stream, graph, raw, node, observer = fixture()
    before = graph_identity(raw)
    observer.attach(raw, node, 'input')
    graph.instantiate();observer.begin();execute(graph, stream)
    observer.discard(stream=stream)
    with pytest.raises(RuntimeError, match='have not begun'):
        observer.finish(gather_schedules=lambda s: (s,))
    observer.detach()
    assert graph_identity(raw) == before
