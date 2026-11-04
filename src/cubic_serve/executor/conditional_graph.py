"""Startup composition of captured children into conditional CUDA graphs.

Memory operations in conditional children use byte-preserving kernels. Captured
model arithmetic, operand addresses, dependencies and original child ownership
stay unchanged. No graph, selector or transport is changed during replay.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from cuda.bindings import driver as d, runtime as rt


@triton.jit
def _memory_kernel(source, destination, n, value, element_bytes,
                   COPY: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    if COPY:
        data = tl.load(source + i, i < n, other=0)
    else:
        data = ((value.to(tl.uint32) >> ((i % element_bytes) * 8)) & 255).to(tl.uint8)
    tl.store(destination + i, data, i < n)


class _Pointer:
    dtype = torch.uint8

    def __init__(self, address):
        self.address = int(address)

    def data_ptr(self):
        return self.address


def _check(result):
    if int(result[0]):
        raise RuntimeError('conditional graph operation failed: ' + str(result[0]))
    return result[1:]


def _nodes(graph):
    _, count = _check(d.cuGraphGetNodes(graph))
    return _check(d.cuGraphGetNodes(graph, count))[0] if count else []


_loaded_kernels = {}


def _prepare_kernel(context_id, source, destination, size, value, element_bytes, copy):
    # Bind the actual scalar widths and pointer alignment before capture. Keep
    # modules for their context lifetime, as copied graphs retain CUfunctions.
    # Warmup compiles/loads without executing or touching model operands.
    compiled = _memory_kernel.warmup(
        _Pointer(source), _Pointer(destination), size, value, element_bytes,
        COPY=copy, BLOCK=256, grid=(triton.cdiv(size, 256),),
    )
    compiled._init_handles()
    _loaded_kernels[context_id, compiled.hash] = compiled


def _replace_memory_node(graph, node, kind):
    if kind == d.CUgraphNodeType.CU_GRAPH_NODE_TYPE_MEMCPY:
        p = _check(d.cuGraphMemcpyNodeGetParams(node))[0]
        if (p.srcMemoryType != d.CUmemorytype.CU_MEMORYTYPE_DEVICE
                or p.dstMemoryType != d.CUmemorytype.CU_MEMORYTYPE_DEVICE
                or p.Height != 1 or p.Depth != 1
                or any((p.srcXInBytes, p.srcY, p.srcZ, p.dstXInBytes, p.dstY, p.dstZ))):
            raise RuntimeError('conditional graph requires a contiguous device copy')
        source, destination, size = int(p.srcDevice), int(p.dstDevice), p.WidthInBytes
        value, element_bytes, copy = 0, 1, True
    else:
        p = _check(d.cuGraphMemsetNodeGetParams(node))[0]
        if p.height != 1 or p.elementSize not in (1, 2, 4):
            raise RuntimeError('conditional graph requires a contiguous device memset')
        source = destination = int(p.dst)
        size, value, element_bytes, copy = p.width * p.elementSize, p.value, p.elementSize, False
    if size < 1:
        raise RuntimeError('conditional graph memory operation is empty')
    n = _check(d.cuGraphNodeGetDependencies(node))[-1]
    incoming = _check(d.cuGraphNodeGetDependencies(node, n)) if n else ([], [], 0)
    n = _check(d.cuGraphNodeGetDependentNodes(node))[-1]
    outgoing = _check(d.cuGraphNodeGetDependentNodes(node, n)) if n else ([], [], 0)
    for edge in (*incoming[1], *outgoing[1]):
        if edge.type or edge.from_port or edge.to_port:
            raise RuntimeError('conditional transport requires ordinary dependency edges')
    context = _check(d.cuCtxGetCurrent())[0]
    _prepare_kernel(int(_check(d.cuCtxGetId(context))[0]), source, destination,
                    size, value, element_bytes, copy)
    captured = torch.cuda.CUDAGraph(keep_graph=True)
    try:
        with torch.cuda.graph(captured, stream=torch.cuda.Stream()):
            _memory_kernel[(triton.cdiv(size, 256),)](
                _Pointer(source), _Pointer(destination), size, value, element_bytes,
                COPY=copy, BLOCK=256,
            )
        captured_nodes = _nodes(captured.raw_cuda_graph())
        if (len(captured_nodes) != 1 or _check(d.cuGraphNodeGetType(captured_nodes[0]))[0]
                != d.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL):
            raise RuntimeError('conditional transport capture contains unexpected operations')
        params = _check(d.cuGraphKernelNodeGetParams(captured_nodes[0]))[0]
        replacement = _check(d.cuGraphAddKernelNode(
            graph, incoming[0], len(incoming[0]), params,
        ))[0]
    finally:
        captured.reset()
    if outgoing[0]:
        _check(d.cuGraphAddDependencies(graph, [replacement] * len(outgoing[0]),
                                      outgoing[0], None, len(outgoing[0])))
    _check(d.cuGraphDestroyNode(node))


def _has_memory(graph):
    found = False
    for node in _nodes(graph):
        kind = _check(d.cuGraphNodeGetType(node))[0]
        if kind in (d.CUgraphNodeType.CU_GRAPH_NODE_TYPE_MEMCPY,
                    d.CUgraphNodeType.CU_GRAPH_NODE_TYPE_MEMSET):
            found = True
        elif kind == d.CUgraphNodeType.CU_GRAPH_NODE_TYPE_GRAPH:
            found |= _has_memory(_check(d.cuGraphChildGraphNodeGetGraph(node))[0])
    return found


def _lower_owned(graph):
    for node in _nodes(graph):
        kind = _check(d.cuGraphNodeGetType(node))[0]
        if kind == d.CUgraphNodeType.CU_GRAPH_NODE_TYPE_GRAPH:
            source = _check(d.cuGraphChildGraphNodeGetGraph(node))[0]
            if _has_memory(source):
                cloned = _check(d.cuGraphClone(source))[0]
                try:
                    _lower_owned(cloned)
                    params = d.CUgraphNodeParams()
                    params.type = kind
                    params.graph.graph = cloned
                    _check(d.cuGraphNodeSetParams(node, params))
                finally:
                    _check(d.cuGraphDestroy(cloned))
        elif kind in (d.CUgraphNodeType.CU_GRAPH_NODE_TYPE_MEMCPY,
                      d.CUgraphNodeType.CU_GRAPH_NODE_TYPE_MEMSET):
            _replace_memory_node(graph, node, kind)


def add_conditional_child(parent, dependencies, source):
    """Append a captured child without sharing mutable graph ownership.

    CUDA copies a source child when it is attached. Clone and normalize before
    attachment, because a borrowed conditional-body view cannot be cloned.
    Temporary definitions are released immediately after that copy; no extra
    executable graph or tensor backing is retained.
    """
    if not _has_memory(source):
        return _check(rt.cudaGraphAddChildGraphNode(
            parent, dependencies, len(dependencies), source,
        ))[0]
    cloned = _check(d.cuGraphClone(source))[0]
    try:
        _lower_owned(cloned)
        return _check(rt.cudaGraphAddChildGraphNode(
            parent, dependencies, len(dependencies), cloned,
        ))[0]
    finally:
        _check(d.cuGraphDestroy(cloned))
