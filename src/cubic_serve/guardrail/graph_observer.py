"""Offline, reversible read-only witnesses for an already captured CUDA DAG.

This proves capture/control identity, never model arithmetic. The caller owns
pre-readiness exclusivity, executable reinstantiation and the independent oracle.
No reference work or host reads run from a model callback. CUDA 13.0 permits
mutation of borrowed embedded child graphs; conditional graphs cannot be cloned.
"""
from __future__ import annotations

import ctypes as ct
import gc
import hashlib
from dataclasses import dataclass

import torch
from cuda.bindings import runtime as rt

from cubic_serve.executor.resident_window import check
from cubic_serve.ops.cuda.launch_cost import _check, _driver, _KernelNodeParams
from cubic_serve.ops.cuda.native import CudaScalar
from cubic_serve.ops.cuda.native_compiler import compile_native

_SOURCE = b'''
#include <stdint.h>
extern "C" __global__ void claim(int* visits,int* slot,int* count,int* header,
                                 int* overflow,int point,int limit,int capacity){
 if(threadIdx.x || blockIdx.x)return;
 int visit=atomicAdd(visits,1),event=atomicAdd(count,1);
 if(visit>=limit || event>=capacity){atomicExch(overflow,1);slot[0]=-1;return;}
 slot[0]=visit;header[event*2]=point;header[event*2+1]=visit;
}
'''
_COPY = b'''
#include <stdint.h>
extern "C" __global__ void snapshot(const int64_t* descriptors,unsigned char* output,
                                    const int* slot,int64_t width){
 int visit=slot[0];if(visit<0)return;
 const int64_t* d=descriptors+blockIdx.x*21;
 const unsigned char* input=(const unsigned char*)(uintptr_t)d[0];
 int64_t count=d[2],element=d[3],dims=d[4];
 for(int64_t i=(int64_t)blockIdx.y*blockDim.x+threadIdx.x;i<count;
     i+=(int64_t)gridDim.y*blockDim.x){
  int64_t index=i,offset=0;
  for(int axis=(int)dims-1;axis>=0;--axis){offset+=(index%d[5+axis])*d[13+axis];index/=d[5+axis];}
  for(int b=0;b<element;++b)output[(int64_t)visit*width+d[1]+i*element+b]=input[offset*element+b];
 }
}
'''


def graph_nodes(graph):
    count = check(rt.cudaGraphGetNodes(graph, 0))[1]
    return () if not count else tuple(check(rt.cudaGraphGetNodes(graph, count))[0])


def graph_edges(graph):
    count = check(rt.cudaGraphGetEdges(graph, 0))[3]
    if not count:
        return ()
    sources, targets, data, _ = check(rt.cudaGraphGetEdges(graph, count))
    return tuple(sorted(
        (int(a), int(b), int(d.from_port), int(d.to_port), int(d.type), tuple(d.reserved))
        for a, b, d in zip(sources, targets, data, strict=True)
    ))


def kernel_identity(node):
    """Copy actual argument values: queried argument pointers are node-owned."""
    driver = _driver()
    driver.cuFuncGetParamInfo.argtypes = [ct.c_void_p, ct.c_size_t,
                                         ct.POINTER(ct.c_size_t), ct.POINTER(ct.c_size_t)]
    params = _KernelNodeParams()
    _check(driver.cuGraphKernelNodeGetParams_v2(int(node), ct.byref(params)))
    function = params.func
    if not function and params.kernel:
        driver.cuKernelGetFunction.argtypes = [ct.POINTER(ct.c_void_p), ct.c_void_p]
        bound = ct.c_void_p()
        _check(driver.cuKernelGetFunction(ct.byref(bound), params.kernel))
        function = bound.value
    pointers = ct.cast(params.parameters, ct.POINTER(ct.c_void_p))
    packed, packed_size = None, None
    if params.extra:
        if params.parameters:
            raise RuntimeError("audit kernel has two argument representations")
        extra = ct.cast(params.extra, ct.POINTER(ct.c_void_p))
        index = 0
        seen = set()
        while extra[index]:
            tag = extra[index]
            if tag not in (1, 2) or tag in seen:
                raise RuntimeError("audit kernel has unsupported packed argument metadata")
            seen.add(tag)
            if tag == 1:  # CU_LAUNCH_PARAM_BUFFER_POINTER, CUDA 13.0.
                packed = extra[index + 1]
            else:  # CU_LAUNCH_PARAM_BUFFER_SIZE points at a size_t.
                packed_size = ct.c_size_t.from_address(extra[index + 1]).value
            index += 2
        if seen != {1, 2}:
            raise RuntimeError("audit kernel packed argument metadata is incomplete")
        # Bind exactly the supplied launch buffer, including its byte length.
        # NCCL v2.29.7 enqueue.cc supplies kernelArgsSize, a variable work prefix
        # of a declared 4KiB argument. ParamInfo's declared size is not the
        # supplied buffer extent. Never read or manufacture its unused tail.
        arguments = ((-1, ct.string_at(packed, packed_size)),)
        return (int(function), params.gx, params.gy, params.gz,
                params.bx, params.by, params.bz, params.shared, arguments)
    arguments = []
    index = 0
    while True:
        offset, size = ct.c_size_t(), ct.c_size_t()
        status = driver.cuFuncGetParamInfo(function, index, ct.byref(offset), ct.byref(size))
        if status == 1:  # CUDA_ERROR_INVALID_VALUE: index past final parameter.
            break
        _check(status)
        if params.parameters:
            address = pointers[index]
        else:
            raise RuntimeError("audit kernel argument storage is missing")
        arguments.append((offset.value, ct.string_at(address, size.value)))
        index += 1
    return (int(function), params.gx, params.gy, params.gz,
            params.bx, params.by, params.bz, params.shared, tuple(arguments))


def graph_identity(graph):
    """Original node IDs, edges, kernel geometry and copied argument bytes.

    Child contents are separate proof units, named by their borrowed graph handle.
    Unsupported node kinds fail closed instead of producing a partial memento.
    """
    rows = []
    for node in graph_nodes(graph):
        kind = check(rt.cudaGraphNodeGetType(node))[0]
        if kind == rt.cudaGraphNodeType.cudaGraphNodeTypeKernel:
            value = kernel_identity(node)
        elif kind == rt.cudaGraphNodeType.cudaGraphNodeTypeGraph:
            value = int(check(rt.cudaGraphChildGraphNodeGetGraph(node))[0])
        elif kind == rt.cudaGraphNodeType.cudaGraphNodeTypeEmpty:
            value = None
        elif kind == rt.cudaGraphNodeType.cudaGraphNodeTypeMemcpy:
            p = check(rt.cudaGraphMemcpyNodeGetParams(node))[0]
            def position(v):
                return int(v.x), int(v.y), int(v.z)
            def pointer(v):
                return int(v.ptr), int(v.pitch), int(v.xsize), int(v.ysize)
            value = (int(p.srcArray), position(p.srcPos), pointer(p.srcPtr),
                     int(p.dstArray), position(p.dstPos), pointer(p.dstPtr),
                     int(p.extent.width), int(p.extent.height), int(p.extent.depth), int(p.kind))
        elif kind == rt.cudaGraphNodeType.cudaGraphNodeTypeMemset:
            p = check(rt.cudaGraphMemsetNodeGetParams(node))[0]
            value = tuple(int(getattr(p, key)) for key in
                          ("dst", "pitch", "value", "elementSize", "width", "height"))
        elif kind == rt.cudaGraphNodeType.cudaGraphNodeTypeEventRecord:
            value = int(check(rt.cudaGraphEventRecordNodeGetEvent(node))[0])
        elif kind == rt.cudaGraphNodeType.cudaGraphNodeTypeWaitEvent:
            value = int(check(rt.cudaGraphEventWaitNodeGetEvent(node))[0])
        else:
            raise RuntimeError(f"audit graph memento has unsupported node kind {int(kind)}")
        rows.append((int(node), int(kind), value))
    return tuple(sorted(rows)), graph_edges(graph)


def prove_graph_clone(source, clone):
    """Prove one direct CUDA clone's nodes, arguments, edges and nested bodies.

    CUDA's origin mapping, rather than node order or a structural hash, binds
    each actual clone node. A clone of a clone needs a separate proof for each
    link. This establishes captured-body identity only: it supplies neither
    input/state provenance nor a numerical, execution or ownership verdict.
    """
    source_rows, source_edges = graph_identity(source)
    clone_rows, clone_edges = graph_identity(clone)
    if len(source_rows) != len(clone_rows):
        raise RuntimeError("captured graph clone node inventory differs")
    mapped = {}
    for node, _, _ in source_rows:
        result = rt.cudaGraphNodeFindInClone(rt.cudaGraphNode_t(node), clone)
        if int(result[0]) != 0:
            raise RuntimeError("captured graph clone origin is unavailable")
        mapped[node] = int(result[1])
    if len(set(mapped.values())) != len(mapped) or set(mapped.values()) != {
            node for node, _, _ in clone_rows}:
        raise RuntimeError("captured graph clone node ownership differs")
    wanted_edges = tuple(sorted((mapped[a], mapped[b], *rest)
                                for a, b, *rest in source_edges))
    if wanted_edges != clone_edges:
        raise RuntimeError("captured graph clone dependencies differ")
    actual = {node: (kind, value) for node, kind, value in clone_rows}
    nested = []
    for node, kind, value in source_rows:
        clone_kind, clone_value = actual[mapped[node]]
        if kind != clone_kind:
            raise RuntimeError("captured graph clone node type differs")
        if kind == int(rt.cudaGraphNodeType.cudaGraphNodeTypeGraph):
            nested.append(prove_graph_clone(rt.cudaGraph_t(value), rt.cudaGraph_t(clone_value)))
        elif value != clone_value:
            raise RuntimeError("captured graph clone parameters differ")
    return dict(nodes=tuple(sorted(mapped.items())), nested=tuple(nested),
                source_digest=hashlib.blake2s(repr((source_rows, source_edges)).encode()).hexdigest(),
                clone_digest=hashlib.blake2s(repr((clone_rows, clone_edges)).encode()).hexdigest())


def prove_child_graph_clones(source, parent, *, expected_count=1):
    """Bind direct embedded copies by CUDA origin, then prove their full DAGs.

    Matching kernel names or fingerprints is insufficient. A copied graph with
    different arguments/edges fails the existing complete recursive proof. The
    parent may contain other explicitly owned children; the caller must account
    for those separately and prove its surrounding control/byte-transport rules.
    """
    if type(expected_count) is not int or expected_count <= 0:
        raise ValueError("embedded graph proof requires a positive copy count")
    original = graph_nodes(source)
    if not original:
        raise ValueError("empty graph has no unambiguous embedded origin")
    copies = []
    for node in graph_nodes(parent):
        if check(rt.cudaGraphNodeGetType(node))[0] != rt.cudaGraphNodeType.cudaGraphNodeTypeGraph:
            continue
        child = check(rt.cudaGraphChildGraphNodeGetGraph(node))[0]
        match = rt.cudaGraphNodeFindInClone(original[0], child)
        if int(match[0]) != 0:
            continue
        copies.append(dict(node=int(node), graph=int(child),
                           clone=prove_graph_clone(source, child)))
    if len(copies) != expected_count:
        raise RuntimeError("embedded sealed graph copy inventory differs")
    return tuple(copies)


def prove_graph_precedes(parent, before, after):
    """Prove actual dependency reachability between two declared parent nodes."""
    nodes = {int(node) for node in graph_nodes(parent)}
    before, after = int(before), int(after)
    if before == after or before not in nodes or after not in nodes:
        raise ValueError("graph order proof requires two distinct owned nodes")
    outgoing = {}
    for start, stop, *_ in graph_edges(parent):
        outgoing.setdefault(start, set()).add(stop)
    pending, visited = [before], set()
    while pending:
        node = pending.pop()
        if node in visited:
            continue
        visited.add(node)
        successors = outgoing.get(node, ())
        if after in successors:
            return dict(before=before, after=after, dependency_path=True)
        pending.extend(successors)
    raise RuntimeError("sealed graph phase dependency is missing")


@dataclass(frozen=True)
class SnapshotPoint:
    name: str
    tensors: tuple[tuple[str, torch.Tensor], ...]
    visits: int


class GraphObserver:
    """Bounded immutable snapshots; schedule consensus precedes CPU payload reads."""

    def __init__(self, points, *, byte_budget, device):
        self.points = tuple(points)
        if not self.points or len({p.name for p in self.points}) != len(self.points):
            raise ValueError("audit snapshot point names must be unique")
        if any(type(p.visits) is not int or p.visits <= 0 for p in self.points):
            raise ValueError("audit snapshot visit capacities must be positive")
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("graph observer requires a CUDA device")
        capacity = sum(p.visits for p in self.points)
        layouts, total = [], capacity * 8 + 8
        for point in self.points:
            if not point.tensors or len({n for n, _ in point.tensors}) != len(point.tensors):
                raise ValueError("audit snapshot fields must be nonempty and unique")
            layout, width = [], 0
            for name, tensor in point.tensors:
                if (tensor.device != self.device or tensor.layout != torch.strided
                        or tensor.ndim > 8 or tensor.is_conj() or tensor.is_neg()):
                    raise ValueError("audit snapshot tensor has unsupported layout")
                size = tensor.numel() * tensor.element_size()
                layout.append((name, tensor, width, size))
                width += size
            layouts.append((tuple(layout), width))
            total += point.visits * width + len(layout) * 21 * 8 + 8
        if type(byte_budget) is not int or total > byte_budget:
            raise MemoryError(f"audit snapshot needs {total} bytes, budget {byte_budget}")
        self.allocated_bytes = total
        self.header = torch.empty((capacity, 2), dtype=torch.int32, device=self.device)
        self.count = torch.zeros(1, dtype=torch.int32, device=self.device)
        self.overflow = torch.zeros_like(self.count)
        self.records, self.transactions = [], []
        self.active = False
        self.bindings = tuple(self._binding(t) for p in self.points for _, t in p.tensors)
        claim_module, claim_entry = compile_native((("entry.cu", _SOURCE),), self.device,
                                                   "audit_graph_claim")
        copy_module, copy_entry = compile_native((("entry.cu", _COPY),), self.device,
                                                 "audit_graph_snapshot")
        for ident, (point, (layout, width)) in enumerate(zip(self.points, layouts, strict=True)):
            descriptors = []
            for _, tensor, offset, _ in layout:
                descriptors.append([tensor.data_ptr(), offset, tensor.numel(),
                                    tensor.element_size(),
                                    tensor.ndim, *tensor.shape, *([1] * (8 - tensor.ndim)),
                                    *tensor.stride(), *([0] * (8 - tensor.ndim))])
            desc = torch.tensor(descriptors, dtype=torch.int64, device=self.device)
            output = torch.empty((point.visits, width), dtype=torch.uint8, device=self.device)
            visits, slot = torch.zeros_like(self.count), torch.zeros_like(self.count)
            claim = claim_module.bind(claim_entry,
                (visits, slot, self.count, self.header, self.overflow,
                 *(CudaScalar("i32", v) for v in (ident, point.visits, capacity))),
                operator_id="guardrail.graph_claim", grid=(1, 1, 1), block=(32, 1, 1))
            copy = copy_module.bind(copy_entry, (desc, output, slot, CudaScalar("i64", width)),
                operator_id="guardrail.graph_snapshot",
                grid=(len(layout), 32, 1), block=(128, 1, 1))
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            stream = torch.cuda.Stream(device=self.device)
            stream.wait_stream(torch.cuda.current_stream(self.device))
            # Retired graph/host-mapping cycles may synchronize in finalizers.
            # Run them before capture, and keep collection outside the capture.
            collecting = gc.isenabled()
            if collecting:
                gc.collect()
                gc.disable()
            try:
                with torch.cuda.graph(graph, stream=stream):
                    claim.launch()
                    copy.launch()
            finally:
                if collecting:
                    gc.enable()
            stream.synchronize()
            self.records.append((layout, output, visits, slot, desc, graph, claim, copy))
        source_stores = {t.untyped_storage().data_ptr() for p in self.points for _, t in p.tensors}
        private = [self.header, self.count, self.overflow]
        for _, output, visits, slot, desc, *_ in self.records:
            private.extend((output, visits, slot, desc))
        stores = [t.untyped_storage().data_ptr() for t in private if t.numel()]
        if len(set(stores)) != len(stores) or source_stores.intersection(stores):
            raise RuntimeError("audit observer storage aliases observed inputs")

    def attach(self, graph, node, point):
        """Add one child before node, preserving all original dependency edges."""
        if self.active:
            raise RuntimeError("cannot mutate an active audit observer")
        ident = next(i for i, p in enumerate(self.points) if p.name == point)
        handle = int(graph)
        existing = next((t for t in self.transactions if int(t[0]) == handle), None)
        if existing is None:
            existing = [graph, graph_identity(graph), []]
            self.transactions.append(existing)
        if int(node) not in {int(n) for n in graph_nodes(graph)}:
            raise ValueError("audit insertion node is outside its graph")
        predecessors = [rt.cudaGraphNode_t(a) for a, b, *_ in graph_edges(graph) if b == int(node)]
        added = check(rt.cudaGraphAddChildGraphNode(
            graph, predecessors, len(predecessors), self.records[ident][5].raw_cuda_graph()))[0]
        existing[2].append(added)
        check(rt.cudaGraphAddDependencies(graph, [added], [node], None, 1))
        return added

    def begin(self):
        if self.active:
            raise RuntimeError("previous audit snapshots are still active")
        if tuple(self._binding(t) for p in self.points for _, t in p.tensors) != self.bindings:
            raise RuntimeError("audit snapshot input storage binding changed")
        self.count.zero_()
        self.overflow.zero_()
        for _, _, visits, slot, *_ in self.records:
            visits.zero_()
            slot.fill_(-1)
        self.active = True

    @staticmethod
    def _binding(tensor):
        return (tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()),
                tensor.dtype, tensor.device)

    def finish(self, *, gather_schedules, read_payload=True):
        if not self.active:
            raise RuntimeError("audit snapshots have not begun")
        if type(read_payload) is not bool:
            raise ValueError("audit payload policy must be explicit")
        torch.cuda.current_stream(self.device).synchronize()
        count = int(self.count.item())
        invalid = bool(self.overflow.item()) or not 0 <= count <= self.header.shape[0]
        schedule = None if invalid else tuple(map(tuple, self.header[:count].cpu().tolist()))
        # Include failures in consensus; no rank escapes while another gathers.
        try:
            schedules = gather_schedules((invalid, schedule))
        finally:
            self.active = False
        if invalid or any(item != (False, schedule) for item in schedules):
            raise RuntimeError("audit snapshot overflow or rank event schedule mismatch")
        if not read_payload:
            # Header agreement is a harness fact, never an input/data proof.
            return None
        events = []
        for ident, visit in schedule:
            if not 0 <= ident < len(self.records) or not 0 <= visit < self.points[ident].visits:
                raise RuntimeError("audit snapshot event index is invalid")
            layout, output, *_ = self.records[ident]
            raw = output[visit].cpu()
            fields = tuple((name, raw[offset:offset + size].clone())
                           for name, _, offset, size in layout)
            events.append((self.points[ident].name, fields))
        return tuple(events)

    def discard(self, *, stream):
        """Release an incomplete observation after its owner's stream finishes.

        Cancellation returns no events, agreement or verdict. The caller still
        propagates its original failure and restores the unmodified graph.
        """
        if torch.device(stream.device) != self.device:
            raise ValueError("audit cancellation belongs to a different device")
        stream.synchronize()
        self.active = False

    def detach(self):
        """Caller synchronizes its graph stream before teardown/reinstantiation."""
        if self.active:
            raise RuntimeError("cannot detach an active audit observer")
        failure = None
        for graph, before, added in reversed(self.transactions):
            for node in reversed(added):
                check(rt.cudaGraphDestroyNode(node))
            if graph_identity(graph) != before:
                failure = RuntimeError(
                    "audit changed original graph nodes, arguments or dependencies")
        self.transactions.clear()
        if failure is not None:
            raise failure
