"""Process-local execution readiness, distinct from numerical audit results.

A cache may choose a kernel, but cannot confer readiness on a new CUDA context.
Constructors bind the real graphs/modules and their storage before this seal is
issued. Audit flags remain untouched; independent qualification still exercises
the ordinary numerical/state oracles in the explicit acceptance workflow.
"""
from __future__ import annotations

import torch

from cubic_serve.ops.cuda.compiled_artifact import bind_tensor


def execution_ready(owner, capability="qualified") -> bool:
    if owner is None or getattr(owner, "_closed", False):
        return False
    if getattr(owner, "exec", True) is None or getattr(owner, "graph", True) is None:
        return False
    return (getattr(owner, capability, False) is True
            or capability in getattr(owner, "_execution_capabilities", ()))


def prepare_execution(owner, *, graphs=(), modules=(), calls=(), tensors=(),
                      capabilities=("qualified",)):
    """Seal resources that were actually constructed in the current process.

    This checks binding/resource admission, never sets an audited-passed flag,
    never persists addresses, and never launches model or fixture work.
    """
    if getattr(owner, "_closed", False) or not (graphs or modules or calls):
        raise RuntimeError("execution preparation requires live bound resources")
    for graph in graphs:
        if graph is None or isinstance(graph, int) and graph == 0:
            raise RuntimeError("execution preparation found a missing CUDA graph")
        if isinstance(graph, torch.cuda.CUDAGraph) and not graph.raw_cuda_graph_exec():
            raise RuntimeError("execution preparation found an uncaptured CUDA graph")
    for module in modules:
        if not getattr(module, "cubin_hash", None):
            raise RuntimeError("execution preparation found an unbound CUDA image")
    for call in calls:
        bindings = getattr(call, "bindings", ())
        operands = getattr(call, "tensors", ())
        if len(bindings) != len(operands) or any(
                not binding.matches(tensor)
                for binding, tensor in zip(bindings, operands, strict=True)):
            raise RuntimeError("execution preparation found changed kernel operands")
    bindings = tuple(bind_tensor(tensor) for tensor in tensors)
    owner._execution_resource_bindings = bindings
    owner._execution_capabilities = frozenset(capabilities)
