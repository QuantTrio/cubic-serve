from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from cubic_serve.executor.runner import configure_reference_math
from cubic_serve.ops.cuda.native import CudaModule, CudaScalar
from cubic_serve.quantization.common.binding import PackedOperatorBinding


def test_native_image_interning_binds_exact_bytes_and_unique_context(monkeypatch):
    from cubic_serve.ops.cuda import native

    monkeypatch.setattr(native, "_module_images", {})
    context = [17]
    loads = []
    monkeypatch.setattr(native.cuda_driver, "cuCtxGetCurrent", lambda: (0, 3))
    monkeypatch.setattr(native.cuda_driver, "cuCtxGetId", lambda handle: (0, context[0]))

    def load(image):
        loads.append(image)
        return 0, len(loads)

    monkeypatch.setattr(native.cuda_driver, "cuModuleLoadData", load)
    assert native._load_cubin_image(b"first") == native._load_cubin_image(bytes(b"first")) == 1
    assert native._load_cubin_image(b"second") == 2
    context[0] = 18  # Same recycled context handle, distinct driver identity.
    assert native._load_cubin_image(b"first") == 3
    assert loads == [b"first", b"second", b"first"]
    monkeypatch.setattr(native, "_hash", lambda image: "collision")
    assert native._load_cubin_image(b"first") == 4
    with pytest.raises(RuntimeError, match="collision"):
        native._load_cubin_image(b"other")


@pytest.mark.parametrize("scalar", [CudaScalar("i32", 2**31), CudaScalar("i64", -(2**63)-1),
                                   CudaScalar("i32", True), CudaScalar("bool", 1)])
def test_native_scalar_rejects_abi_coercion(scalar):
    with pytest.raises(ValueError):
        scalar.pack()


@pytest.mark.skipif(os.environ.get("CUBIC_RUN_CUDA_NATIVE_TESTS") != "1",
                    reason="set CUBIC_RUN_CUDA_NATIVE_TESTS=1 for CUDA qualification")
def test_native_image_receipt_graph_and_operand_ownership():
    configure_reference_math()
    baseline = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda-13.0")) / "bin/nvcc"
    nvcc = str(baseline) if baseline.is_file() else shutil.which("nvcc")
    assert nvcc is not None
    source = b'''extern "C" __global__ void copy16(const unsigned short* x,
        unsigned short* y, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) y[i] = x[i];
    }'''
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "copy.cu"
        path.write_bytes(source)
        target = Path(tmp) / "copy.ptx"
        flags = ("--ptx", "-arch=compute_80", "-std=c++17")
        subprocess.run([nvcc, *flags, str(path), "-o", str(target)], check=True,
                       capture_output=True, timeout=60)
        # Allocate on the caller's CUDA context before driver linking.
        x = torch.arange(33, dtype=torch.bfloat16, device="cuda:0")
        y = torch.full_like(x, float("nan"))
        module = CudaModule(target.read_bytes(), sources={"copy.cu": source},
                            compiler_metadata={"compiler": "nvcc", "flags": flags},
                            device=x.device)
        sibling = CudaModule(module.ptx, cubin=module.cubin, sources={"copy.cu": source},
                             compiler_metadata={"compiler": "nvcc", "flags": flags},
                             device=x.device)
        assert sibling.handle == module.handle
        assert sibling.cubin_hash == module.cubin_hash
        launch = module.bind("copy16", (x, y, CudaScalar("i32", 33)),
                             operator_id="test.cuda.copy16", grid=(1, 1, 1), block=(64, 1, 1))
        receipt = launch.launch()
        assert torch.equal(x, y)
        artifact, = launch.project_artifacts()
        import hashlib

        assert module.ptx_hash == "b2s256:" + hashlib.blake2s(module.ptx).hexdigest()
        assert module.cubin_hash == "b2s256:" + hashlib.blake2s(module.cubin).hexdigest()
        assert artifact.backend == "cuda-driver"
        assert tuple(k for k, _ in artifact.stage_hashes) == ("ptx", "cubin")
        assert artifact.compiled_hash == dict(artifact.stage_hashes)["cubin"]
        assert receipt.bindings == launch.bindings and receipt.grid == artifact.grid
        with pytest.raises(ValueError):
            replace(artifact, stage_hashes=artifact.stage_hashes[:1])
        with pytest.raises(ValueError):
            replace(artifact, backend="triton")
        with pytest.raises(ValueError, match="cubin stage"):
            replace(artifact, compiled_hash="b2s256:" + "0" * 64)
        with pytest.raises(AttributeError, match="immutable"):
            launch.grid = (2, 1, 1)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch.launch()
        x.copy_(torch.randn_like(x))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8))
        graph.reset()
        scratch = torch.empty_like(x)
        replacement = torch.empty_like(x)
        first = module.bind("copy16", (x, scratch, CudaScalar("i32", 33)),
                            operator_id="test.cuda.first", grid=(1, 1, 1), block=(64, 1, 1))
        second = module.bind("copy16", (scratch, y, CudaScalar("i32", 33)),
                             operator_id="test.cuda.second", grid=(1, 1, 1), block=(64, 1, 1))
        private = torch.empty_like(x)
        composed = PackedOperatorBinding(y, (first, second), scratch=(scratch, private),
                                         reusable_scratch=True)
        composed.rebind_private_scratch(replacement)
        assert composed.scratch[1] is private
        scratch.fill_(float("nan"))
        receipts = composed.launch()
        assert torch.equal(x, y) and torch.equal(x, replacement)
        assert len(composed.project_artifacts()) == len(receipts) == 2
        assert all(r._primitive_token is composed._receipt_token for r in receipts)
        parameter = first._parameters[0]
        first._parameters[0] = 0
        with pytest.raises(ValueError, match="operands changed"):
            first.launch()
        first._parameters[0] = parameter
        scalar = first._packed[-1].value
        first._packed[-1].value = 32
        with pytest.raises(ValueError, match="operands changed"):
            first.launch()
        first._packed[-1].value = scalar
        y.resize_(34)
        replacement.fill_(123)
        with pytest.raises(ValueError, match="operands changed"):
            composed.launch()
        assert bool((replacement == 123).all()), "preflight must precede every producer"
        with pytest.raises(ValueError, match="operands changed"):
            launch.launch()
