"""Operator cache reuse follows CUDA kernels and GPU model, not the project."""
from types import ModuleType, SimpleNamespace

import torch

from cubic_serve.ops.cuda import cache_identity as cache


def test_same_gpu_model_reuses_across_uuid_and_cuda_ordinal(monkeypatch, tmp_path):
    monkeypatch.setenv("CUBIC_SERVE_CACHE_DIR", str(tmp_path))
    properties = SimpleNamespace(name="Model A", uuid="physical-0", total_memory=123456,
        multi_processor_count=96, L2_cache_size=2048,
        shared_memory_per_block_optin=65536, warp_size=32)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda d: properties)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d: (9, 0))
    cache.gpu_model.cache_clear()
    first = cache.plan_path("operator", ("binary-A",), torch.device("cuda", 0), (128, "int4"))
    properties.uuid = "physical-7"
    second = cache.plan_path("operator", ("binary-A",), torch.device("cuda", 7), (128, "int4"))
    assert first == second
    cache.gpu_model.cache_clear()


def test_geometry_is_an_entry_but_kernel_or_gpu_change_is_a_new_fingerprint(monkeypatch, tmp_path):
    monkeypatch.setenv("CUBIC_SERVE_CACHE_DIR", str(tmp_path))
    model = ["Model A"]
    monkeypatch.setattr(cache, "gpu_model", lambda d: tuple(model))
    device = torch.device("cuda", 0)
    first = cache.plan_path("operator", ("binary-A",), device, (128, "int4"))
    second = cache.plan_path("operator", ("binary-A",), device, (256, "int8"))
    assert first.parent == second.parent and first != second
    assert first.parent != cache.plan_path("operator", ("binary-B",), device, (128, "int4")).parent
    model[0] = "Model B"
    assert first.parent != cache.plan_path("operator", ("binary-A",), device, (128, "int4")).parent


def test_python_and_logging_changes_do_not_change_kernel_identity(tmp_path):
    source = tmp_path / "owner.py"
    source.write_text("print('old logging')\n")
    native = tmp_path / "op.cu"
    native.write_text("__global__ void op() {}\n")
    module = ModuleType("owner")
    module.__file__ = str(source)
    first = cache.module_kernels(module)
    source.write_text("print('new logging and controller implementation')\n")
    cache.module_kernels.cache_clear()
    assert cache.module_kernels(module) == first
    native.write_text("__global__ void op() { __syncthreads(); }\n")
    cache.module_kernels.cache_clear()
    assert cache.module_kernels(module) != first


def test_resource_partition_is_an_entry_not_a_gpu_model_fingerprint(monkeypatch, tmp_path):
    monkeypatch.setenv("CUBIC_SERVE_CACHE_DIR", str(tmp_path))
    resources = [96, 80 * 2**30]
    monkeypatch.setattr(cache, "gpu_model", lambda d: ("GPU model", *resources))
    before = cache.plan_path("operator", ("binary",), "cuda:0", (64, "bf16"))
    resources[:] = [48, 40 * 2**30]
    after = cache.plan_path("operator", ("binary",), "cuda:7", (64, "bf16"))
    assert before.parent == after.parent and before != after
