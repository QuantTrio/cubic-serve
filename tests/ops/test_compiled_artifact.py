from dataclasses import replace

import pytest
import torch

from cubic_serve.ops.cuda.compiled_artifact import TensorBinding, bind_tensor


def test_scalar_binding_is_one_element_and_zero_extent_remains_invalid():
    scalar = TensorBinding(1, 2, 2, 0, (), (), "torch.float32", "cuda:0")
    assert scalar.shape == scalar.stride == ()
    for shape, stride in (((0,), (1,)), ((), (1,)), ((1,), ())):
        with pytest.raises(ValueError):
            replace(scalar, shape=shape, stride=stride)


def test_live_binding_comparison_preserves_every_identity_field():
    tensor = torch.arange(24).reshape(4, 6)
    binding = bind_tensor(tensor)
    assert binding.matches(tensor)
    assert not binding.matches(tensor.view_as(tensor))
    for field, value in (
        ("object_id", binding.object_id + 1), ("pointer", binding.pointer + 8),
        ("storage_pointer", binding.storage_pointer + 8), ("storage_offset", 1),
        ("shape", (6, 4)), ("stride", (1, 4)), ("dtype", "torch.float32"),
        ("device", "cuda:0"),
    ):
        assert not replace(binding, **{field: value}).matches(tensor)
    tensor.transpose_(0, 1)
    assert not binding.matches(tensor)
    tensor.set_(torch.arange(24))
    assert not binding.matches(tensor)
    scalar = torch.tensor(1.)
    assert bind_tensor(scalar).matches(scalar)


def test_binding_reuse_rechecks_abi_and_does_not_retain_tensor_storage():
    import gc
    import weakref

    from cubic_serve.ops.cuda.compiled_artifact import _TENSOR_BINDINGS

    tensor = torch.arange(12).reshape(3, 4)
    first = bind_tensor(tensor)
    assert bind_tensor(tensor) is first
    tensor.transpose_(0, 1)
    second = bind_tensor(tensor)
    assert second != first and second.matches(tensor)
    tensor.set_(torch.arange(12).reshape(4, 3))
    third = bind_tensor(tensor)
    assert third != second and third.matches(tensor)
    count = len(_TENSOR_BINDINGS)
    reference = weakref.ref(tensor)
    del tensor
    gc.collect()
    assert reference() is None
    assert len(_TENSOR_BINDINGS) < count


def test_artifact_memo_rechecks_stages_metadata_specialization_aliases_and_weak_ownership(monkeypatch):
    import gc
    import weakref
    from collections import namedtuple
    from types import SimpleNamespace
    from triton.backends.compiler import GPUTarget
    from triton.compiler.compiler import CompiledKernel
    from cubic_serve.ops.cuda import compiled_artifact as module
    kernel = object.__new__(CompiledKernel)
    kernel.hash = 'compiled-key'
    kernel.asm = {name: name + '-contents' for name in ('source', *module.ARTIFACT_STAGES)}
    kernel.asm['cubin'] = kernel.kernel = b'actual-executable'
    Meta = namedtuple('Meta', ('target', *module.COMPILER_FLAGS))
    flags = dict(debug=False, enable_fp_fusion=True, enable_reflect_ftz=False,
                 num_ctas=1, num_stages=3, num_warps=4, sanitize_overflow=False)
    kernel.metadata = Meta(GPUTarget('cuda', 120, 32), **flags)
    kernel.src = SimpleNamespace(name='test', language='triton', ext='ttir',
                                 signature={'0': '*bf16'}, constants={'ROWS': 1}, attrs={})
    monkeypatch.setattr(module.cuda_driver, 'cuDriverGetVersion', lambda: (0, 13020))
    monkeypatch.setattr(module.torch.cuda, 'get_device_capability', lambda *a: (12, 0))
    calls = []
    original_hash = module._hash
    monkeypatch.setattr(module, '_hash', lambda value: calls.append(type(value)) or original_hash(value))
    def project(grid=(1, 1, 1)):
        return module.project_triton_artifact(kernel, operator_id='test.operator',
            device=torch.device('cuda', 0), launch_signature='fixed', grid=grid)
    first = project()
    count = len(calls)
    assert project() is first and len(calls) == count
    kernel.asm['ptx'] = 'different-ptx'
    assert project() != first
    second = project()
    kernel.metadata = kernel.metadata._replace(num_warps=8)
    assert project() != second
    third = project()
    kernel.src.constants['ROWS'] = 2
    assert project() != third
    assert project((2, 1, 1)) != project()
    with pytest.raises(ValueError):
        project([1, 1, 1])
    kernel.asm['cubin'] = b'alias-disagreement'
    with pytest.raises(ValueError, match='cubin views'):
        project()
    reference = weakref.ref(kernel)
    del project, kernel
    gc.collect()
    assert reference() is None
