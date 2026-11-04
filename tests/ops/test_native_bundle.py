"""Installed code identity and device coverage must fail closed on corruption."""
import hashlib
import json

import pytest

from cubic_serve.ops.cuda.native_bundle import BUNDLE_SCHEMA, installed_image, source_key


def bundle(tmp_path):
    sources = {'entry.cu': b'CUDA recipe', 'format.cuh': b'format arithmetic'}
    key = source_key('test_operator', sources, minimum_isa=80)
    ptx, image = b'.visible .entry test(', b'actual cubin bytes'
    (tmp_path/'kernel.ptx').write_bytes(ptx)
    (tmp_path/'sm120.cubin').write_bytes(image)
    unit = {'namespace': 'test_operator', 'minimum_isa': 80, 'entry': 'test',
            'ptx': 'kernel.ptx', 'ptx_sha256': hashlib.sha256(ptx).hexdigest(),
            'images': {'120': {'path': 'sm120.cubin',
                                'sha256': hashlib.sha256(image).hexdigest()}},
            'compiler_metadata': {'compiler': 'CUDA 13.0', 'aot': True}}
    (tmp_path/'manifest.json').write_text(json.dumps(
        {'schema': BUNDLE_SCHEMA, 'architectures': [120], 'units': {key: unit}}))
    return sources


def test_installed_image_never_reads_a_runtime_cache(tmp_path, monkeypatch):
    sources = bundle(tmp_path)
    monkeypatch.setenv('CUBIC_SERVE_CACHE_DIR', str(tmp_path/'unused-runtime'))
    ptx, image, entry, metadata = installed_image(
        'test_operator', sources, minimum_isa=80, sm=120, root=tmp_path)
    assert ptx and image == b'actual cubin bytes' and entry == 'test' and metadata['aot']
    assert not (tmp_path/'unused-runtime').exists()


def test_only_uncovered_device_may_fall_back(tmp_path):
    sources = bundle(tmp_path)
    assert installed_image('test_operator', sources, minimum_isa=80, sm=130, root=tmp_path) is None
    with pytest.raises(RuntimeError, match='required operator recipe'):
        installed_image('test_operator', {**sources, 'format.cuh': b'changed arithmetic'},
                        minimum_isa=80, sm=120, root=tmp_path)


def test_corrupt_code_object_is_not_a_compile_cache_miss(tmp_path):
    sources = bundle(tmp_path)
    (tmp_path/'sm120.cubin').write_bytes(b'corrupt')
    with pytest.raises(RuntimeError, match='digest'):
        installed_image('test_operator', sources, minimum_isa=80, sm=120, root=tmp_path)


def test_source_extensions_require_explicit_complete_catalog(tmp_path):
    sources = bundle(tmp_path)
    path = tmp_path/'manifest.json'
    data = json.loads(path.read_text())
    data['catalog_keys'] = sorted(data['units'])
    data['allow_source_extensions'] = True
    path.write_text(json.dumps(data))
    from cubic_serve.ops.cuda.native_bundle import _manifest
    _manifest.cache_clear()
    extension = {**sources, 'format.cuh': b'new exact-source recipe'}
    assert installed_image('test_operator', extension, minimum_isa=80,
                           sm=120, root=tmp_path) is None
    # An advertised entry cannot be deleted and treated as an extension.
    data['units'].clear()
    path.write_text(json.dumps(data)); _manifest.cache_clear()
    with pytest.raises(RuntimeError, match='declared recipe catalog'):
        installed_image('test_operator', sources, minimum_isa=80, sm=120, root=tmp_path)


def test_advertised_source_extension_image_still_fails_closed(tmp_path):
    sources = bundle(tmp_path)
    path = tmp_path/'manifest.json'
    data = json.loads(path.read_text())
    data['catalog_keys'] = sorted(data['units'])
    data['allow_source_extensions'] = True
    next(iter(data['units'].values()))['images'].clear()
    path.write_text(json.dumps(data))
    from cubic_serve.ops.cuda.native_bundle import _manifest
    _manifest.cache_clear()
    with pytest.raises(RuntimeError, match='advertised device image'):
        installed_image('test_operator', sources, minimum_isa=80, sm=120, root=tmp_path)


def test_source_identity_includes_names_contents_namespace_and_instruction_contract():
    source = {'entry.cu': b'abc', 'header.cuh': b'def'}
    key = source_key('a', source, minimum_isa=80)
    assert key == source_key('a', dict(reversed(list(source.items()))), minimum_isa=80)
    assert key != source_key('b', source, minimum_isa=80)
    assert key != source_key('a', source, minimum_isa=90)
    assert key != source_key('a', {**source, 'header.cuh': b'deg'}, minimum_isa=80)
    with pytest.raises(ValueError):
        source_key('a', {**source, '../header': b'escape'}, minimum_isa=80)


def test_installed_path_does_not_invoke_nvcc(monkeypatch):
    from types import SimpleNamespace
    from cubic_serve.ops.cuda import native_compiler as compiler
    monkeypatch.setattr(compiler.torch.cuda, 'get_device_capability', lambda _: (12, 0))
    monkeypatch.setattr(compiler, 'installed_image', lambda *a, **k:
                        (b'ptx', b'cubin', 'entry', {'aot': True}))
    def forbidden(*args, **kwargs):
        raise AssertionError('An installed code object cannot invoke a compiler')
    monkeypatch.setattr(compiler.subprocess, 'run', forbidden)
    calls = []
    monkeypatch.setattr(compiler, 'CudaModule',
                        lambda *args, **kwargs: calls.append(kwargs) or SimpleNamespace())
    compiler.compile_native.cache_clear()
    try:
        _, entry = compiler.compile_native((('entry.cu', b'kernel'),), 'cuda:0', 'installed')
        assert entry == 'entry' and calls[0]['cubin'] == b'cubin'
    finally:
        compiler.compile_native.cache_clear()
