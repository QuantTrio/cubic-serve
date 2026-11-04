"""Cache placement is an engine policy shared by admission and compilers."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from cubic_serve import cache_root, compiler_cache_root
from cubic_serve.models.checkpoint import _inventory_cache_path
from cubic_serve.runtime.performance import _qualification_cache_root


@pytest.mark.parametrize('configured', [None, 'explicit'])
def test_metadata_and_receipt_cache_share_root(monkeypatch, tmp_path, configured):
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path / 'xdg'))
    monkeypatch.delenv('CUBIC_SERVE_CACHE_DIR', raising=False)
    expected = tmp_path / 'xdg' / 'cubic-serve'
    if configured:
        expected = tmp_path / configured
        monkeypatch.setenv('CUBIC_SERVE_CACHE_DIR', str(expected))
    assert cache_root() == expected
    assert _inventory_cache_path(tmp_path).parent == expected / 'checkpoints'
    assert _qualification_cache_root() == expected / 'qualification'


def test_fresh_process_sets_compiler_paths_before_torch_or_triton_import(tmp_path):
    env = dict(os.environ, CUBIC_SERVE_CACHE_DIR=str(tmp_path/'runtime'),
               CUBIC_SERVE_COMPILE_CACHE_DIR=str(tmp_path/'compiler'),
               TRITON_CACHE_DIR='/unused/legacy', CUDA_CACHE_PATH='/unused/legacy')
    code = '''import os,sys
from pathlib import Path
import cubic_serve
root=Path(os.environ['CUBIC_SERVE_CACHE_DIR'])/'compile'
assert 'torch' not in sys.modules and 'triton' not in sys.modules
assert os.environ['TRITON_CACHE_DIR']==str(root/'triton')
assert os.environ['CUDA_CACHE_PATH']==str(root/'cuda-driver')
assert os.environ['TORCHINDUCTOR_CACHE_DIR']==str(root/'inductor')
assert os.environ['TORCH_EXTENSIONS_DIR']==str(root/'torch-extensions')
assert os.environ['PYTORCH_KERNEL_CACHE_PATH']==str(root/'torch-kernels')
assert os.environ['TORCH_HOME']==str(root.parent/'torch')
'''
    subprocess.run([sys.executable, '-c', code], env=env, check=True, timeout=10)


def test_native_compiler_writes_under_common_root(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from cubic_serve.ops.cuda import native_compiler as native

    monkeypatch.setenv('CUBIC_SERVE_CACHE_DIR', str(tmp_path / 'chosen'))
    monkeypatch.setenv('CUBIC_SERVE_COMPILE_CACHE_DIR', str(tmp_path / 'compiler'))
    monkeypatch.setattr(native, 'installed_image', lambda *a, **kw: None)
    monkeypatch.setattr(native.torch.cuda, 'get_device_capability', lambda _: (8, 0))
    def run(args, **kwargs):
        if args[-1] == '--version':
            return SimpleNamespace(stdout='Cuda compilation tools, release 13.0', returncode=0)
        Path(args[-1]).write_bytes(b'.visible .entry test_entry(')
        return SimpleNamespace(stdout='', stderr='', returncode=0)
    monkeypatch.setattr(native.subprocess, 'run', run)
    monkeypatch.setattr(native, 'CudaModule', lambda *a, **kw: SimpleNamespace())
    native.compile_native.cache_clear()
    try:
        _, entry = native.compile_native((('entry.cu', b'kernel'),), 'cuda:0', 'test_cache')
        assert entry == 'test_entry'
        assert len(list((tmp_path / 'chosen' / 'compile' / 'test_cache').rglob('kernel.ptx'))) == 1
        assert not (tmp_path / 'compiler').exists()
    finally:
        native.compile_native.cache_clear()


def test_runtime_root_also_selects_compiler_directory(monkeypatch, tmp_path):
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path/'xdg'))
    monkeypatch.setenv('CUBIC_SERVE_CACHE_DIR', str(tmp_path/'runtime'))
    monkeypatch.delenv('CUBIC_SERVE_COMPILE_CACHE_DIR', raising=False)
    assert cache_root() == tmp_path/'runtime'
    assert compiler_cache_root() == tmp_path/'runtime/compile'


def test_relative_root_stays_fixed_for_workers_after_chdir(tmp_path):
    source = Path(__file__).resolve().parents[2]/'src'
    env = {**os.environ, 'CUBIC_SERVE_CACHE_DIR': './cache', 'PYTHONPATH': str(source)}
    elsewhere = tmp_path/'worker'; elsewhere.mkdir()
    code = r"""import os,sys,subprocess
from pathlib import Path
import cubic_serve
root=Path.cwd()/'cache'
assert cubic_serve.cache_root()==root
os.chdir('worker')
assert cubic_serve.cache_root()==root
assert Path(os.environ['TRITON_CACHE_DIR'])==root/'compile/triton'
subprocess.run([sys.executable,'-c',
    'from pathlib import Path;import cubic_serve,sys;assert cubic_serve.cache_root()==Path(sys.argv[1])',
    str(root)], check=True)
"""
    subprocess.run([sys.executable, '-c', code], env=env, cwd=tmp_path, check=True, timeout=15)


def test_triton_disk_cache_disappears_with_common_root(tmp_path):
    import shutil
    source = Path(__file__).resolve().parents[2]/'src'
    root = tmp_path/'chosen'
    outside = tmp_path/'legacy'
    env = {**os.environ, 'PYTHONPATH': str(source), 'CUBIC_SERVE_CACHE_DIR': str(root),
           'TRITON_CACHE_DIR': str(outside), 'CUBIC_SERVE_COMPILE_CACHE_DIR': str(outside)}
    probe = r"""import hashlib,sys
from pathlib import Path
import cubic_serve
from triton.runtime.cache import get_cache_manager
cache=get_cache_manager(hashlib.sha256(b'cache-placement-proof').hexdigest())
entry=cache.get_file('proof.bin')
assert (entry is not None)==(sys.argv[1]=='warm')
path=Path(cache.put(b'cache-placement-proof','proof.bin'))
assert path.is_relative_to(cubic_serve.cache_root())
"""
    for mode in ('cold', 'warm'):
        subprocess.run([sys.executable, '-c', probe, mode], env=env, check=True, timeout=15)
    assert not outside.exists()
    shutil.rmtree(root)
    subprocess.run([sys.executable, '-c', probe, 'cold'], env=env, check=True, timeout=15)
