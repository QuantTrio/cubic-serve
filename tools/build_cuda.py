"""Build platform wheels containing CUDA code objects without initializing a GPU.

Operator catalogs own a finite set of arithmetic/tile recipes. Neither model
checkpoints, serving traces nor the user's runtime cache are build inputs.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

from setuptools.command.build_py import build_py
from setuptools.command.bdist_wheel import bdist_wheel
from check_install import cuda_variant, package_version, stamp_package_version

# Portable artifact coverage, not a device-dependent execution/tuning table.
BASELINE_ARCHS = (80, 86, 87, 88, 89, 90, 100, 103, 110, 120)


def _load_source(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot load build definition: {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    bytecode = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = bytecode
    return module


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _run(command: list[str], *, timeout: int = 120) -> str:
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError('CUDA wheel build failed: ' + result.stderr[-8000:])
    return result.stdout


def distribution_version(package: Path, *, bundled: bool) -> str:
    base = package_version((package / '__init__.py').read_text())
    if not bundled:
        return base
    if '+' in base:
        raise RuntimeError('The source version must not include a CUDA build label')
    nvcc = Path(os.environ.get('CUDA_HOME', '/usr/local/cuda-13.0')) / 'bin/nvcc'
    return f"{base}+{cuda_variant(_run([str(nvcc), '--version']))}"


def build_catalog(package: Path, destination: Path, *, units=None) -> dict:
    """Build exact-source images with a bounded, reusable publisher build cache."""
    import concurrent.futures
    import fcntl
    import time

    package = package.resolve()
    cache_paths = _load_source('_cubic_build_cache_paths', package/'__init__.py')
    contract = _load_source('_cubic_build_native_bundle', package/'ops/cuda/native_bundle.py')
    if units is None:
        sys.path.insert(0, str(package.parent))
        catalog = _load_source('cubic_serve.ops.cuda.native_catalog',
                               package/'ops/cuda/native_catalog.py')
        units = catalog.recipes(package)
    compiler_root = Path(os.environ.get('CUDA_HOME', '/usr/local/cuda-13.0'))/'bin'
    nvcc, ptxas = compiler_root/'nvcc', compiler_root/'ptxas'
    version = _run([str(nvcc), '--version'])
    if 'release 13.0' not in version:
        raise RuntimeError('Native wheel artifacts require CUDA 13.0')
    available = set(map(int, re.findall(r'sm_(\d+)', _run([str(nvcc), '--list-gpu-code']))))
    if set(BASELINE_ARCHS)-available:
        raise RuntimeError('CUDA compiler omits mandatory baseline architectures')
    torch_spec = importlib.util.find_spec('torch')
    if torch_spec is None or torch_spec.origin is None:
        raise RuntimeError('Building native CUDA artifacts requires Torch headers')
    includes = Path(torch_spec.origin).parent/'include'
    from importlib.metadata import version as distribution_version
    torch_version = distribution_version('torch')
    identity = _digest(json.dumps([version, torch_version, BASELINE_ARCHS,
                                  'ptx-O3-ptxas-O4-cxx17-v1']).encode())
    cache = cache_paths.compiler_cache_root()/'wheel'/identity
    cache.mkdir(parents=True, exist_ok=True)
    destination.mkdir(parents=True, exist_ok=True)
    jobs = int(os.environ.get('CUBIC_CUDA_BUILD_JOBS', min(8, os.cpu_count() or 1)))
    if not 1 <= jobs <= 64:
        raise RuntimeError('CUBIC_CUDA_BUILD_JOBS must be between 1 and 64')
    manifest = {'schema': contract.BUNDLE_SCHEMA, 'architectures': list(BASELINE_ARCHS),
                'compiler': version, 'units': {}, 'catalog_keys': [],
                'allow_source_extensions': True}

    def build_unit(unit):
        namespace, sources, minimum = unit['namespace'], unit['sources'], unit['minimum_isa']
        key = contract.source_key(namespace, sources, minimum_isa=minimum)
        cached = cache/key
        with (cache/(key+'.lock')).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            metadata = None
            try:
                candidate = json.loads((cached/'receipt.json').read_text())
                expected = {'kernel.ptx', *(f'sm{sm}.cubin' for sm in BASELINE_ARCHS
                                            if sm >= minimum)}
                if (not isinstance(candidate, dict)
                        or not isinstance(candidate.get('files'), dict)
                        or set(candidate['files']) != expected
                        or not isinstance(candidate.get('entry'), str) or not candidate['entry']
                        or not isinstance(candidate.get('compiler_metadata'), dict)
                        or candidate['compiler_metadata'].get('aot') is not True):
                    raise ValueError('Build cache receipt is incomplete')
                for name, digest in candidate['files'].items():
                    if _digest((cached/name).read_bytes()) != digest:
                        raise ValueError('Build cache digest differs')
                if candidate['source_key'] != key:
                    raise ValueError('Build cache source identity differs')
                metadata = candidate
            except (OSError, ValueError, KeyError, TypeError):
                pass
            if metadata is None:
                flags = ('--ptx', f'-arch=compute_{minimum}', '-std=c++17', '-O3',
                         '--expt-relaxed-constexpr')
                with tempfile.TemporaryDirectory(prefix=key+'.', dir=cache) as temporary:
                    folder = Path(temporary)
                    source_dir = folder/'source'; source_dir.mkdir()
                    for name, data in sources.items():
                        path = source_dir/name
                        path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
                    ptx = folder/'kernel.ptx'
                    _run([str(nvcc), *flags, '-I', str(source_dir), '-I', str(includes),
                          str(source_dir/'entry.cu'), '-o', str(ptx)])
                    entries = re.findall(rb'\.visible \.entry (\S+)\(', ptx.read_bytes())
                    if len(entries) != 1:
                        raise RuntimeError('Native catalog unit must expose exactly one entry')
                    for arch in BASELINE_ARCHS:
                        if arch >= minimum:
                            _run([str(ptxas), '-O4', f'--gpu-name=sm_{arch}', str(ptx),
                                  '-o', str(folder/f'sm{arch}.cubin')])
                    shutil.rmtree(source_dir)
                    metadata = {'source_key': key, 'entry': entries[0].decode(),
                                'files': {f.name: _digest(f.read_bytes()) for f in folder.iterdir()},
                                'compiler_metadata': {'compiler': version, 'flags': flags,
                                                      'assembly_flags': ('-O4',),
                                                      'torch': torch_version, 'aot': True}}
                    (folder/'receipt.json').write_text(json.dumps(metadata, sort_keys=True))
                    if cached.exists():
                        shutil.rmtree(cached)
                    # TemporaryDirectory cleanup tolerates the renamed source.
                    folder.rename(cached)
            target = destination/namespace/key
            target.mkdir(parents=True, exist_ok=True)
            for name in metadata['files']:
                output = target/name
                if not output.exists():
                    try:
                        os.link(cached/name, output)
                    except OSError:
                        shutil.copy2(cached/name, output)
            relative = Path(namespace)/key
            images = {str(arch): {'path': str(relative/f'sm{arch}.cubin'),
                                 'sha256': metadata['files'][f'sm{arch}.cubin']}
                      for arch in BASELINE_ARCHS if arch >= minimum}
            return key, {'namespace': namespace, 'minimum_isa': minimum,
                         'family': unit.get('family', namespace), 'entry': metadata['entry'],
                         'ptx': str(relative/'kernel.ptx'),
                         'ptx_sha256': metadata['files']['kernel.ptx'], 'images': images,
                         'compiler_metadata': metadata['compiler_metadata']}

    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        pending = set()
        iterator = iter(units)
        exhausted = False
        while pending or not exhausted:
            while len(pending) < 2*jobs and not exhausted:
                try:
                    pending.add(pool.submit(build_unit, next(iterator)))
                except StopIteration:
                    exhausted = True
            if not pending:
                continue
            done, pending = concurrent.futures.wait(
                pending, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                key, record = future.result()
                if key in manifest['units']:
                    raise RuntimeError('Duplicate source-owned native build recipe')
                manifest['units'][key] = record
                if len(manifest['units']) % 128 == 0:
                    print(f"CUDA bundle: {len(manifest['units'])} recipes built "
                          f"in {time.monotonic()-started:.1f}s", flush=True)
    if not manifest['units']:
        raise RuntimeError('Empty native catalog cannot produce a binary wheel')
    manifest['catalog_keys'] = sorted(manifest['units'])
    (destination/'manifest.json').write_text(json.dumps(manifest, sort_keys=True)+'\n')
    print(f"CUDA bundle complete: {len(manifest['units'])} recipes in "
          f"{time.monotonic()-started:.1f}s", flush=True)
    return manifest


class BuildPy(build_py):
    """Populate build_lib only; source checkouts and runtime caches stay untouched."""
    def run(self):
        super().run()
        package = Path(self.build_lib)/'cubic_serve'
        # Catalog imports can traverse runtime packages. Never ship interpreter
        # bytecode left by an earlier build or generated while emitting sources.
        for bytecode in package.rglob('__pycache__'):
            shutil.rmtree(bytecode)
        destination = package/'_native'
        if destination.exists():
            shutil.rmtree(destination)
        if self.editable_mode or os.environ.get('CUBIC_BUILD_CUDA') != '1':
            return  # A source/editable build cannot inherit a previous native bundle.
        version_file = package / '__init__.py'
        version_file.write_text(stamp_package_version(
            version_file.read_text(), self.distribution.get_version()))
        old = sys.dont_write_bytecode
        try:
            sys.dont_write_bytecode = True
            build_catalog(package, destination)
        finally:
            sys.dont_write_bytecode = old

    def get_source_files(self):
        # PEP 517 hooks must survive a wheel built from the source distribution.
        return [*super().get_source_files(), 'tools/build_cuda.py',
                'tools/cubic_build_backend.py']

    def get_outputs(self, include_bytecode=1):
        outputs = super().get_outputs(include_bytecode)
        root = Path(self.build_lib)/'cubic_serve/_native'
        if root.exists():
            outputs.extend(str(path) for path in root.rglob('*') if path.is_file())
        return list(dict.fromkeys(outputs))


class BinaryWheel(bdist_wheel):
    """CUDA code objects are platform artifacts with no Python-extension ABI."""
    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self):
        _, _, platform = super().get_tag()
        return 'py3', 'none', platform
