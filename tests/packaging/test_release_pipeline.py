"""Release failures must preserve usable checkpoints and exact byte evidence."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest


@pytest.fixture
def release(monkeypatch):
    tools = Path(__file__).resolve().parents[2]/'tools'
    monkeypatch.syspath_prepend(str(tools))
    spec = importlib.util.spec_from_file_location('release_pipeline_test', tools/'build_release.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_corrupt_publication_cannot_replace_existing_checkpoint(release, tmp_path, monkeypatch):
    source, target = tmp_path/'new.whl', tmp_path/'installed.whl'
    source.write_bytes(b'verified new wheel')
    target.write_bytes(b'known good wheel')
    copy = release.shutil.copyfile

    def corrupt(src, dst):
        copy(src, dst)
        Path(dst).write_bytes(b'incomplete or corrupt copy')

    monkeypatch.setattr(release.shutil, 'copyfile', corrupt)
    with pytest.raises(RuntimeError, match='checksum'):
        release.publish(source, target)
    assert target.read_bytes() == b'known good wheel'
    assert sorted(p.name for p in tmp_path.iterdir()) == ['installed.whl', 'new.whl']


def test_exact_payload_comparison_ignores_only_zip_container_details(release, tmp_path):
    def wheel(name, year, body=b'kernel', metadata=b'contract'):
        path = tmp_path/name
        with zipfile.ZipFile(path, 'w') as output:
            for member, data in (('kernel.cubin', body), ('manifest.json', metadata)):
                info = zipfile.ZipInfo(member, date_time=(year, 1, 1, 0, 0, 0))
                output.writestr(info, data)
        return release.payload_digest(path)

    original = wheel('a.whl', 2020)
    assert original == wheel('b.whl', 2026)
    assert original != wheel('c.whl', 2026, body=b'changed code')
    assert original != wheel('d.whl', 2026, metadata=b'changed contract')


def test_disposable_install_reuses_resolved_pth_dependency_directories(release, tmp_path):
    venv = tmp_path/'venv'
    subprocess.run([sys.executable, '-m', 'venv', '--without-pip', str(venv)], check=True)
    extra = tmp_path/'dependencies/site-packages'
    extra.mkdir(parents=True)
    site = venv/f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages'
    (site/'shared.pth').write_text(str(extra)+'\n')
    paths = release.dependency_paths(venv/'bin/python', cwd=tmp_path,
                                    env={**os.environ, 'PYTHONPATH': ''})
    assert str(site) in paths and str(extra) in paths


def test_publisher_cache_uses_common_root(release, tmp_path, monkeypatch):
    import build_cuda

    root, outside = tmp_path/'cache', tmp_path/'legacy'
    monkeypatch.setenv('CUBIC_SERVE_CACHE_DIR', str(root))
    monkeypatch.setenv('CUBIC_CUDA_BUILD_CACHE_DIR', str(outside))
    compiled = []

    def run(args):
        if args[-1] == '--version':
            return 'Cuda compilation tools, release 13.0'
        if args[-1] == '--list-gpu-code':
            return ' '.join(f'sm_{sm}' for sm in build_cuda.BASELINE_ARCHS)
        compiled.append(args)
        Path(args[-1]).write_bytes(b'.visible .entry cache_test(' if '--ptx' in args
                                  else b'test cubin')
        return ''

    monkeypatch.setattr(build_cuda, '_run', run)
    package = Path(__file__).resolve().parents[2]/'src/cubic_serve'
    unit = {'namespace': 'cache_test', 'sources': {'entry.cu': b'test kernel'},
            'minimum_isa': 80}
    build_cuda.build_catalog(package, tmp_path/'bundle', units=[unit])
    assert len(list((root/'compile/wheel').iterdir())) == 1
    assert not outside.exists()
    assert len(compiled) == 1 + len(build_cuda.BASELINE_ARCHS)
    compiled.clear()
    build_cuda.build_catalog(package, tmp_path/'warm_bundle', units=[unit])
    assert not compiled


@pytest.mark.parametrize('toolkit,label', [('13.0', 'cu130'), ('12.8', 'cu128'), ('13.1', 'cu131')])
def test_cuda_version_label_follows_selected_compiler(release, monkeypatch, toolkit, label):
    import build_cuda

    package = Path(__file__).resolve().parents[2]/'src/cubic_serve'
    monkeypatch.setenv('CUDA_HOME', '/selected/toolkit')
    calls = []

    def compiler(args):
        calls.append(args)
        return f'CUDA compilation tools, release {toolkit}, V{toolkit}.0'

    monkeypatch.setattr(build_cuda, '_run', compiler)
    base = build_cuda.distribution_version(package, bundled=False)
    assert not calls
    assert build_cuda.distribution_version(package, bundled=True) == f'{base}+{label}'
    assert calls == [['/selected/toolkit/bin/nvcc', '--version']]


def test_version_stamp_changes_only_the_declared_package_version(release):
    from check_install import package_version, stamp_package_version

    source = '"""Package."""\n__version__ = "1.2.3"\nvalue = "1.2.3"\n'
    assert stamp_package_version(source, '1.2.3') == source
    stamped = stamp_package_version(source, '1.2.3+cu128')
    assert package_version(stamped) == '1.2.3+cu128'
    assert stamped == '"""Package."""\n__version__ = \'1.2.3+cu128\'\nvalue = "1.2.3"\n'


def _versioned_source_wheel(path, *, version='1.2.3', package=None):
    root = f'cubic_serve-{version}.dist-info'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr(root+'/WHEEL', 'Wheel-Version: 1.0\nTag: py3-none-any\n')
        archive.writestr(root+'/METADATA', f'Name: cubic-serve\nVersion: {version}\n')
        archive.writestr('cubic_serve/__init__.py', package or f'__version__ = {version!r}\n')
        archive.writestr('cubic_serve/cli.py', '')
        archive.writestr('cubic_serve/py.typed', '')


def test_wheel_check_rejects_filename_and_metadata_version_mismatch(release, tmp_path):
    from check_install import check_wheel

    wheel = tmp_path/'cubic_serve-1.2.3-py3-none-any.whl'
    _versioned_source_wheel(wheel)
    assert check_wheel(wheel, None)['version'] == '1.2.3'
    wrong = wheel.with_name('cubic_serve-1.2.3+cu130-py3-none-any.whl')
    wheel.rename(wrong)
    with pytest.raises(RuntimeError, match='filename and distribution metadata'):
        check_wheel(wrong, None)


def test_wheel_check_rejects_package_version_mismatch(release, tmp_path):
    from check_install import check_wheel

    wheel = tmp_path/'cubic_serve-1.2.3-py3-none-any.whl'
    _versioned_source_wheel(wheel, package='__version__ = "0.0.0"\n')
    with pytest.raises(RuntimeError, match='package and distribution versions'):
        check_wheel(wheel, None)


@pytest.mark.parametrize('private_file', ['NOTES.md', 'cubic_serve/NOTES.md',
                                       'tools/local_helper.py', 'private/records.jsonl',
                                       'cubic_serve/private/records.jsonl'])
def test_wheel_check_rejects_private_engineering_files(release, tmp_path, private_file):
    from check_install import check_wheel

    wheel = tmp_path/'cubic_serve-1.2.3-py3-none-any.whl'
    _versioned_source_wheel(wheel)
    with zipfile.ZipFile(wheel, 'a') as archive:
        archive.writestr(private_file, 'private engineering data')
    with pytest.raises(RuntimeError, match='non-runtime repository data'):
        check_wheel(wheel, None)


def test_source_check_still_rejects_unrelated_package_edits(release, tmp_path):
    from check_install import check_wheel

    source = tmp_path/'source/src/cubic_serve'
    source.mkdir(parents=True)
    (source/'__init__.py').write_text('__version__ = "1.2.3"\nvalue = 7\n')
    wheel = tmp_path/'cubic_serve-1.2.3-py3-none-any.whl'
    _versioned_source_wheel(wheel, package='__version__ = "1.2.3"\nvalue = 8\n')
    with pytest.raises(RuntimeError, match='source mismatch'):
        check_wheel(wheel, tmp_path/'source')


@pytest.mark.parametrize('label', ['cu130', 'cu128'])
def test_cuda_wheel_label_must_match_actual_compiler(release, tmp_path, label):
    import hashlib
    from check_install import check_wheel

    version = f'1.2.3+{label}'
    path = tmp_path/f'cubic_serve-{version}-py3-none-linux_x86_64.whl'
    _versioned_source_wheel(path, version=version)
    baseline = [80, 86, 87, 88, 89, 90, 100, 103, 110, 120]
    payload = b'test code object'
    digest = hashlib.sha256(payload).hexdigest()
    unit = {'minimum_isa': 80, 'ptx': 'kernel.ptx', 'ptx_sha256': digest,
            'images': {str(sm): {'path': f'sm{sm}.cubin', 'sha256': digest} for sm in baseline}}
    # This fixture tests archive/metadata validation, not executable CUDA math.
    with zipfile.ZipFile(path, 'a') as archive:
        archive.writestr('cubic_serve/_native/kernel.ptx', payload)
        for sm in baseline:
            archive.writestr(f'cubic_serve/_native/sm{sm}.cubin', payload)
        archive.writestr('cubic_serve/_native/manifest.json', json.dumps({
            'architectures': baseline, 'compiler': 'CUDA tools, release 13.0, V13.0.0',
            'catalog_keys': ['test'], 'units': {'test': unit}}))
    # Replace the source-wheel WHEEL header without duplicate ZIP members.
    with zipfile.ZipFile(path) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    members[f'cubic_serve-{version}.dist-info/WHEEL'] = (
        b'Wheel-Version: 1.0\nTag: py3-none-linux_x86_64\nRoot-Is-Purelib: false\n')
    with zipfile.ZipFile(path, 'w') as archive:
        for name, body in members.items():
            archive.writestr(name, body)
    if label == 'cu130':
        assert check_wheel(path, None)['cuda_variant'] == label
    else:
        with pytest.raises(RuntimeError, match='label differs from its native compiler'):
            check_wheel(path, None)
