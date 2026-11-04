#!/usr/bin/env python3
"""Check distribution contents or an installed package without allocating GPUs."""
from __future__ import annotations

import argparse
import ast
from email.parser import BytesParser
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import pkgutil
import re
import subprocess
import sys
import tempfile
import zipfile


def package_version(source: str) -> str:
    assignments = [node for node in ast.parse(source).body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == '__version__'
                           for target in node.targets)]
    if len(assignments) != 1 or not isinstance(value := ast.literal_eval(assignments[0].value), str):
        raise RuntimeError('Package must declare one literal __version__')
    return value


def stamp_package_version(source: str, version: str) -> str:
    if package_version(source) == version:
        return source
    node = next(node for node in ast.parse(source).body if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == '__version__'
                        for target in node.targets))
    lines = source.splitlines(keepends=True)
    lines[node.lineno - 1:node.end_lineno] = [f'__version__ = {version!r}\n']
    return ''.join(lines)


def cuda_variant(compiler: str) -> str:
    match = re.search(r'\brelease (\d+)\.(\d+)\b', compiler)
    if match is None:
        raise RuntimeError('Cannot identify the CUDA compiler release')
    return f'cu{int(match[1])}{int(match[2])}'


def native_file(path: Path) -> bool:
    return path.suffix in {'.cu', '.cuh', '.h', '.hpp', '.LICENSE'} or path.name in {
        'LICENSE', 'NOTICE.md', 'manifest.json', 'py.typed',
    }


def sha256_file(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def check_wheel(path: Path, source_root: Path | None) -> dict:
    from packaging.utils import canonicalize_name, parse_wheel_filename
    from packaging.version import Version

    with zipfile.ZipFile(path) as wheel:
        names = set(wheel.namelist())
        metadata = next(n for n in names if n.endswith('.dist-info/WHEEL'))
        tags = wheel.read(metadata).decode()
        info = BytesParser().parsebytes(wheel.read(metadata.removesuffix('WHEEL') + 'METADATA'))
        distribution, version, _, _ = parse_wheel_filename(path.name)
        if (distribution != 'cubic-serve' or canonicalize_name(info['Name']) != distribution
                or Version(info['Version']) != version
                or metadata != f'cubic_serve-{version}.dist-info/WHEEL'):
            raise RuntimeError('Wheel filename and distribution metadata differ')
        # PEP427 permits Python-independent native data in the purelib scheme
        # of a platform wheel; pip spreads that scheme into site-packages.
        if 'cubic_serve/cli.py' in names:
            prefix = ''
        else:
            candidates = [n[:-len('cubic_serve/cli.py')] for n in names
                          if n.endswith('.data/purelib/cubic_serve/cli.py')]
            if len(candidates) != 1:
                raise RuntimeError('Wheel has no unique Cubic package installation root')
            prefix = candidates[0]
        if 'Tag: py3-none-' not in tags:
            raise RuntimeError('Cubic wheel must be independent of the CPython extension ABI')
        if any(n.endswith(('.so', '.pyc')) for n in names):
            raise RuntimeError('Wheel contains a host/Python-specific build artifact')
        bundle_path = prefix + 'cubic_serve/_native/manifest.json'
        installed_version = package_version(wheel.read(prefix + 'cubic_serve/__init__.py').decode())
        if installed_version != str(version):
            raise RuntimeError('Wheel package and distribution versions differ')
        images = 0
        units = 0
        variant = None
        declared = set()
        if bundle_path in names:
            bundle = json.loads(wheel.read(bundle_path))
            variant = cuda_variant(bundle['compiler'])
            baseline = [80, 86, 87, 88, 89, 90, 100, 103, 110, 120]
            if (bundle['architectures'] != baseline
                    or set(bundle['catalog_keys']) != set(bundle['units'])):
                raise RuntimeError('Native bundle catalog/architecture coverage differs')
            for unit in bundle['units'].values():
                expected = {str(sm) for sm in baseline if sm >= unit['minimum_isa']}
                if set(unit['images']) != expected:
                    raise RuntimeError('Native recipe omits baseline device images')
                members = [(unit['ptx'], unit['ptx_sha256'])]
                members += [(v['path'], v['sha256']) for v in unit['images'].values()]
                for relative, digest in members:
                    if Path(relative).is_absolute() or '..' in Path(relative).parts:
                        raise RuntimeError('Native bundle path escapes its package')
                    name = prefix + 'cubic_serve/_native/' + relative
                    if hashlib.sha256(wheel.read(name)).hexdigest() != digest:
                        raise RuntimeError('Native image differs from its manifest digest')
                    declared.add(name)
                units += 1
                images += len(unit['images'])
            actual = {n for n in names if n.endswith(('.cubin', '.ptx'))}
            if declared != actual or 'Root-Is-Purelib: false' not in tags:
                raise RuntimeError('Native payload differs from its advertised catalog')
        elif ('Tag: py3-none-any' not in tags
              or any(n.endswith(('.cubin', '.ptx')) for n in names)):
            raise RuntimeError('Source wheel has an undeclared native payload')
        if version.local != variant:
            raise RuntimeError('Wheel CUDA version label differs from its native compiler')
        if any(Path(n).is_absolute() or '..' in Path(n).parts or not (
                n.startswith(metadata.removesuffix('WHEEL'))
                or n in declared
                or n.startswith(prefix + 'cubic_serve/')
                and (n.endswith('.py') or native_file(Path(n))))
               for n in names if not n.endswith('/')):
            raise RuntimeError('Distribution contains non-runtime repository data')
        native = sorted(n for n in names if n.startswith(prefix + 'cubic_serve/')
                        and not n.startswith(prefix + 'cubic_serve/_native/')
                        and native_file(Path(n)))
        if not native:
            raise RuntimeError('Distribution omits native source data')
        if source_root is not None:
            package = source_root / 'src/cubic_serve'
            for source in package.rglob('*'):
                if not source.is_file() or '__pycache__' in source.parts:
                    continue
                if source.suffix != '.py' and not native_file(source):
                    continue
                name = prefix + 'cubic_serve/' + source.relative_to(package).as_posix()
                expected = source.read_bytes()
                if source == package / '__init__.py':
                    if package_version(source.read_text()) != version.public:
                        raise RuntimeError('Wheel base version differs from source')
                    expected = stamp_package_version(source.read_text(), str(version)).encode()
                if name not in names or wheel.read(name) != expected:
                    raise RuntimeError(f'Distribution source mismatch: {name}')
        return {'wheel': path.name, 'sha256': sha256_file(path), 'bytes': path.stat().st_size,
                'version': str(version), 'cuda_variant': variant,
                'native_files': len(native), 'python_files': sum(n.endswith('.py') for n in names),
                'aot_recipes': units, 'cubin_images': images}


def check_git_free_startup() -> dict:
    """Reach normal startup/version logging without Git or loading model weights."""
    import cubic_serve

    with tempfile.TemporaryDirectory(prefix='cubic-git-free-') as empty:
        result = subprocess.run(
            [sys.executable, '-m', 'cubic_serve.cli', 'serve', 'installation-check',
             '--execution-mode', 'eager', '--port', '65536'],
            cwd=empty, env={**os.environ, 'PATH': empty, 'CUDA_VISIBLE_DEVICES': ''},
            capture_output=True, text=True,
        )
    output = result.stdout + result.stderr
    if (result.returncode != 2
            or f'cubic-serve version={cubic_serve.__version__}' not in output
            or 'error: --port must be between 0 and 65535' not in output):
        raise RuntimeError(f'Installed startup without Git failed:\n{output}')
    return {'git_free_startup': 'passed'}


def check_installed() -> dict:
    # Keep this set before importing any runtime dependency or walking packages.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import cubic_serve
    from cubic_serve.runtime.prerequisites import check_triton_host_compilation
    host = check_triton_host_compilation()
    root = Path(cubic_serve.__file__).resolve().parent
    imported = []
    for module in pkgutil.walk_packages(cubic_serve.__path__, 'cubic_serve.'):
        importlib.import_module(module.name)
        imported.append(module.name)
    for args in (['--version'], ['--help'], ['serve', '--help'], ['qualify', '--help']):
        subprocess.run([sys.executable, '-m', 'cubic_serve.cli', *args], check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if importlib.metadata.version('cubic-serve') != cubic_serve.__version__:
        raise RuntimeError('Installed metadata and package version differ')
    startup = check_git_free_startup()
    files = [p for p in root.rglob('*') if p.is_file() and native_file(p)]
    return {'python': sys.version.split()[0], 'package': str(root),
            'modules_imported': len(imported), 'native_files': len(files),
            'torch': importlib.metadata.version('torch'),
            'triton': importlib.metadata.version('triton'), **host, **startup}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheel', type=Path)
    parser.add_argument('--source-root', type=Path)
    args = parser.parse_args()
    result = check_wheel(args.wheel, args.source_root) if args.wheel else check_installed()
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
