#!/usr/bin/env python3
"""Build and validate a CUDA release offline, retaining resumable compiler work.

Run from the repository root:

    CUDA_HOME=/usr/local/cuda-13.0 ./venv/bin/python tools/build_release.py
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

from check_install import check_wheel, sha256_file

ROOT = Path(__file__).resolve().parents[1]


def available_memory() -> int:
    """Honor host availability and a container's remaining memory allowance."""
    values = [int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
              if line.startswith('MemAvailable:')]
    try:
        limit = Path('/sys/fs/cgroup/memory.max').read_text().strip()
        if limit != 'max':
            values.append(int(limit)-int(Path('/sys/fs/cgroup/memory.current').read_text()))
    except (OSError, ValueError):
        pass
    return max(0, min(values))


def payload_digest(wheel: Path) -> str:
    """Compare exact runtime/source/code bytes independently of ZIP timestamps."""
    digest = hashlib.sha256()
    with zipfile.ZipFile(wheel) as archive:
        for name in sorted(archive.namelist()):
            digest.update(name.encode()+b'\0')
            digest.update(hashlib.sha256(archive.read(name)).digest())
    return digest.hexdigest()


def dependency_paths(interpreter: Path, *, cwd: Path, env: dict[str, str]) -> list[str]:
    # A venv can inherit dependency directories through .pth files. Its declared
    # site.getsitepackages() alone omits those already-resolved import locations.
    result = subprocess.run([str(interpreter), '-c',
        'import json,sys;print(json.dumps([p for p in sys.path '
        'if p.endswith(("site-packages","dist-packages"))]))'],
        cwd=cwd, env=env, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


@contextmanager
def exclusive(path: Path):
    with path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f'Another release build owns {path}') from None
        yield


def publish(source: Path, target: Path):
    # Cross-filesystem staging is allowed. Only the final same-directory rename
    # replaces a known-good file; the release receipt is published last.
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix='.'+target.name,
                                     delete=False) as temporary:
        pending = Path(temporary.name)
    try:
        shutil.copyfile(source, pending)
        if sha256_file(pending) != sha256_file(source):
            raise RuntimeError('Publication copy failed its checksum')
        pending.chmod(0o644)
        os.replace(pending, target)
    finally:
        pending.unlink(missing_ok=True)


def verify_installation(interpreter: Path, wheel: Path, sandbox: Path, *, env, run, index: int):
    """Reuse dependency bytes without installing into or uninstalling from their owner."""
    interpreter = interpreter.expanduser().absolute()
    dependencies = dependency_paths(interpreter, cwd=sandbox.parent, env=env)
    run(f'venv-{index}', [str(interpreter), '-m', 'venv', '--without-pip', str(sandbox)])
    python = sandbox/'bin/python'
    library = subprocess.run([str(python), '-c',
        'import sysconfig;print(sysconfig.get_path("purelib"))'],
        cwd=sandbox.parent, env=env, capture_output=True, text=True, check=True)
    (Path(library.stdout.strip())/'publisher-dependencies.pth').write_text(
        '\n'.join(dependencies)+'\n')
    # The shared environment may already contain Cubic at the same version.
    # Ignore its installation; never uninstall or mutate that dependency owner.
    run(f'install-{index}', [sys.executable, '-m', 'pip', '--python', str(python),
        'install', '--no-index', '--no-deps', '--ignore-installed', str(wheel)], cwd=sandbox.parent)
    run(f'imports-{index}', [str(python), str(ROOT/'tools/check_install.py')], cwd=sandbox.parent)
    installed = json.loads((sandbox.parent/f'imports-{index}.log').read_text())
    if not Path(installed['package']).is_relative_to(sandbox):
        raise RuntimeError('Installation check imported outside its disposable environment')
    run(f'dependencies-{index}', [sys.executable, '-m', 'pip', '--python',
                                str(python), 'check'], cwd=sandbox.parent)
    shutil.rmtree(sandbox)
    return installed


def main() -> None:
    from build_cuda import _load_source
    cache_paths = _load_source('_cubic_release_cache_paths', ROOT/'src/cubic_serve/__init__.py')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'dist')
    parser.add_argument('--work-dir', type=Path,
                        default=cache_paths.cache_root()/'release')
    parser.add_argument('--jobs', type=int, help='Compiler workers; bounded by CPU and free RAM')
    parser.add_argument('--verify-python', type=Path, action='append',
                        help='Test in a disposable environment using this interpreter/dependencies; '
                             'repeat for a Python matrix (default: current interpreter)')
    args = parser.parse_args()
    output, work = (p.expanduser().resolve() for p in (args.output_dir, args.work_dir))
    cache = cache_paths.compiler_cache_root()/'wheel'
    work.mkdir(parents=True, exist_ok=True)
    with exclusive(work/'release.lock'):
        attempt = Path(tempfile.mkdtemp(prefix='attempt-', dir=work))
        report = {'status': 'running', 'attempt': str(attempt), 'stages': {}, 'python': sys.version,
                  'compiler_cache': str(cache), 'downloads': False, 'gpu_initialized': False}
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONPATH': '',
               'PYTHONDONTWRITEBYTECODE': '1', 'PIP_NO_INDEX': '1',
               'CUBIC_SERVE_CACHE_DIR': str(cache_paths.cache_root())}

        def run(stage: str, command: list[str], *, cwd: Path = ROOT, source_imports=False):
            print(f'{stage} ...', flush=True)
            started = time.monotonic()
            log = attempt/(stage+'.log')
            with log.open('w') as stream:
                child_env = {**env, 'PYTHONPATH': str(ROOT/'src')} if source_imports else env
                result = subprocess.run(command, cwd=cwd, env=child_env, stdout=stream,
                                        stderr=subprocess.STDOUT)
            report['stages'][stage] = {'seconds': time.monotonic()-started,
                                       'exit_code': result.returncode, 'log': str(log)}
            if result.returncode:
                raise RuntimeError(f'{stage} failed; see {log}')

        try:
            if sys.platform != 'linux' or os.uname().machine != 'x86_64':
                raise RuntimeError('Release builds require Linux x86-64')
            from packaging.requirements import Requirement
            from packaging.version import Version
            from setuptools.config.pyprojecttoml import read_configuration
            if Version(version('setuptools')) < Version('75'):
                raise RuntimeError('Release builds require setuptools>=75')
            project = read_configuration(ROOT/'pyproject.toml', expand=False)['project']
            missing = []
            for value in [*project['dependencies'], 'pytest>=8', 'pip>=23']:
                requirement = Requirement(value)
                if requirement.marker and not requirement.marker.evaluate():
                    continue
                try:
                    installed = version(requirement.name)
                except PackageNotFoundError:
                    installed = None
                if installed is None or installed not in requirement.specifier:
                    missing.append(f'{requirement} (installed: {installed})')
            if missing:
                raise RuntimeError('Install publisher dependencies first:\n'+'\n'.join(missing))
            # Fail before native compilation when a target interpreter lacks
            # the host headers/toolchain that Triton needs on a fresh machine.
            for index, interpreter in enumerate(args.verify_python or [Path(sys.executable)]):
                run(f'host-prerequisites-{index}', [str(interpreter.expanduser().absolute()),
                    '-c', 'import json;from cubic_serve.runtime.prerequisites import '
                    'check_triton_host_compilation;print(json.dumps(check_triton_host_compilation()))'],
                    source_imports=True)
            cuda = Path(env.get('CUDA_HOME', '/usr/local/cuda-13.0')).resolve()
            env['CUDA_HOME'] = str(cuda)
            run('compiler-preflight', [str(cuda/'bin/nvcc'), '--version'])
            if 'release 13.0' not in (attempt/'compiler-preflight.log').read_text():
                raise RuntimeError('The artifact baseline requires CUDA 13.0')
            run('assembler-preflight', [str(cuda/'bin/ptxas'), '--version'])
            memory = available_memory()
            if memory < 3*2**30:
                raise RuntimeError('Release compilation requires at least 3 GiB available host RAM')
            bound = min(64, os.cpu_count() or 1, max(1, (memory-2*2**30)//(512*2**20)))
            if args.jobs is not None and not 1 <= args.jobs <= 64:
                raise RuntimeError('--jobs must be between 1 and 64')
            jobs = min(args.jobs or 8, bound)
            env['CUBIC_CUDA_BUILD_JOBS'] = str(jobs)
            git = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True,
                                 text=True)
            report.update(source_commit=git.stdout.strip() if git.returncode == 0 else None,
                          jobs=jobs, available_host_bytes=memory,
                          dependencies={Requirement(x).name: version(Requirement(x).name)
                                        for x in project['dependencies']
                                        if not Requirement(x).marker
                                        or Requirement(x).marker.evaluate()})
            dirty = subprocess.run(['git', 'status', '--porcelain'], cwd=ROOT,
                                   capture_output=True, text=True)
            report['source_changes'] = dirty.stdout.splitlines() if dirty.returncode == 0 else None
            epoch = subprocess.run(['git', 'log', '-1', '--format=%ct'], cwd=ROOT,
                                   capture_output=True, text=True)
            env.setdefault('SOURCE_DATE_EPOCH', epoch.stdout.strip() if epoch.returncode == 0
                           else '315532800')
            report['source_date_epoch'] = env['SOURCE_DATE_EPOCH']
            tests = ['tests/packaging', 'tests/ops/test_native_bundle.py']
            run('contracts', [sys.executable, '-m', 'pytest', '-q', *tests], source_imports=True)
            source = attempt/'source'; source.mkdir()
            native = attempt/'wheel'; native.mkdir()
            run('sdist', [sys.executable, '-c',
                          'import sys;sys.path.insert(0,"tools");'
                          'import cubic_build_backend as b;b.build_sdist(sys.argv[1])', str(source)])
            archives = list(source.glob('*.tar.gz'))
            if len(archives) != 1:
                raise RuntimeError('Expected exactly one source distribution')
            run('wheel-from-sdist', [sys.executable, '-m', 'pip', 'wheel', '--no-index',
                '--no-deps', '--no-build-isolation', '--no-cache-dir',
                '--config-settings=cuda-bundle=true', '--wheel-dir', str(native), str(archives[0])])
            wheels = list(native.glob('*.whl'))
            if len(wheels) != 1:
                raise RuntimeError('Expected exactly one CUDA wheel')
            print('Validating every code object and packaged source ...', flush=True)
            result = check_wheel(wheels[0], ROOT)
            if not result['aot_recipes']:
                raise RuntimeError('Release wheel has no AOT CUDA kernels')
            report['wheel'] = {**result, 'payload_sha256': payload_digest(wheels[0])}
            report['sdist'] = {'file': archives[0].name, 'sha256': sha256_file(archives[0])}
            report['installed_python'] = []
            for index, interpreter in enumerate(args.verify_python or [Path(sys.executable)]):
                installed = verify_installation(interpreter, wheels[0], attempt/f'install-{index}',
                                                env=env, run=run, index=index)
                report['installed_python'].append(installed)
            output.mkdir(parents=True, exist_ok=True)
            for artifact in (archives[0], wheels[0]):
                publish(artifact, output/artifact.name)
            report['status'] = 'passed'
            receipt = attempt/'release.json'
            receipt.write_text(json.dumps(report, indent=2)+'\n')
            checksums = attempt/'SHA256SUMS'
            checksums.write_text(''.join(f'{sha256_file(p)}  {p.name}\n'
                                         for p in (archives[0], wheels[0])))
            publish(checksums, output/'SHA256SUMS')
            publish(receipt, output/'release.json')
            # Preserve logs and receipts; validated files now live in output.
            for artifact in (archives[0], wheels[0]):
                artifact.unlink()
            print(f"Built {output/wheels[0].name} ({result['bytes']/2**20:.2f} MiB)", flush=True)
            print(f'Build checks passed; receipt: {output/"release.json"}', flush=True)
        except Exception as error:
            report.update(status='failed', error=str(error))
            (attempt/'release.json').write_text(json.dumps(report, indent=2)+'\n')
            print(f'{error}\nBuild evidence retained in {attempt}. '
                  'Completed compiler units are reusable; fix the cause and rerun.', file=sys.stderr)
            raise SystemExit(1) from error


if __name__ == '__main__':
    main()
