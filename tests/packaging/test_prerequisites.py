"""Fresh-machine host prerequisites fail before allocating model resources."""
import sys

import pytest

from cubic_serve import cli
from cubic_serve.runtime import prerequisites


def test_missing_python_headers_has_actionable_error(tmp_path, monkeypatch):
    monkeypatch.setattr(prerequisites, '_python_include', lambda: tmp_path)
    monkeypatch.setattr(prerequisites.shutil, 'which', lambda _: '/usr/bin/gcc')
    with pytest.raises(prerequisites.RuntimePrerequisiteError) as failure:
        prerequisites.require_host_build_prerequisites()
    text = str(failure.value)
    version = f'{sys.version_info.major}.{sys.version_info.minor}'
    assert f'Python {version} development headers (Python.h)' in text
    assert 'sudo apt-get update' in text
    assert f'sudo apt-get install python{version}-dev build-essential' in text
    assert 'Then retry your launch command.' in text
    assert 'launch.sh' not in text


def test_missing_compiler_does_not_pass_with_headers_present(tmp_path, monkeypatch):
    (tmp_path/'Python.h').touch()
    monkeypatch.setattr(prerequisites, '_python_include', lambda: tmp_path)
    monkeypatch.setattr(prerequisites.shutil, 'which', lambda _: None)
    with pytest.raises(prerequisites.RuntimePrerequisiteError, match='a C compiler'):
        prerequisites.require_host_build_prerequisites()


@pytest.mark.parametrize('command', ['serve', 'qualify'])
def test_missing_prerequisites_are_logged_before_model_loading(
    command, tmp_path, monkeypatch, capsys,
):
    monkeypatch.setattr(prerequisites, '_python_include', lambda: tmp_path)
    def forbidden(*args, **kwargs):
        raise AssertionError('Model/worker startup cannot precede prerequisite checks')
    monkeypatch.setattr(cli, '_serve_qwen3_5', forbidden)
    monkeypatch.setattr(cli, '_qualify_startup', forbidden)
    assert cli.main([command, '/not-loaded', '--devices', '0', '--tp-size', '1']) == 2
    text = capsys.readouterr().err
    assert '| ERROR |' in text and 'Python.h' in text and 'sudo apt-get update' in text


def test_host_compile_probe_bypasses_triton_cache(monkeypatch):
    from triton.runtime import build
    def forbidden(*args, **kwargs):
        raise AssertionError('A warm cache cannot replace installation validation')
    monkeypatch.setattr(build, 'get_cache_manager', forbidden)
    assert prerequisites.check_triton_host_compilation()['triton_host_compilation'] == 'passed'
