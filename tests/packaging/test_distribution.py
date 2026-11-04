"""Portability contracts are checked before building the release artifact."""
import ast
import hashlib
import io
import runpy
import subprocess
import sys
from pathlib import Path

import pytest
from typing_extensions import TypeAliasType

from cubic_serve import __version__
from cubic_serve._compat import StrEnum
from cubic_serve.cli import build_parser
from cubic_serve.contracts.model import ScalarDType
from cubic_serve.guardrail.replay import _decode

ROOT = Path(__file__).resolve().parents[2]


def test_runtime_sources_parse_on_python310():
    for source in (ROOT / 'src/cubic_serve').rglob('*.py'):
        ast.parse(source.read_text(), filename=str(source), feature_version=(3, 10))


@pytest.mark.parametrize('size', [0, 1, 262143, 262144, 262145, 800003])
@pytest.mark.parametrize('algorithm', ['blake2s', 'sha256'])
def test_file_digest_fallback_matches_content_hash(monkeypatch, tmp_path, size, algorithm):
    monkeypatch.delattr(hashlib, 'file_digest', raising=False)
    fallback = runpy.run_path(str(ROOT / 'src/cubic_serve/_compat.py'))['file_digest']
    data = (b'\x00\xffCubic\n' * ((size // 8) + 1))[:size]
    path = tmp_path / 'payload.bin'
    path.write_bytes(data)
    expected = hashlib.new(algorithm, data).digest()
    with path.open('rb') as stream:
        assert fallback(stream, algorithm).digest() == expected
    assert fallback(io.BytesIO(data), lambda: hashlib.new(algorithm)).digest() == expected


def test_file_digest_scratch_is_bounded_and_hash_errors_propagate(monkeypatch):
    monkeypatch.delattr(hashlib, 'file_digest', raising=False)
    fallback = runpy.run_path(str(ROOT / 'src/cubic_serve/_compat.py'))['file_digest']
    data = b'bounded scratch' * 100000

    class Stream:
        def __init__(self):
            self.source = io.BytesIO(data)
            self.sizes = []

        def readable(self):
            return True

        def readinto(self, buffer):
            self.sizes.append(len(buffer))
            return self.source.readinto(buffer)

    stream = Stream()
    assert fallback(stream, 'blake2s').digest() == hashlib.blake2s(data).digest()
    assert len(stream.sizes) > 2 and max(stream.sizes) == 256 * 1024
    monkeypatch.setattr(stream, 'readinto', lambda buffer: None)
    with pytest.raises(BlockingIOError):
        fallback(stream, 'blake2s')


def test_processor_file_hash_works_on_supported_python(tmp_path):
    from cubic_serve.models.qwen3_5.multimodal import _content_digest_file

    data = b'processor recipe\x00\xff' * 20000
    path = tmp_path / 'tokenizer.json'
    path.write_bytes(data)
    assert _content_digest_file(path) == 'b2s256:' + hashlib.blake2s(data).hexdigest()


def test_string_enum_keeps_wire_and_format_semantics():
    assert issubclass(ScalarDType, StrEnum)
    assert str(ScalarDType.BF16) == ScalarDType.BF16.value
    assert f'{ScalarDType.BF16}' == ScalarDType.BF16.value


def test_replay_decodes_backported_type_alias_without_weakening_validation():
    alias = TypeAliasType('Rows', tuple[int, ...])
    assert _decode(alias, [1, 2]) == (1, 2)


def test_build_tool_can_check_wheel_without_runtime_imports():
    result = subprocess.run([sys.executable, '-S', str(ROOT / 'tools/check_install.py'), '--help'],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_cli_version(capsys):
    with pytest.raises(SystemExit) as result:
        build_parser().parse_args(['--version'])
    assert result.value.code == 0
    assert capsys.readouterr().out.strip() == f'cubic {__version__}'


def test_startup_version_with_empty_path_and_outside_checkout(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'tools'))
    monkeypatch.setenv('PYTHONPATH', str(ROOT / 'src'))
    from check_install import check_git_free_startup

    assert check_git_free_startup() == {'git_free_startup': 'passed'}
