"""Portability contracts are checked before building the release artifact."""
import ast
from pathlib import Path
import subprocess
import sys

import pytest

from cubic_serve import __version__
from cubic_serve._compat import StrEnum
from cubic_serve.cli import build_parser
from cubic_serve.contracts.model import ScalarDType
from cubic_serve.guardrail.replay import _decode
from typing_extensions import TypeAliasType

ROOT = Path(__file__).resolve().parents[2]


def test_runtime_sources_parse_on_python310():
    for source in (ROOT / 'src/cubic_serve').rglob('*.py'):
        ast.parse(source.read_text(), filename=str(source), feature_version=(3, 10))


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
