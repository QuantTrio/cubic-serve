"""Cache dependencies follow execution owners; presentation is outside them."""

import ast
from pathlib import Path

import pytest

from cubic_serve.contracts.implementation import implementation_sources


def _write(root, name, source):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    return path


@pytest.fixture
def package(tmp_path):
    root = tmp_path / "cubic_serve"
    _write(root, "__init__.py", "")
    _write(root, "runtime/__init__.py", "")
    _write(root, "runtime/logging.py", "def report(): return 'display v1'\n")
    _write(root, "engine/__init__.py", "")
    _write(root, "engine/math.py", "SCALE = 2\n")
    _write(root, "engine/entry.py", "from .math import SCALE\n"
           "from cubic_serve.runtime.logging import report\n"
           "def run(x): return x * SCALE\n")
    return root


def test_presentation_and_unrelated_sources_do_not_change_execution_identity(package):
    before = implementation_sources("cubic_serve.engine.entry", package_root=package)
    _write(package, "runtime/logging.py", "def report(): return 'different format and level'\n")
    _write(package, "unused.py", "raise RuntimeError('not an execution dependency')\n")
    after = implementation_sources("cubic_serve.engine.entry", package_root=package)
    assert before == after
    assert "runtime/logging.py" not in dict(after)
    assert "unused.py" not in dict(after)


@pytest.mark.parametrize("name,source", [
    ("engine/math.py", "SCALE = 3\n"),
    ("engine/entry.py", "from .math import SCALE\ndef run(x): return x + SCALE\n"),
    ("engine/__init__.py", "INITIAL_STATE = 1\n"),
    ("__init__.py", "ABI_VERSION = 2\n"),
])
def test_changed_execution_or_package_initialization_invalidates(package, name, source):
    before = implementation_sources("cubic_serve.engine.entry", package_root=package)
    _write(package, name, source)
    assert implementation_sources("cubic_serve.engine.entry", package_root=package) != before


def test_native_owner_changes_invalidate(package):
    _write(package, "engine/kernel.cu", "__global__ void run() {}\n")
    before = implementation_sources("cubic_serve.engine.entry", package_root=package)
    _write(package, "engine/kernel.cu", "__global__ void run() { __syncthreads(); }\n")
    assert implementation_sources("cubic_serve.engine.entry", package_root=package) != before


def test_package_source_takes_precedence_over_same_named_module(package):
    _write(package, "engine/math/__init__.py", "SCALE = 7\n")
    manifest = dict(implementation_sources("cubic_serve.engine.entry", package_root=package))
    assert "engine/math/__init__.py" in manifest
    assert "engine/math.py" not in manifest


def test_import_cycles_and_literal_dynamic_dependencies(package):
    _write(package, "engine/math.py", "from . import entry\nSCALE = 2\n")
    _write(package, "engine/dynamic.py", "VALUE = 7\n")
    with (package / "engine/entry.py").open("a") as stream:
        stream.write("from importlib import import_module\n"
                     "dynamic = import_module('cubic_serve.engine.dynamic')\n")
    manifest = implementation_sources("cubic_serve.engine.entry", package_root=package)
    assert "engine/dynamic.py" in dict(manifest)
    assert len(manifest) == len(dict(manifest))
    assert manifest == implementation_sources("cubic_serve.engine.entry", package_root=package)


def test_unknown_entry_and_invalid_source_fail_closed(package):
    with pytest.raises(ValueError, match="no package source"):
        implementation_sources("cubic_serve.missing", package_root=package)
    _write(package, "engine/math.py", "def invalid(\n")
    with pytest.raises(SyntaxError):
        implementation_sources("cubic_serve.engine.entry", package_root=package)


@pytest.mark.parametrize("guard, runtime", [
    ("from typing import TYPE_CHECKING\n", False),
    ("TYPE_CHECKING = True\n", True),
    ("from typing import TYPE_CHECKING\nTYPE_CHECKING = True\n", True),
    ("from typing import TYPE_CHECKING\nfrom .math import SCALE as TYPE_CHECKING\n", True),
])
def test_annotation_imports_are_separate_from_runtime_dependencies(package, guard, runtime):
    _write(package, "engine/annotation.py", "TYPE = 'annotation'\n")
    _write(package, "engine/entry.py", guard + "if TYPE_CHECKING:\n"
           "    from .annotation import TYPE\n")
    manifest = dict(implementation_sources("cubic_serve.engine.entry", package_root=package))
    assert ("engine/annotation.py" in manifest) == runtime


@pytest.mark.parametrize("entry", [
    "cubic_serve.runtime.quantized_graph",
    "cubic_serve.ops.cuda.launch_cost",
    "cubic_serve.quantization.common.marlin.cost",
])
def test_real_cache_domains_do_not_read_presentation_source(monkeypatch, entry):
    original = Path.read_bytes
    def read(path):
        assert path.parts[-2:] != ("runtime", "logging.py")
        return original(path)
    monkeypatch.setattr(Path, "read_bytes", read)
    manifest = dict(implementation_sources(entry))
    assert "runtime/logging.py" not in manifest
    assert "cli.py" not in manifest
    assert entry.removeprefix("cubic_serve.").replace(".", "/") + ".py" in manifest
    # Execution receipts use this source walker. Calibration caches bind their
    # compiled images instead; walking those modules must not invent an import
    # of the walker merely because this test calls it externally.
    assert ("contracts/implementation.py" in manifest) == (
        entry == "cubic_serve.runtime.quantized_graph")


def test_operator_calibration_does_not_depend_on_http_or_controller():
    manifest = dict(implementation_sources("cubic_serve.ops.cuda.launch_cost"))
    assert not any(path.startswith("api/") for path in manifest)
    assert "runtime/continuous_batch.py" not in manifest


def test_presentation_boundary_cannot_import_execution_components():
    from cubic_serve.runtime import logging as presentation

    tree = ast.parse(Path(presentation.__file__).read_bytes())
    for node in ast.walk(tree):
        names = ([alias.name for alias in node.names] if isinstance(node, ast.Import) else
                 [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
        assert not any(name == "cubic_serve" or name.startswith("cubic_serve.")
                       for name in names)
