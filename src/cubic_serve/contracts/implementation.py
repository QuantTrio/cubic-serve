"""Implementation dependencies owned by an execution or calibration entry point.

Follow package imports, not the contents of the installed package directory.
Presentation is an explicit external effect: its implementation is not an input
to numerical or timing evidence. Actual native images and dynamically loaded
model/format implementations remain bound by their existing artifact contracts.
"""

from __future__ import annotations

import ast
import hashlib
from functools import cache
from importlib.util import resolve_name
from pathlib import Path

_PACKAGE = "cubic_serve"
_PACKAGE_ROOT = Path(__file__).parents[1]
_PRESENTATION = frozenset(("cubic_serve.runtime.logging",))
_NATIVE_SUFFIXES = frozenset((".cu", ".cuh", ".h", ".hpp", ".cpp", ".cc"))


class _Imports(ast.NodeVisitor):
    def __init__(self, package: str, rebound: set[str]):
        self.package = package
        self.rebound = rebound
        self.annotation_guards: set[str] = set()
        self.names: set[str] = set()

    def visit_Import(self, node: ast.Import) -> None:
        self.names.update(alias.name for alias in node.names)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        base = (resolve_name("." * node.level + (node.module or ""), self.package)
                if node.level else node.module or "")
        self.names.add(base)
        self.names.update(base + "." + alias.name for alias in node.names
                          if alias.name != "*")
        if base == "typing":
            self.annotation_guards.update(alias.asname or alias.name for alias in node.names
                                          if alias.name == "TYPE_CHECKING")

    def visit_If(self, node: ast.If) -> None:
        if (isinstance(node.test, ast.Name)
                and node.test.id in self.annotation_guards - self.rebound):
            for statement in node.orelse:
                self.visit(statement)
        else:
            self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        # Literal dynamic imports are dependencies too. Variable dispatch stays
        # owned by the selected model/format binding, not a guessed directory.
        name = node.func.id if isinstance(node.func, ast.Name) else (
            node.func.attr if isinstance(node.func, ast.Attribute) else "")
        if (name in ("import_module", "__import__") and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)):
            target = node.args[0].value
            if target.startswith(".") and name == "import_module":
                package = (node.args[1].value if len(node.args) > 1
                           and isinstance(node.args[1], ast.Constant)
                           and isinstance(node.args[1].value, str) else self.package)
                target = resolve_name(target, package)
            self.names.add(target)
        self.generic_visit(node)


@cache
def _imports(source: bytes, package: str) -> frozenset[str]:
    tree = ast.parse(source)
    rebound = {node.id for node in ast.walk(tree)
               if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)}
    rebound.update(node.arg for node in ast.walk(tree) if isinstance(node, ast.arg))
    rebound.update(node.name for node in ast.walk(tree)
                   if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            rebound.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            rebound.update(alias.asname or alias.name for alias in node.names
                           if node.module != "typing" or alias.name != "TYPE_CHECKING")
    visitor = _Imports(package, rebound)
    visitor.visit(tree)
    return frozenset(visitor.names)


def implementation_sources(
    *entries: str, package_root: Path | None = None,
) -> tuple[tuple[str, str], ...]:
    """Return a portable, deterministic dependency manifest without imports.

    Source bytes, including constants and control flow, are retained exactly;
    there is no AST rewriting or log-statement removal. Non-imported siblings
    cannot invalidate the entry point. Adjacent native sources conservatively
    accompany each imported owner; compiled artifacts still bind exact images.
    """
    root = _PACKAGE_ROOT if package_root is None else package_root

    def source_path(module: str) -> Path | None:
        if module != _PACKAGE and not module.startswith(_PACKAGE + "."):
            return None
        path = root.joinpath(*module.split(".")[1:])
        if (path / "__init__.py").is_file():
            return path / "__init__.py"
        if path.with_suffix(".py").is_file():
            return path.with_suffix(".py")
        return None

    pending = list(entries)
    seen: set[str] = set()
    sources: dict[str, str] = {}
    native_owners: set[Path] = set()
    while pending:
        module = pending.pop()
        if module in seen or module in _PRESENTATION:
            continue
        seen.add(module)
        path = source_path(module)
        if path is None:
            if module in entries:
                raise ValueError(f"implementation entry has no package source: {module}")
            continue
        raw = path.read_bytes()
        sources[path.relative_to(root).as_posix()] = hashlib.blake2s(raw).hexdigest()
        package = module if path.name == "__init__.py" else module.rpartition(".")[0]
        pending.extend(name for name in _imports(raw, package)
                       if name == _PACKAGE or name.startswith(_PACKAGE + "."))
        # Importing a submodule also executes its parent packages.
        parent = module.rpartition(".")[0]
        if parent:
            pending.append(parent)
        if path.parent not in native_owners:
            native_owners.add(path.parent)
            for native in path.parent.iterdir():
                if native.is_file() and native.suffix in _NATIVE_SUFFIXES:
                    sources[native.relative_to(root).as_posix()] = (
                        hashlib.blake2s(native.read_bytes()).hexdigest())
    return tuple(sorted(sources.items()))


@cache
def implementation_source(path: str) -> tuple[tuple[str, str], ...]:
    """Bind a callback's local dependencies, or its external owned source."""
    source = Path(path).resolve()
    try:
        relative = source.relative_to(_PACKAGE_ROOT.resolve()).with_suffix("")
    except ValueError:
        return ((source.as_posix(), hashlib.blake2s(source.read_bytes()).hexdigest()),)
    parts = relative.parts[:-1] if relative.name == "__init__" else relative.parts
    return implementation_sources(".".join((_PACKAGE, *parts)))
