from __future__ import annotations

import ast
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"

_COMPILE_PREFIXES = (
    "functorch.compile",
    "torch._dynamo",
    "torch._export",
    "torch._functorch",
    "torch._inductor",
    "torch.compile",
    "torch.compiler",
    "torch.export",
    "torch.jit",
)
_REFERENCE_FORBIDDEN_PREFIXES = (
    "flash_attn",
    "triton",
    "xformers.ops",
    "torch.cuda.CUDAGraph",
    "torch.cuda.ExternalStream",
    "torch.cuda.Stream",
    "torch.cuda.graph",
    "torch.cuda.make_graphed_callables",
    "torch.library",
    "torch.ops",
    "torch.utils.cpp_extension",
)
_REFERENCE_FORBIDDEN_EXACT = frozenset({"torch.nn.functional.scaled_dot_product_attention"})


def _python_sources() -> tuple[Path, ...]:
    return tuple(sorted(SOURCE_ROOT.rglob("*.py")))


def _is_prefix(value: str, prefixes: tuple[str, ...]) -> bool:
    return any(value == prefix or value.startswith(f"{prefix}.") for prefix in prefixes)


def _raw_qualified_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        owner = _raw_qualified_name(node.value)
        if owner is not None:
            return f"{owner}.{node.attr}"
    return None


def _aliases(tree: ast.Module) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                local = item.asname or item.name.partition(".")[0]
                aliases[local] = item.name if item.asname else local
        elif isinstance(node, ast.ImportFrom) and node.module:
            for item in node.names:
                if item.name != "*":
                    aliases[item.asname or item.name] = f"{node.module}.{item.name}"
    return aliases


def _qualified_name(node: ast.expr, aliases: dict[str, str]) -> str | None:
    raw = _raw_qualified_name(node)
    if raw is None:
        return None
    root, separator, tail = raw.partition(".")
    resolved = aliases.get(root, root)
    return f"{resolved}.{tail}" if separator else resolved


def _references(path: Path) -> tuple[tuple[ast.AST, str], ...]:
    tree = ast.parse(path.read_bytes(), filename=str(path))
    aliases = _aliases(tree)
    references: list[tuple[ast.AST, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            references.extend((node, item.name) for item in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            references.append((node, node.module))
            references.extend(
                (node, f"{node.module}.{item.name}") for item in node.names if item.name != "*"
            )
        elif isinstance(node, ast.Attribute):
            qualified = _qualified_name(node, aliases)
            if qualified is not None:
                references.append((node, qualified))
    return tuple(references)


def _location(path: Path, node: ast.AST, qualified_name: str) -> str:
    return f"{path.relative_to(PROJECT_ROOT)}:{getattr(node, 'lineno', 1)}: {qualified_name}"


class SourcePolicyTests(unittest.TestCase):
    def test_operator_seals_have_no_production_leaf_hydrator_or_blind_issuer(self) -> None:
        forbidden_names = {
            "_OPERATOR_EVIDENCE_TOKEN",
            "_issue_arithmetic_evidence",
            "_issue_composite_arithmetic_evidence",
            "_issue_operator_evidence",
        }
        violations: list[str] = []
        allowed_factory = Path("cubic_serve/guardrail/operator_evidence.py")
        for path in _python_sources():
            tree = ast.parse(path.read_bytes(), filename=str(path))
            aliases = _aliases(tree)
            relative = path.relative_to(SOURCE_ROOT)
            parents = {
                child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)
            }
            for node in ast.walk(tree):
                if isinstance(node, (ast.Name, ast.Attribute)):
                    name = _raw_qualified_name(node)
                    if name is not None and name.rpartition(".")[2] in forbidden_names:
                        violations.append(_location(path, node, name))
                if not isinstance(node, ast.Call):
                    continue
                called = _qualified_name(node.func, aliases)
                if called != "object.__new__" or len(node.args) != 1:
                    continue
                target = _qualified_name(node.args[0], aliases)
                seal_name = "" if target is None else target.rpartition(".")[2]
                owner = parents.get(node)
                owner_names: list[str] = []
                while owner is not None:
                    if isinstance(owner, ast.FunctionDef):
                        owner_names.append(owner.name)
                    owner = parents.get(owner)
                expected_owner = {
                    "CompositeArithmeticEvidenceSeal": "seal_composite_arithmetic_evidence",
                    "OperatorEvidenceSeal": "seal_operator_evidence",
                }.get(seal_name)
                if seal_name == "ArithmeticEvidenceSeal" or (
                    seal_name in {"CompositeArithmeticEvidenceSeal", "OperatorEvidenceSeal"}
                    and (
                        relative != allowed_factory
                        or owner_names[:1] != [expected_owner]
                    )
                ):
                    violations.append(_location(path, node, f"object.__new__({target})"))
        self.assertEqual(
            [],
            violations,
            "production forbids leaf hydration and exposes composite/operator hydration only "
            "inside their validated guardrail factories",
        )

    def test_v01_has_no_implicit_compile_entrypoint(self) -> None:
        violations: list[str] = []
        for path in _python_sources():
            for node, qualified_name in _references(path):
                if _is_prefix(qualified_name, _COMPILE_PREFIXES):
                    violations.append(_location(path, node, qualified_name))

        self.assertEqual([], violations, "v0.1 forbids implicit graph compilation")

    def test_nonoptimized_python_has_no_fused_graph_or_async_backend_entrypoint(self) -> None:
        violations: list[str] = []
        for path in _python_sources():
            relative = path.relative_to(SOURCE_ROOT / "cubic_serve")
            optimized_path = len(relative.parts) >= 2 and relative.parts[:2] in {
                ("ops", "cuda"),
                ("ops", "optimized"),
            }
            built_in_performance_path = relative in {
                Path("executor/model_runner.py"),
                Path("runtime/performance.py"),
            }
            if (
                optimized_path
                or built_in_performance_path
                or relative == Path("executor/cuda_graph.py")
            ):
                continue
            tree = ast.parse(path.read_bytes(), filename=str(path))
            aliases = _aliases(tree)
            for node, qualified_name in _references(path):
                if qualified_name in _REFERENCE_FORBIDDEN_EXACT or _is_prefix(
                    qualified_name, _REFERENCE_FORBIDDEN_PREFIXES
                ):
                    violations.append(_location(path, node, qualified_name))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    resolved_name = _qualified_name(node.func, aliases)
                    if resolved_name in {"torch.autocast", "torch.cuda.amp.autocast"}:
                        enabled = next(
                            (
                                keyword.value
                                for keyword in node.keywords
                                if keyword.arg == "enabled"
                            ),
                            None,
                        )
                        if not isinstance(enabled, ast.Constant) or enabled.value is not False:
                            violations.append(_location(path, node, resolved_name))
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        assigned_name = _qualified_name(target, aliases)
                        if assigned_name == "torch.backends.cudnn.benchmark" and (
                            not isinstance(node.value, ast.Constant)
                            or node.value.value is not False
                        ):
                            violations.append(_location(path, node, assigned_name))

        self.assertEqual(
            [],
            violations,
            "reference/core Python must not enter fused, graph, autotune, or async paths",
        )


if __name__ == "__main__":
    unittest.main()
