"""Source-level budget for built-in model adapters."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path


_ROOT = Path(__file__).parents[2] / "src" / "cubic_serve" / "models"
_QWEN3_5_FILES = {"__init__.py", "model.py", "weights.py", "multimodal.py", "output.py"}
_FORBIDDEN_IMPORT_ROOTS = {
    "cubic_serve.api",
    "cubic_serve.executor",
    "cubic_serve.runtime",
    "torch.distributed",
}


class ModelAdapterLayoutTests(unittest.TestCase):
    def test_qwen3_5_owns_its_output_recipe(self) -> None:
        model_root = _ROOT / "qwen3_5"
        actual = {path.name for path in model_root.glob("*.py")}
        self.assertEqual(actual, _QWEN3_5_FILES)

    def test_qwen3_5_does_not_import_engine_or_serving_authority(self) -> None:
        for path in sorted((_ROOT / "qwen3_5").glob("*.py")):
            tree = ast.parse(path.read_text())
            imported = {
                node.module
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) and node.module is not None
            }
            imported.update(
                alias.name
                for node in ast.walk(tree)
                if isinstance(node, ast.Import)
                for alias in node.names
            )
            violations = {
                name
                for name in imported
                if any(name == root or name.startswith(root + ".") for root in _FORBIDDEN_IMPORT_ROOTS)
            }
            self.assertFalse(violations, f"{path.name} imports engine authority: {violations}")


if __name__ == "__main__":
    unittest.main()
