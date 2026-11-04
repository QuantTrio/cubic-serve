from __future__ import annotations

import importlib.util
import os
import pickle
import py_compile
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from cubic_serve.models import loader as m

PREFIX = "_cubic_serve_definition_"
ENTRY = (
    "from dataclasses import replace\n"
    "from cubic_serve.models.qwen3_5 import MODEL_DEFINITION as BASE\n"
    "def parse(raw):\n    from .payload import VALUE\n    return VALUE\n"
    "MODEL_DEFINITION = replace(BASE, parse_config=parse)\n"
)


class DefinitionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        (self.root / "__init__.py").write_text(ENTRY)
        (self.root / "payload.py").write_text("VALUE = 17\n")
        self.before_modules = {n for n in sys.modules if n.startswith(PREFIX)}
        self.before_meta = tuple(sys.meta_path)
        self.before_path = tuple(sys.path)

    def tearDown(self):
        self.assertEqual(tuple(sys.meta_path), self.before_meta)
        self.assertEqual(tuple(sys.path), self.before_path)
        self.assertEqual({n for n in sys.modules if n.startswith(PREFIX)}, self.before_modules)

    def test_two_loads_lazy_import_and_independent_cleanup(self):
        a = m.load_model_definition(self.root)
        b = m.load_model_definition(self.root)
        try:
            self.assertNotEqual(a.module_name, b.module_name)
            self.assertEqual(a.source_digest, b.source_digest)
            self.assertNotIn(a.module_name + ".payload", sys.modules)
            self.assertEqual(a.definition.parse_config({}), 17)
            a.close()
            a.close()
            self.assertNotIn(a.module_name, sys.modules)
            self.assertIn(b.module_name, sys.modules)
            self.assertEqual(b.definition.parse_config({}), 17)
            b.verify_source()
            with self.assertRaisesRegex(m.ModelAdapterLoadError, "closed"):
                a.verify_source()
        finally:
            a.close()
            b.close()

    def test_compiles_actual_snapshot_despite_valid_stale_pyc(self):
        f = self.root / "payload.py"
        stamp = f.stat()
        py_compile.compile(str(f), doraise=True)
        f.write_text("VALUE = 29\n")
        os.utime(f, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        cached_spec = importlib.util.spec_from_file_location("_stale_control", f)
        cached = importlib.util.module_from_spec(cached_spec)
        cached_spec.loader.exec_module(cached)
        self.assertEqual(cached.VALUE, 17)
        with m.load_model_definition(self.root) as a:
            self.assertEqual(a.definition.parse_config({}), 29)

    def test_changes_rejected_and_lazy_import_keeps_admitted_snapshot(self):
        with m.load_model_definition(self.root) as a:
            (self.root / "payload.py").write_text("VALUE = 29\n")
            with self.assertRaisesRegex(m.ModelAdapterLoadError, "changed"):
                a.verify_source()
            self.assertEqual(a.definition.parse_config({}), 17)

    def test_enter_failure_closes_owned_namespace(self):
        a = m.load_model_definition(self.root)
        (self.root / "new.py").write_text("NEW = True\n")
        with self.assertRaisesRegex(m.ModelAdapterLoadError, "changed"), a:
            pass

    def test_import_failure_and_invalid_definition_cleanup(self):
        for text in ('raise RuntimeError("private detail")\n', "MODEL_DEFINITION = object()\n"):
            (self.root / "__init__.py").write_text(text)
            with self.assertRaises(m.ModelAdapterLoadError):
                m.load_model_definition(self.root)

    def test_unused_source_is_checked_before_execution(self):
        (self.root / "payload.py").write_text("import torch\nx = torch.compile(lambda y:y)\n")
        with self.assertRaises(m.ModelAdapterSourcePolicyError):
            m.load_model_definition(self.root)

    def test_source_change_during_import_rejected(self):
        (self.root / "__init__.py").write_text(
            ENTRY
            + 'from pathlib import Path\n'
            + 'Path(__file__).with_name("payload.py").write_text("VALUE = 29\\n")\n'
        )
        with self.assertRaisesRegex(m.ModelAdapterLoadError, "changed"):
            m.load_model_definition(self.root)

    def test_single_file_and_existing_native_declaration_identity(self):
        from cubic_serve.models.qwen3_5 import MODEL_DEFINITION as BASE

        p = self.root / "standalone.py"
        p.write_text("from cubic_serve.models.qwen3_5 import MODEL_DEFINITION\n")
        with m.load_model_definition(p) as a:
            self.assertIs(a.definition, BASE)

    def test_explicit_file_entry_keeps_lazy_sibling_imports_in_its_owned_namespace(self):
        p = self.root / "standalone.py"
        p.write_text(ENTRY)
        with m.load_model_definition(p) as a:
            self.assertEqual(a.definition.parse_config({}), 17)
            self.assertTrue(a.module_name.startswith(PREFIX))
            (self.root / "payload.py").write_text("VALUE = 29\n")
            self.assertEqual(a.definition.parse_config({}), 17)
            with self.assertRaisesRegex(m.ModelAdapterLoadError, "changed"):
                a.verify_source()

    def test_removed_source_rejected(self):
        with m.load_model_definition(self.root) as a:
            (self.root / "payload.py").unlink()
            with self.assertRaisesRegex(m.ModelAdapterLoadError, "changed"):
                a.verify_source()


class DefinitionMetadataTests(unittest.TestCase):
    setUp = DefinitionTests.setUp
    tearDown = DefinitionTests.tearDown

    def test_recipe_identity_ignores_owned_namespace_and_binds_config_values(self):
        (self.root / "__init__.py").write_text(
            "from dataclasses import dataclass, replace\n"
            "from cubic_serve.models.qwen3_5 import MODEL_DEFINITION as BASE\n"
            "@dataclass(frozen=True)\nclass Config:\n    value: int\n"
            "MODEL_DEFINITION=replace(BASE,parse_config=lambda raw:Config(raw['value']))\n"
        )
        with (
            m.load_model_definition(self.root) as first,
            m.load_model_definition(self.root) as second,
        ):
            self.assertNotEqual(first.module_name, second.module_name)
            config = first.definition.parse_config({"value": 7})
            recipe = first.execution_recipe(config)
            self.assertEqual(
                recipe, second.execution_recipe(second.definition.parse_config({"value": 7})),
            )
            identity = first.program_identity("weights", recipe)
            self.assertEqual(identity, second.program_identity("weights", recipe))
            changed = first.execution_recipe(first.definition.parse_config({"value": 8}))
            self.assertNotEqual(identity, first.program_identity("weights", changed))
            self.assertNotEqual(identity, first.program_identity("other", recipe))

    def test_custom_recipe_reuses_existing_contract_and_rejects_invalid_recipe(self):
        (self.root / "__init__.py").write_text(
            ENTRY + "from cubic_serve.models.protocol import ProgramExecutionRecipe\n"
            "MODEL_DEFINITION=replace(MODEL_DEFINITION,program_recipe=lambda config:"
            "ProgramExecutionRecipe('custom','v1',settings=(('choice',str(config)),)))\n"
        )
        with m.load_model_definition(self.root) as owner:
            self.assertEqual(owner.execution_recipe(17).settings, (("choice", "17"),))
        (self.root / "__init__.py").write_text(
            ENTRY + "MODEL_DEFINITION=replace(MODEL_DEFINITION,program_recipe=lambda config:{})\n"
        )
        with (
            m.load_model_definition(self.root) as owner,
            self.assertRaisesRegex(m.ModelAdapterValidationError, "invalid type"),
        ):
            owner.execution_recipe(17)
        (self.root / "__init__.py").write_text(ENTRY)
        with (
            m.load_model_definition(self.root) as owner,
            self.assertRaisesRegex(m.ModelAdapterValidationError, "program recipe"),
        ):
            owner.execution_recipe(17)
        with self.assertRaisesRegex(m.ModelAdapterLoadError, "closed"):
            owner.execution_recipe(17)

    def test_custom_config_and_lazy_lambda_restore_before_metadata(self):
        (self.root / "__init__.py").write_text(
            "from dataclasses import dataclass, replace\n"
            "from cubic_serve.models.qwen3_5 import MODEL_DEFINITION as BASE\n"
            "@dataclass(frozen=True)\nclass Config:\n    value: int\n"
            "def parse(raw):\n    from .payload import VALUE\n"
            "    return Config(raw['value'] + VALUE)\n"
            "MODEL_DEFINITION=replace(BASE,parse_config=parse,text_model=lambda:Config(23))\n"
        )
        owner = m.load_model_definition(self.root)
        try:
            admission = SimpleNamespace(
                config=owner.definition.parse_config({"value": 5}),
                model_definition_source=owner,
            )
            expected = pickle.dumps(admission.config)
            payload = pickle.dumps(owner.bind_metadata(admission))
            self.assertIs(admission.model_definition_source, owner)
        finally:
            owner.close()
        value = pickle.loads(payload)
        with value.model_definition_source as restored:
            self.assertIs(type(value), SimpleNamespace)
            self.assertEqual(value.config.value, 22)
            self.assertIs(type(value.config), type(restored.definition.parse_config({"value": 5})))
            self.assertEqual(pickle.dumps(value.config), expected)
            self.assertEqual(restored.definition.text_model().value, 23)

    def test_changed_source_rejected_before_entry_execution(self):
        marker = self.root / "entry-ran"
        (self.root / "__init__.py").write_text(
            f"from pathlib import Path\nPath({str(marker)!r}).touch()\n" + ENTRY
        )
        with m.load_model_definition(self.root) as owner:
            payload = pickle.dumps(owner.bind_metadata(SimpleNamespace(config=17)))
        marker.unlink()
        (self.root / "payload.py").write_text("VALUE = 29\n")
        with self.assertRaisesRegex(m.ModelAdapterLoadError, "changed"):
            pickle.loads(payload)
        self.assertFalse(marker.exists())

    def test_deserialization_failure_closes_restored_namespace(self):
        with m.load_model_definition(self.root) as owner:
            payload = pickle.dumps(owner)
        restored = pickle.loads(payload)
        with self.assertRaises(pickle.UnpicklingError):
            m._restore_definition_metadata(restored, b"invalid metadata")
        with self.assertRaisesRegex(m.ModelAdapterLoadError, "closed"):
            restored.verify_source()

    def test_active_namespace_and_foreign_owner_are_rejected(self):
        with m.load_model_definition(self.root) as owner:
            with self.assertRaisesRegex(m.ModelAdapterLoadError, "already in use"):
                pickle.loads(pickle.dumps(owner))
            with self.assertRaisesRegex(m.ModelAdapterLoadError, "different"):
                owner.bind_metadata(SimpleNamespace(model_definition_source=object()))
        with self.assertRaisesRegex(m.ModelAdapterLoadError, "closed"):
            pickle.dumps(owner)


class NativeDefinitionSourceTests(unittest.TestCase):
    setUp = DefinitionTests.setUp
    tearDown = DefinitionTests.tearDown

    def test_native_helpers_do_not_inherit_reference_only_ban(self):
        (self.root / "payload.py").write_text(
            "import torch\nVALUE = 17\ndef bind():\n"
            "    return torch.ops.cubic_example, torch.utils.cpp_extension\n"
        )
        with m.load_model_definition(self.root) as a:
            self.assertEqual(a.definition.parse_config({}), 17)

    def test_reference_ABI_still_rejects_native_helpers(self):
        for code in ("import torch\nx = torch.ops.cubic_example\n", "import triton\n"):
            (self.root / "__init__.py").write_text(code)
            with self.assertRaises(m.ModelAdapterSourcePolicyError):
                m.load_model_adapter(self.root)

    def test_compile_aliases_rejected_for_native_definition(self):
        for code in (
            "from torch import compile as c\nx=c(lambda y:y)\n",
            "import torch as t\nx=t.compile(lambda y:y)\n",
        ):
            (self.root / "payload.py").write_text(code)
            with self.assertRaisesRegex(
                m.ModelAdapterSourcePolicyError, "unsupported torch compilation"
            ):
                m.load_model_definition(self.root)


class PackageDefinitionTests(unittest.TestCase):
    setUp = DefinitionTests.setUp
    tearDown = DefinitionTests.tearDown

    def test_nested_and_namespace_packages_keep_lazy_snapshot_imports(self):
        block = self.root / "blocks"
        block.mkdir()
        (block / "worker.py").write_text(
            "def value():\n    from ..payload import VALUE\n    return VALUE\n"
        )
        (self.root / "__init__.py").write_text(
            ENTRY.replace("from .payload import VALUE", "from .blocks.worker import value").replace(
                "return VALUE", "return value()"
            )
        )
        for regular in (False, True):
            with self.subTest(regular_package=regular):
                if regular:
                    (block / "__init__.py").write_text("MARKER = 3\n")
                (self.root / "payload.py").write_text("VALUE = 17\n")
                with m.load_model_definition(self.root) as loaded:
                    self.assertNotIn(loaded.module_name + ".blocks.worker", sys.modules)
                    (self.root / "payload.py").write_text("VALUE = 29\n")
                    self.assertEqual(loaded.definition.parse_config({}), 17)
                    with self.assertRaisesRegex(m.ModelAdapterLoadError, "changed"):
                        loaded.verify_source()

    def test_nested_sources_are_checked_before_entry_side_effects(self):
        block = self.root / "blocks"
        block.mkdir()
        (block / "unused.py").write_text("from torch import compile as c\nx=c(int)\n")
        marker = self.root / "ran.txt"
        (self.root / "__init__.py").write_text(
            f"from pathlib import Path\nPath({str(marker)!r}).touch()\n" + ENTRY
        )
        with self.assertRaises(m.ModelAdapterSourcePolicyError):
            m.load_model_definition(self.root)
        self.assertFalse(marker.exists())

    def test_explicit_file_entry_does_not_execute_parent_init(self):
        (self.root / "__init__.py").write_text("raise RuntimeError('wrong entry')\n")
        entry = self.root / "adapter.py"
        entry.write_text(ENTRY)
        with m.load_model_definition(entry) as loaded:
            self.assertEqual(loaded.definition.parse_config({}), 17)

    def test_hf_extensionless_blob_symlinks_preserve_python_module_names(self):
        blobs = self.root / "blobs"
        blobs.mkdir()
        for filename in ("__init__.py", "payload.py"):
            original = self.root / filename
            blob = blobs / filename.removesuffix(".py")
            blob.write_bytes(original.read_bytes())
            original.unlink()
            original.symlink_to(blob)
        with m.load_model_definition(self.root) as loaded:
            self.assertEqual(loaded.source_path.name, "__init__.py")
            self.assertEqual(loaded.definition.parse_config({}), 17)
            (blobs / "payload").write_text("VALUE = 29\n")
            with self.assertRaisesRegex(m.ModelAdapterLoadError, "changed"):
                loaded.verify_source()

    def test_invalid_python_fails_with_loader_error_without_namespace_leak(self):
        (self.root / "payload.py").write_text("def invalid(\n")
        with self.assertRaises(m.ModelAdapterLoadError):
            m.load_model_definition(self.root)

    def test_new_files_cannot_enter_an_admitted_namespace(self):
        (self.root / "__init__.py").write_text(ENTRY.replace(".payload", ".later"))
        with m.load_model_definition(self.root) as loaded:
            (self.root / "later.py").write_text("VALUE = 29\n")
            with self.assertRaises(ModuleNotFoundError):
                loaded.definition.parse_config({})
            self.assertNotIn(loaded.module_name + ".later", sys.modules)

    def test_python_package_precedes_same_named_module(self):
        (self.root / "payload").mkdir()
        (self.root / "payload" / "__init__.py").write_text("VALUE = 31\n")
        with m.load_model_definition(self.root) as loaded:
            self.assertEqual(loaded.definition.parse_config({}), 31)
