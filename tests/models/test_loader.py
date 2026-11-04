from __future__ import annotations

import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import torch

from cubic_serve.contracts.execution import DecodeProvider, ExecutionMode
from cubic_serve.executor.runner import ReferenceExecutionContext
from cubic_serve.models import (
    ModelAdapterABIError,
    ModelAdapterSourcePolicyError,
    ModelAdapterValidationError,
    ModelBuildContext,
    load_model_adapter,
)

REPOSITORY = Path(__file__).resolve().parents[2]
TOY_ADAPTER = REPOSITORY / "examples" / "toy_adapter.py"


def _derived_adapter_source(describe_body: str) -> str:
    body = textwrap.indent(textwrap.dedent(describe_body).strip(), " " * 8)
    return (
        "from dataclasses import replace\n\n"
        "from examples.toy_adapter import ToyAdapter\n\n"
        "CUBIC_MODEL_ADAPTER_API_VERSION = 1\n\n"
        "class DerivedAdapter(ToyAdapter):\n"
        "    def describe(self):\n"
        f"{body}\n\n"
        "def create_model_adapter():\n"
        "    return DerivedAdapter()\n"
    )


class ModelAdapterLoaderTests(unittest.TestCase):
    def test_directory_closes_relative_import_digest_and_model_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "qwen_like"
            root.mkdir()
            weights = root / "weights.py"
            weights.write_text("FORMAT = 'v1'\n")
            (root / "model.py").write_text(
                "from examples.toy_adapter import ToyAdapter\n\n"
                "class Adapter(ToyAdapter):\n"
                "    pass\n"
            )
            (root / "__init__.py").write_text(
                "from .model import Adapter\n\n"
                "CUBIC_MODEL_ADAPTER_API_VERSION = 1\n"
                "MODEL = Adapter()\n\n"
                "def create_model_adapter():\n"
                "    return MODEL\n"
            )

            loaded = load_model_adapter(root)

            self.assertEqual(loaded.source_path, (root / "__init__.py").resolve())
            self.assertEqual(type(loaded.adapter).__name__, "Adapter")
            self.assertFalse(any(name.startswith(loaded.module_name) for name in sys.modules))
            context = ModelBuildContext(
                device=torch.device("cpu"),
                rank_id=0,
                world_size=1,
                owned_layer_ids=(0,),
            )
            with ReferenceExecutionContext():
                program = loaded.build_program(context)
            self.assertEqual(type(program).__name__, "ToyProgram")

            weights.write_text("FORMAT = 'v2'\n")
            changed = load_model_adapter(root)
            self.assertNotEqual(loaded.source_digest, changed.source_digest)

    def test_absolute_and_relative_paths_load_without_global_registration(self) -> None:
        before_path = tuple(sys.path)
        absolute = load_model_adapter(TOY_ADAPTER)
        relative_path = Path(os.path.relpath(TOY_ADAPTER, Path.cwd()))
        relative = load_model_adapter(relative_path)

        self.assertEqual(absolute.source_path, TOY_ADAPTER.resolve())
        self.assertEqual(relative.source_path, TOY_ADAPTER.resolve())
        self.assertEqual(tuple(sys.path), before_path)
        self.assertNotIn(absolute.module_name, sys.modules)
        self.assertNotIn(relative.module_name, sys.modules)
        self.assertEqual(absolute.reference_plan.mode, ExecutionMode.REFERENCE)
        self.assertEqual(
            len(absolute.reference_plan.bindings),
            len(absolute.descriptor.required_ops),
        )
        self.assertTrue(absolute.descriptor.model_spec.is_full_bf16)
        self.assertEqual(
            {item.name for item in absolute.descriptor.model_spec.modalities},
            {"text", "image", "video"},
        )

    def test_each_load_returns_an_independent_instance_and_module(self) -> None:
        first = load_model_adapter(TOY_ADAPTER)
        second = load_model_adapter(TOY_ADAPTER)

        self.assertIsNot(first, second)
        self.assertIsNot(first.adapter, second.adapter)
        self.assertNotEqual(first.module_name, second.module_name)
        self.assertEqual(first.source_digest, second.source_digest)

    def test_program_build_requires_the_reference_execution_context(self) -> None:
        loaded = load_model_adapter(TOY_ADAPTER)
        context = ModelBuildContext(
            device=torch.device("cpu"),
            rank_id=0,
            world_size=1,
            owned_layer_ids=(0,),
        )

        with self.assertRaisesRegex(RuntimeError, "ReferenceExecutionContext"):
            loaded.build_program(context)
        with ReferenceExecutionContext():
            program = loaded.build_program(context)

        self.assertEqual(type(program).__name__, "ToyProgram")

    def test_bad_api_version_factory_shape_and_identity_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bad_version = root / "bad_version.py"
            bad_version.write_text(
                "CUBIC_MODEL_ADAPTER_API_VERSION = 9\ndef create_model_adapter(): return object()\n"
            )
            bad_factory = root / "bad_factory.py"
            bad_factory.write_text(
                "CUBIC_MODEL_ADAPTER_API_VERSION = 1\n"
                "def create_model_adapter(config): return object()\n"
            )
            bad_identity = root / "bad_identity.py"
            bad_identity.write_text(
                "from examples.toy_adapter import ToyAdapter\n"
                "CUBIC_MODEL_ADAPTER_API_VERSION = 1\n"
                "MODEL = ToyAdapter()\n"
                "def create_model_adapter(): return ToyAdapter()\n"
            )

            with self.assertRaisesRegex(ModelAdapterABIError, "must equal 1"):
                load_model_adapter(bad_version)
            with self.assertRaisesRegex(ModelAdapterABIError, "exactly"):
                load_model_adapter(bad_factory)
            with self.assertRaisesRegex(ModelAdapterABIError, "exported MODEL"):
                load_model_adapter(bad_identity)

    def test_missing_reference_implementation_fails_sealing(self) -> None:
        source = _derived_adapter_source(
            """
            descriptor = super().describe()
            unknown = replace(
                descriptor.required_ops[0],
                semantic_op="fixture.input.missing",
            )
            return replace(
                descriptor,
                required_ops=(unknown,),
                reference_implementations=(),
            )
            """
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing_reference.py"
            path.write_text(source)
            with self.assertRaisesRegex(ModelAdapterValidationError, "missing reference"):
                load_model_adapter(path)

    def test_non_bf16_model_fails_closed(self) -> None:
        source = _derived_adapter_source(
            """
            from cubic_serve.contracts.model import ScalarDType

            descriptor = super().describe()
            cache = replace(descriptor.model_spec.cache, kv_dtype=ScalarDType.FP16)
            model = replace(descriptor.model_spec, cache=cache)
            return replace(descriptor, model_spec=model)
            """
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fp16_kv.py"
            path.write_text(source)
            with self.assertRaisesRegex(ModelAdapterValidationError, "full BF16"):
                load_model_adapter(path)

    def test_lora_weight_variant_fails_closed(self) -> None:
        source = _derived_adapter_source(
            """
            descriptor = super().describe()
            variant = replace(
                descriptor.weight_variant,
                adapter_id="fixture/lora",
                adapter_revision="1",
                adapter_scale=1.0,
            )
            return replace(descriptor, weight_variant=variant)
            """
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lora.py"
            path.write_text(source)
            with self.assertRaisesRegex(ModelAdapterValidationError, "non-base"):
                load_model_adapter(path)

    def test_speculative_decode_metadata_is_sealed_as_provider_capability(self) -> None:
        source = _derived_adapter_source(
            """
            from cubic_serve.contracts.execution import DecodeExtensionMeta, DecodeProvider

            descriptor = super().describe()
            speculative = DecodeExtensionMeta(
                provider=DecodeProvider.MTP,
                provider_version="fixture-v1",
                max_proposal_tokens=2,
                state_schema_id="fixture-state-v1",
            )
            return replace(descriptor, decode_extension=speculative)
            """
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "speculative.py"
            path.write_text(source)
            loaded = load_model_adapter(path)

        self.assertEqual(loaded.descriptor.decode_extension.provider, DecodeProvider.MTP)
        self.assertEqual(loaded.descriptor.decode_extension.max_proposal_tokens, 2)

    def test_compilation_apis_are_rejected_before_adapter_execution(self) -> None:
        forbidden_sources = {
            "direct": "import torch\ncompiled = torch.compile(lambda value: value)\n",
            "import_alias": "import torch as tensor_lib\ncompiled = tensor_lib.compile\n",
            "root_alias": (
                "import torch as tensor_lib\n"
                "other_name = tensor_lib\n"
                "compiled = other_name.compile\n"
            ),
            "from_alias": "from torch import compile as make_fast\n",
            "compiler_namespace": "import torch.compiler as compiler\n",
            "dynamo": "import torch._dynamo as dynamo\n",
            "inductor": "from torch import _inductor as backend\n",
            "aot_autograd": "from torch._functorch import aot_autograd as aot\n",
            "functorch_aot": "from functorch.compile import aot_function\n",
            "torchscript": "from torch.jit import script\n",
            "export": "import torch.export as export\n",
            "fx_capture": "from torch.fx import symbolic_trace as capture\n",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "executed"
            for name, forbidden in forbidden_sources.items():
                with self.subTest(name=name):
                    path = root / f"{name}.py"
                    path.write_text(
                        f"from pathlib import Path\nPath({str(marker)!r}).touch()\n{forbidden}"
                    )
                    with self.assertRaisesRegex(
                        ModelAdapterSourcePolicyError,
                        "compilation is unsupported.*eager-only",
                    ):
                        load_model_adapter(path)
                    self.assertFalse(marker.exists(), "source executed before admission")

    def test_graph_autotune_and_custom_kernel_apis_fail_before_source_execution(self) -> None:
        forbidden_sources = {
            "cuda_graph": "from torch.cuda import CUDAGraph\n",
            "graphed_callables": "from torch.cuda import make_graphed_callables\n",
            "cudnn_autotune": "import torch\nchoice = torch.backends.cudnn.benchmark\n",
            "autocast": "from torch import autocast\n",
            "custom_op": "from torch.library import custom_op\n",
            "native_extension": "from torch.utils.cpp_extension import load\n",
            "ops_dispatch": "import torch\ncustom = torch.ops.fixture\n",
            "sdpa": "from torch.nn.functional import scaled_dot_product_attention\n",
            "triton_autotune": "from triton import autotune\n",
            "flash_attention": "from flash_attn import flash_attn_func\n",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "executed"
            for name, forbidden in forbidden_sources.items():
                with self.subTest(name=name):
                    path = root / f"{name}.py"
                    path.write_text(
                        f"from pathlib import Path\nPath({str(marker)!r}).touch()\n{forbidden}"
                    )
                    with self.assertRaisesRegex(
                        ModelAdapterSourcePolicyError,
                        "unsupported eager reference API.*eager-only",
                    ):
                        load_model_adapter(path)
                    self.assertFalse(marker.exists(), "source executed before admission")

    def test_compile_names_in_comments_and_strings_do_not_trigger_policy(self) -> None:
        source = (
            "# torch.compile(lambda value: value) must remain disabled\n"
            "POLICY_NOTE = 'torch._dynamo and torch._inductor are unsupported'\n"
            + _derived_adapter_source("return super().describe()")
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy_note.py"
            path.write_text(source)
            loaded = load_model_adapter(path)

        self.assertTrue(loaded.descriptor.model_spec.is_full_bf16)


if __name__ == "__main__":
    unittest.main()
