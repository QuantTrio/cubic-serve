from __future__ import annotations

import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from cubic_serve.cli import _device_memory_budgets, _parse_speculative_config, build_parser, main
from cubic_serve.contracts.model import WeightVariantMeta
from cubic_serve.runtime.speculative import DraftPolicyConfig

REPOSITORY = Path(__file__).resolve().parents[2]
TOY_ADAPTER = REPOSITORY / "examples" / "toy_adapter.py"


class CLITests(unittest.TestCase):
    def test_startup_version_does_not_require_git(self):
        from cubic_serve import __version__
        from cubic_serve.cli import _serve_model

        args = build_parser().parse_args([
            "serve", "/checkpoint", "--execution-mode", "eager", "--port", "65536",
        ])
        with patch("subprocess.run", side_effect=FileNotFoundError("git")) as run, \
                patch("cubic_serve.cli._LOGGER.info") as info:
            with self.assertRaisesRegex(ValueError, "--port must be between"):
                _serve_model(args)
        run.assert_not_called()
        self.assertEqual(info.call_args_list[1].args, ("cubic-serve version=%s", __version__))

    def test_serve_and_qualify_help_group_serving_and_development_options(self):
        for command in ("serve", "qualify"):
            output = io.StringIO()
            with redirect_stdout(output), self.assertRaises(SystemExit) as result:
                build_parser().parse_args([command, "--help"])
            self.assertEqual(result.exception.code, 0)
            text = output.getvalue()
            server = text.split("\nServer:\n")[1].split("\nModel and generation:\n")[0]
            self.assertIn("--host", server)
            self.assertIn("--port", server)
            self.assertIn("--served-model-name", server)
            self.assertIn("--log-config", server)
            self.assertIn("--api-key", server)
            debug = text.split("\nAdvanced, audit and debug:\n")[1]
            for flag in ("--startup-numerical-audit", "--execution-mode", "--operator-config"):
                self.assertIn(flag, debug)
                self.assertNotIn(flag, server)
            for title in ("Devices and parallelism", "Memory and cache", "Scheduling", "Multimodal input"):
                self.assertIn(f"\n{title}:\n", text)

    def test_api_key_is_optional_explicit_and_rejects_invalid_values_without_echo(self):
        with patch.dict("os.environ", {"CUBIC_SERVE_API_KEY": "unused", "VLLM_API_KEY": "unused"}):
            self.assertIsNone(build_parser().parse_args(["serve", "/checkpoint"]).api_key)
        args = build_parser().parse_args(["serve", "/checkpoint", "--api-key", "sk-test-secret"])
        self.assertEqual(args.api_key, "sk-test-secret")
        for key in ("", "secret with spaces", "secret\nkey", "秘密", "secret\x00key"):
            output = io.StringIO()
            with redirect_stderr(output), self.assertRaises(SystemExit) as result:
                build_parser().parse_args(["serve", "/checkpoint", "--api-key", key])
            self.assertEqual(result.exception.code, 2)
            if key:
                self.assertNotIn(key, output.getvalue())

    def test_served_model_name_accepts_aliases_and_rejects_empty_names(self):
        args = build_parser().parse_args([
            "serve", "/checkpoint", "--served-model-name", "my-model", "my-model-v1",
            "--port", "8001"])
        self.assertEqual(args.model, "/checkpoint")
        self.assertEqual(args.served_model_name, ["my-model", "my-model-v1"])
        self.assertEqual(args.port, 8001)
        self.assertIsNone(build_parser().parse_args(["serve", "/checkpoint"]).served_model_name)
        for name in ("", " ", " padded"):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as result:
                build_parser().parse_args(["serve", "/checkpoint", "--served-model-name", name])
            self.assertEqual(result.exception.code, 2)

    def test_serving_processes_default_to_four_threads_and_preserve_explicit_settings(self):
        import os

        probe = """
import json, sys
from types import SimpleNamespace
launch = sys.argv[1]
if launch == 'serve':
    sys.argv = ['cubic', 'serve']
    import cubic_serve.cli
import torch
from cubic_serve.runtime import quantized_graph, logging as presentation
from cubic_serve.executor import vision
frontend = torch.get_num_threads() if launch == 'serve' else None
observed = []
class StopProbe(RuntimeError):
    pass
def observe():
    observed.append(torch.get_num_threads())
    raise StopProbe()
presentation.configure_logging = lambda: None
quantized_graph.configure_logging = lambda: None
quantized_graph.configure_reference_math = observe
vision.configure_reference_math = observe
try:
    quantized_graph._worker(0, '', SimpleNamespace(draft_provider=None), None, None, 1,
        SimpleNamespace(send=lambda value: None), 10, None, None)
except StopProbe:
    pass
try:
    vision._encoder_rank(None, 0, 0, None, '', 1)
except StopProbe:
    pass
print(json.dumps({'frontend': frontend, 'workers': observed,
                  'cuda_initialized': torch.cuda.is_initialized()}))
"""
        for launch, settings, expected in (
            ("serve", {}, 4),
            ("direct", {}, 4),
            ("serve", {"OMP_NUM_THREADS": "2"}, 2),
            ("serve", {"MKL_NUM_THREADS": "3"}, 3),
            ("serve", {"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "3"}, 3),
        ):
            with self.subTest(launch=launch, settings=settings):
                env = {key: value for key, value in os.environ.items()
                       if key not in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")}
                env.update(settings)
                completed = subprocess.run(
                    [sys.executable, "-c", probe, launch], env=env,
                    capture_output=True, text=True,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertEqual(json.loads(completed.stdout), {
                    "frontend": expected if launch == "serve" else None,
                    "workers": [expected, expected], "cuda_initialized": False,
                })

    def test_serving_allocator_default_precedes_torch_and_preserves_explicit_settings(self):
        import os

        probe = """
import json, os, sys
sys.argv = ['cubic-serve', sys.argv[1]]
import cubic_serve.cli
import torch
print(json.dumps({
    'modern': os.environ.get('PYTORCH_ALLOC_CONF'),
    'legacy': os.environ.get('PYTORCH_CUDA_ALLOC_CONF'),
    'cuda_initialized': torch.cuda.is_initialized(),
}))
"""
        cases = (
            ("serve", {}, "expandable_segments:True", None),
            ("qualify", {}, "expandable_segments:True", None),
            ("serve", {"PYTORCH_ALLOC_CONF": "expandable_segments:False"},
             "expandable_segments:False", None),
            ("serve", {"PYTORCH_CUDA_ALLOC_CONF": ""}, None, ""),
        )
        for command, settings, modern, legacy in cases:
            with self.subTest(command=command, settings=settings):
                env = {k: v for k, v in os.environ.items()
                       if k not in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")}
                env.update(settings)
                completed = subprocess.run(
                    [sys.executable, "-c", probe, command], env=env,
                    capture_output=True, text=True, check=True,
                )
                observed = json.loads(completed.stdout)
                self.assertEqual(observed, {
                    "modern": modern, "legacy": legacy, "cuda_initialized": False,
                })

    def test_interrupted_qualification_is_not_success_and_restores_signals(self):
        import signal

        previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
        cleaned = []

        def interrupted(_args):
            try:
                signal.raise_signal(signal.SIGINT)
            finally:
                signal.raise_signal(signal.SIGINT)
                cleaned.append(True)

        with patch("cubic_serve.cli._qualify_startup", side_effect=interrupted):
            self.assertEqual(main(["qualify", "/checkpoint"]), 130)
        self.assertEqual(cleaned, [True])
        self.assertEqual({sig: signal.getsignal(sig) for sig in previous}, previous)

        with patch("cubic_serve.cli._qualify_startup") as qualify:
            self.assertEqual(main(["qualify", "/checkpoint"]), 0)
            qualify.assert_called_once()

    def test_startup_qualification_is_an_explicit_command_without_http(self):
        from cubic_serve.cli import _qualify_startup

        parser = build_parser()
        normal = parser.parse_args(["serve", "/checkpoint"])
        explicit = parser.parse_args(["qualify", "/checkpoint", "--devices", "4,5", "--tp-size", "2"])
        self.assertFalse(getattr(normal, "startup_qualification", False))
        self.assertTrue(explicit.startup_qualification)
        executor = MagicMock()
        with patch("cubic_serve.cli._text_graph_backend", return_value=(None, executor, "model")) as build, \
             patch("cubic_serve.cli._serve_model") as serve, \
             patch("cubic_serve.cli.socket.create_server") as bind:
            _qualify_startup(explicit)
        build.assert_called_once()
        executor.close.assert_called_once()
        serve.assert_not_called()
        bind.assert_not_called()

    def test_kv_storage_has_independent_formats_and_strict_true_false_rotation(self):
        from cubic_serve.cli import _kv_cache_storage
        from cubic_serve.runtime.cache.quantization import CacheDType, CacheGroupAxis

        parser = build_parser()
        defaults = parser.parse_args(["serve", "model"])
        self.assertEqual((defaults.k_cache_dtype, defaults.v_cache_dtype,
                          defaults.enable_kv_cache_rotation), ("auto", "auto", False))
        for key in ("fp8", "int8", "int6", "int4"):
            for value in ("auto", "fp8", "int8", "int6", "int4"):
                args = parser.parse_args(["serve", "model", "--k-cache-dtype", key,
                                          "--v-cache-dtype", value,
                                          "--enable-kv-cache-rotation", "true"])
                storage = _kv_cache_storage(args)
                self.assertEqual((storage.key.dtype, storage.value.dtype),
                                 (CacheDType(key), CacheDType(value)))
                self.assertEqual((storage.key.axis, storage.value.axis),
                                 (CacheGroupAxis.TOKENS, CacheGroupAxis.CHANNELS))
                self.assertTrue(storage.enable_rotation)
        for value in ("on", "off", "1", "0", "yes"):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser.parse_args(["serve", "model", "--enable-kv-cache-rotation", value])

    def test_structured_output_policy_is_explicit_and_strict(self):
        from cubic_serve.cli import _structured_outputs_kwargs

        args = build_parser().parse_args([
            "serve", "model", "--structured-outputs-config",
            '{"enable_in_reasoning":true,"disable_any_whitespace":true}',
        ])
        self.assertEqual(_structured_outputs_kwargs(args), {
            "structured_output_in_reasoning": True,
            "structured_output_disable_whitespace": True,
        })
        self.assertEqual(_structured_outputs_kwargs(SimpleNamespace()), {})
        for invalid in ('[]', '{"enable_in_reasoning":1}', '{"backend":"unknown"}',
                        '{"unknown":true}'):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                _structured_outputs_kwargs(SimpleNamespace(structured_outputs_config=invalid))

    def setUp(self):
        # Backend wiring tests do not initialize real CUDA contexts.
        memory_patch = patch("cubic_serve.cli._device_memory_budgets", return_value={})
        memory_patch.start()
        self.addCleanup(memory_patch.stop)
        self.model_config = SimpleNamespace(text_config=SimpleNamespace(dtype="bfloat16"))
        config_patch = patch(
            "cubic_serve.models.builtin.parse_builtin_model_config_json",
            return_value=self.model_config,
        )
        config_patch.start()
        self.addCleanup(config_patch.stop)
        admission_patch = patch("cubic_serve.models.builtin.admit_builtin_checkpoint")
        self.checkpoint_admission = admission_patch.start()
        self.addCleanup(admission_patch.stop)

        def frontend_definition(_config):
            from cubic_serve.models.qwen3_5 import multimodal as input_module
            from cubic_serve.models.qwen3_5.output import Qwen3_5OutputCodec
            from cubic_serve.models.qwen3_5.serving import admit_reference, load_input_encoder

            return SimpleNamespace(
                frontend_factories=lambda *, multimodal: (
                    input_module.QwenMultimodalProcessor if multimodal
                    else input_module.Qwen3_5TextProcessor, Qwen3_5OutputCodec,
                ),
                input_encoder=load_input_encoder, reference_admission=admit_reference,
                reference_input_encoder=lambda *a, **k: (
                    input_module.load_qwen3_5_vision_reference(*a, **k)
                ),
            )

        frontend_patch = patch(
            "cubic_serve.models.builtin.builtin_model_definition", side_effect=frontend_definition
        )
        frontend_patch.start()
        self.addCleanup(frontend_patch.stop)
        from cubic_serve.models.builtin import builtin_model_definition_for_type

        def registered_definition(model_type):
            native = builtin_model_definition_for_type(model_type)
            result = frontend_definition(self.model_config)
            if native.reference_admission is None:
                result.reference_admission = None
            return result

        registry_patch = patch(
            "cubic_serve.models.builtin.builtin_model_definition_for_type",
            side_effect=registered_definition,
        )
        registry_patch.start()
        self.addCleanup(registry_patch.stop)
        config_dict_patch = patch(
            "cubic_serve.models.builtin.parse_builtin_model_config", return_value=self.model_config
        )
        config_dict_patch.start()
        self.addCleanup(config_dict_patch.stop)

    def test_resource_controls_use_gib_and_reject_obsolete_fraction_flags(self):
        parser = build_parser()
        defaults = parser.parse_args(["serve", "/checkpoint"])
        self.assertEqual(defaults.mm_worker_layout, "colocated")
        self.assertEqual(parser.parse_args([
            "serve", "/checkpoint", "--mm-worker-layout", "separate"]).mm_worker_layout,
            "separate")
        self.assertEqual((defaults.gpu_memory_bytes, defaults.gpu_reserve_bytes,
                          defaults.multimodal_memory_bytes, defaults.mm_feature_cache_bytes),
                         (None, 1 << 30, 3 << 29, 1 << 28))
        args = parser.parse_args(["serve", "/checkpoint", "--gpu-memory-gib", "22",
                                  "--gpu-reserve-gib", "0.5", "--multimodal-memory-gib", "1.5",
                                  "--mm-preprocess-workers", "3", "--mm-feature-cache-gib", "0",
                                  "--video-fps", "0.5", "--video-max-frames", "16"])
        self.assertEqual((args.gpu_memory_bytes, args.gpu_reserve_bytes,
                          args.multimodal_memory_bytes, args.mm_feature_cache_bytes),
                         (22 << 30, 1 << 29, 3 << 29, 0))
        self.assertEqual((args.video_fps, args.video_max_frames), (0.5, 16))
        decimal = parser.parse_args(["serve", "/checkpoint", "--multimodal-memory-gib", "2.2"])
        self.assertEqual(decimal.multimodal_memory_bytes, 2362232012)
        for flag in ("--multimodal-memory-fraction", "--vision-memory-fraction", "--mm-feature-cache-mib"):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser.parse_args(["serve", "/checkpoint", flag, "0.1"])
        for flag in ("--gpu-memory-gib", "--gpu-reserve-gib", "--multimodal-memory-gib", "--mm-feature-cache-gib"):
            for invalid in ("nan", "inf", "-1"):
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    parser.parse_args(["serve", "/checkpoint", flag, invalid])

    def test_gpu_budget_seals_math_before_cuda_and_counts_each_physical_device_once(self):
        from cubic_serve.runtime.cache import DeviceMemorySnapshot

        args = build_parser().parse_args(["serve", "/checkpoint", "--gpu-memory-gib", "22"])
        events = []

        def snapshot(device, *, initialize_cuda):
            self.assertEqual(events[0], "math")
            self.assertFalse(initialize_cuda)
            events.append(device)
            return DeviceMemorySnapshot(device, 24 << 30, 24 << 30)

        with (
            patch("cubic_serve.executor.runner.configure_reference_math",
                  side_effect=lambda: events.append("math")),
            patch("cubic_serve.runtime.cache.read_cuda_memory_snapshot", side_effect=snapshot),
            patch("cubic_serve.runtime.cache.DeviceMemoryBudget.apply_allocator_limit") as apply,
        ):
            budgets = _device_memory_budgets(args, (2, 3, 6, 7, 3, 4))
        self.assertEqual(events, ["math", "cuda:2", "cuda:3", "cuda:6", "cuda:7", "cuda:4"])
        apply.assert_not_called()
        self.assertEqual(set(budgets), {2, 3, 6, 7, 4})
        self.assertTrue(all(b.limit_bytes == 22 << 30 for b in budgets.values()))

    def test_encoder_peak_input_is_reserved_on_its_physical_device_before_kv_admission(self):
        from cubic_serve.cli import _text_graph_backend

        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "config.json").write_text(json.dumps({
                "model_type": "qwen4_exp", "text_config": {"dtype": "bfloat16"}}))
            args = build_parser().parse_args([
                "serve", directory, "--tp-size", "4", "--devices", "2,3,6,7",
                "--device", "cuda:3", "--video-fps", "0.5", "--video-max-frames", "16",
                "--mm-worker-layout", "separate", "--cuda-graph-max-tokens", "512"])

            def encoder(*args, devices, **kwargs):
                self.assertEqual(devices, (3, 2, 6, 7))
                self.assertIs(kwargs["model_config"], self.model_config)
                return MagicMock(input_reserves_by_device={d: 8 << 20 for d in devices})

            with (
                patch("cubic_serve.models.qwen3_5.output.Qwen3_5OutputCodec"),
                patch("cubic_serve.executor.vision.load_qwen_vision_tp", side_effect=encoder),
                patch("cubic_serve.models.qwen3_5.multimodal.QwenMultimodalProcessor") as processor,
                patch("cubic_serve.runtime.quantized_graph.TextGraphExecutor") as executor,
                patch("cubic_serve.runtime.token_chat.TokenChatBackend"),
            ):
                executor.return_value.semantic_digest = "b2s256:" + "0" * 64
                _text_graph_backend(args, None)
                self.assertIs(
                    self.checkpoint_admission.call_args.kwargs["model_config"], self.model_config
                )
                self.assertIs(processor.call_args.kwargs["model_config"], self.model_config)
                self.assertEqual(executor.call_args.kwargs["input_encoder_reserves"],
                                 {d: 8 << 20 for d in (2, 3, 6, 7)})
                self.assertEqual(executor.call_args.kwargs["cuda_graph_max_tokens"], 512)
                policy = processor.call_args.kwargs["resource_policy"]
                self.assertEqual((policy.video_frames_per_second, policy.video_max_frames), (0.5, 16))

    def test_plain_flash_next_selects_its_model_graph_before_legacy_admission(self):
        from cubic_serve.cli import _serve_qwen3_5

        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "config.json").write_text(json.dumps({
                "model_type": "qwen4_exp", "text_config": {"dtype": "bfloat16"},
            }))
            args = build_parser().parse_args([
                "serve", directory, "--tp-size", "4", "--api-key", "local-test-secret",
            ])
            backend, executor = MagicMock(), MagicMock()
            executor.max_sequence_tokens = 262144
            with (
                patch("cubic_serve.cli.socket.create_server"),
                patch("cubic_serve.cli._text_graph_backend",
                      return_value=(backend, executor, "flash")) as factory,
                patch("cubic_serve.models.builtin.admit_qwen3_5_reference") as legacy,
                patch("cubic_serve.api.CubicASGIApp") as app,
                patch("uvicorn.Config"),
                patch("cubic_serve.cli._run_http_server"),
                patch("cubic_serve.cli._LOGGER.info") as info,
            ):
                _serve_qwen3_5(args)
            factory.assert_called_once_with(args, None)
            app.assert_called_once_with(backend, model_id="flash", served_model_name=None,
                                        api_key="local-test-secret")
            self.assertEqual(args.api_key, "local-test-secret")
            self.assertEqual(json.loads(info.call_args_list[0].args[1])["api_key"], "<redacted>")
            self.assertNotIn("local-test-secret", str(info.call_args_list))
            legacy.assert_not_called()
            executor.close.assert_called_once()

    def test_wna8_uses_common_precision_defaults_and_preserves_explicit_overrides(self):
        from cubic_serve.cli import _serve_operator_config

        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "config.json").write_text(
                json.dumps({"model_type": "qwen4_exp"})
            )
            parser = build_parser()
            args = parser.parse_args(["serve", directory, "--compute-mode", "wna8"])
            config = _serve_operator_config(args)
            self.assertEqual(
                (config.compute_mode, config.activation_dtype,
                 config.attention_dtype, config.softmax_dtype),
                ("wna8", "auto", "auto", "fp32"),
            )
            args = parser.parse_args([
                "serve", directory, "--compute-mode", "wna8",
                "--attention-dtype", "fp32",
            ])
            self.assertEqual(_serve_operator_config(args).attention_dtype, "fp32")

    def test_flash_next_graph_forwards_runtime_state_dtype(self):
        from cubic_serve.cli import _text_graph_backend

        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "config.json").write_text(json.dumps({
                "model_type": "qwen4_exp",
                "text_config": {"dtype": "bfloat16", "mamba_ssm_dtype": "float32"},
            }))
            for dtype in (None, "auto", "float32", "bfloat16"):
                args = build_parser().parse_args([
                    "serve", directory, "--execution-mode", "debug",
                    "--tp-size", "4", "--devices", "2,3,6,7",
                ] + (["--mamba-ssm-cache-dtype", dtype] if dtype else []))
                with (
                    self.subTest(dtype=dtype),
                    patch("cubic_serve.models.qwen3_5.output.Qwen3_5OutputCodec"),
                    patch("cubic_serve.models.qwen3_5.multimodal.QwenMultimodalProcessor"),
                    patch("cubic_serve.executor.vision.load_qwen_vision_tp") as vision,
                    patch("cubic_serve.runtime.quantized_graph.TextGraphExecutor") as executor,
                    patch("cubic_serve.runtime.token_chat.TokenChatBackend"),
                ):
                    executor.return_value.semantic_digest = "b2s256:" + "0" * 64
                    _text_graph_backend(args, None)
                    self.assertEqual(vision.call_args.kwargs["devices"], (2, 3, 6, 7))
                    self.assertEqual(
                        executor.call_args.kwargs["mamba_ssm_cache_dtype"], dtype or "bfloat16"
                    )

    def test_compute_mode_is_independent_and_part_of_operator_identity(self):
        from cubic_serve.ops.strategies import OperatorConfig

        parser = build_parser()
        self.assertEqual(parser.parse_args(["serve", "/checkpoint"]).compute_mode, "wna16")
        args = parser.parse_args(["serve", "/checkpoint", "--compute-mode", "wna8",
                                  "--execution-mode", "eager", "--dtype", "fp16"])
        self.assertEqual((args.compute_mode, args.execution_mode, args.dtype),
                         ("wna8", "eager", "fp16"))
        config = OperatorConfig.from_dict({"version": 1, "compute_mode": "wna8"})
        self.assertEqual(config.compute_mode, "wna8")
        self.assertNotEqual(config.digest, OperatorConfig().digest)
        with self.assertRaises(ValueError):
            OperatorConfig(compute_mode="int4")

    def test_device_pool_uses_only_the_requested_execution_ranks(self) -> None:
        from cubic_serve.cli import _parse_devices

        self.assertEqual(_parse_devices("0,1,2,3", 1), (0,))
        self.assertEqual(_parse_devices("3,2,1,0", 2), (3, 2))
        self.assertEqual(_parse_devices("0,1,2,3", 4), (0, 1, 2, 3))
        with self.assertRaisesRegex(ValueError, "requires 4 devices"):
            _parse_devices("0,1", 4)
        with self.assertRaisesRegex(ValueError, "distinct"):
            _parse_devices("0,0", 1)

    def test_quantized_graph_preserves_checkpoint_and_explicit_fp16(self) -> None:
        self.model_config.text_config.dtype = "float16"
        from cubic_serve.cli import _text_graph_backend, build_parser

        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "config.json").write_text(
                json.dumps({"text_config": {"dtype": "float16"}})
            )
            for dtype in ("auto", "fp16", "float16", "bf16"):
                with (
                    self.subTest(dtype=dtype),
                    patch("cubic_serve.models.qwen3_5.output.Qwen3_5OutputCodec") as codec,
                    patch("cubic_serve.models.qwen3_5.multimodal.QwenMultimodalProcessor"),
                    patch("cubic_serve.executor.vision.load_qwen_vision_tp"),
                    patch("cubic_serve.runtime.quantized_graph.TextGraphExecutor") as executor,
                    patch("cubic_serve.runtime.token_chat.TokenChatBackend"),
                ):
                    codec.return_value.eos_token_id = 1
                    codec.return_value.eos_token_ids = (1, 2)
                    executor.return_value.semantic_digest = "b2s256:" + "0" * 64
                    args = build_parser().parse_args(["serve", directory, "--dtype", dtype])
                    _text_graph_backend(args, None)
                    self.assertEqual(
                        executor.call_args.kwargs["compute_dtype"],
                        "float16" if dtype == "auto" else dtype,
                    )
                    self.assertEqual(executor.call_args.kwargs["eos_token_ids"], (1, 2))

    def test_numerical_acceptance_is_explicit_and_forwarded(self) -> None:
        from cubic_serve.cli import _text_graph_backend

        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "config.json").write_text(
                json.dumps({"text_config": {"dtype": "bfloat16"}})
            )
            for enabled in (False, True):
                with (
                    self.subTest(enabled=enabled),
                    patch("cubic_serve.models.qwen3_5.output.Qwen3_5OutputCodec") as codec,
                    patch("cubic_serve.models.qwen3_5.multimodal.QwenMultimodalProcessor"),
                    patch("cubic_serve.executor.vision.load_qwen_vision_tp"),
                    patch("cubic_serve.runtime.quantized_graph.TextGraphExecutor") as executor,
                    patch("cubic_serve.runtime.token_chat.TokenChatBackend"),
                ):
                    codec.return_value.eos_token_id = 1
                    codec.return_value.eos_token_ids = (1, 2)
                    executor.return_value.semantic_digest = "b2s256:" + "0" * 64
                    options = ["--startup-numerical-audit"] if enabled else []
                    args = build_parser().parse_args(["serve", directory, *options])
                    _text_graph_backend(args, None)
                    self.assertIs(executor.call_args.kwargs["startup_numerical_audit"], enabled)

    def test_repeated_interrupt_during_cleanup_is_quiet_and_restores_handler(self) -> None:
        script = """
import signal
from cubic_serve.cli import _serve_interrupts
original = signal.getsignal(signal.SIGINT)
with _serve_interrupts():
    try:
        signal.raise_signal(signal.SIGINT)
    finally:
        signal.raise_signal(signal.SIGINT)
        print('cleanup completed')
assert signal.getsignal(signal.SIGINT) == original
"""
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=10
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "cleanup completed")
        self.assertEqual(result.stderr, "")

    def test_http_shutdown_disconnects_before_waiting_and_ignores_repeat_signal(self) -> None:
        import asyncio
        import signal

        from cubic_serve.cli import _run_http_server

        events = []
        class BaseServer:
            def __init__(self, config):
                self.should_exit = False
                self.server_state = SimpleNamespace(connections=[
                    SimpleNamespace(transport=SimpleNamespace(close=lambda: events.append("close")))
                ])
            def handle_exit(self, sig, frame):
                events.append("signal")
                self.should_exit = True
            async def shutdown(self, sockets=None):
                events.append("shutdown")
            def run(self, sockets):
                self.handle_exit(signal.SIGINT, None)
                self.handle_exit(signal.SIGINT, None)
                asyncio.run(self.shutdown(sockets))

        with patch("uvicorn.Server", BaseServer):
            _run_http_server(None, MagicMock())
        self.assertEqual(events, ["signal", "close", "shutdown"])

    def test_console_switch_follows_successful_http_startup_and_summary(self) -> None:
        import asyncio
        from cubic_serve.cli import _run_http_server

        for started, exiting in ((True, False), (False, False), (True, True)):
            events = []

            class BaseServer:
                def __init__(self, config):
                    self.started, self.should_exit = started, exiting

                async def startup(self, sockets=None):
                    events.append("http-startup")

                def run(self, sockets):
                    asyncio.run(self.startup(sockets))

            with (
                self.subTest(started=started, exiting=exiting),
                patch("uvicorn.Server", BaseServer),
                patch("cubic_serve.runtime.logging.finish_log_startup",
                      side_effect=lambda: events.append("file-only")),
            ):
                _run_http_server(None, MagicMock(),
                                 startup_summary=lambda: events.append("summary"))
                self.assertEqual(events, ["http-startup", "summary", "file-only"]
                                 if started and not exiting else ["http-startup"])

    def test_serve_defaults_to_cuda_and_keeps_explicit_debug_modes(self) -> None:
        parser = build_parser()

        self.assertEqual(
            parser.parse_args(["serve", "/checkpoint"]).execution_mode,
            "cuda",
        )
        self.assertEqual(
            parser.parse_args(
                ["serve", "/checkpoint", "--execution-mode", "cuda"]
            ).execution_mode,
            "cuda",
        )
        for mode in ("eager", "debug"):
            self.assertEqual(
                parser.parse_args(
                    ["serve", "/checkpoint", "--execution-mode", mode]
                ).execution_mode,
                mode,
            )
        for obsolete in ("reference", "performance", "graph-debug"):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser.parse_args(["serve", "/checkpoint", "--execution-mode", obsolete])

    def test_dynamic_policy_is_typed_and_rejects_unknown_parameters(self) -> None:
        payload = {"method": "mtp", "num_speculative_tokens": 1,
                   "mode": "dynamic", "max_tokens": 7,
                   "policy": {"cost_sample_window": 1, "decision_tokens": 32}}
        parsed = _parse_speculative_config(json.dumps(payload))
        self.assertEqual(parsed.policy.cost_sample_window, 1)
        self.assertEqual(parsed.policy.decision_tokens, 32)
        for invalid in ({"cost_sample_window": True}, {"cost_ema_alpha": float("nan")},
                        {"unknown": 1}, {"history_decay": 1.0}, {"prior_success": "4"}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                _parse_speculative_config(json.dumps(dict(payload, policy=invalid)))

    def test_mtp_fixed_depth_is_one_parameterized_domain(self) -> None:
        for draft_count in (1, 2, 3, 5, 8):
            parsed = _parse_speculative_config(
                json.dumps({"method": "mtp", "num_speculative_tokens": draft_count})
            )
            self.assertEqual(
                (parsed.initial_tokens, parsed.max_tokens, parsed.dynamic),
                (draft_count, draft_count, False),
            )
        dynamic = _parse_speculative_config(
            json.dumps(
                {
                    "method": "mtp",
                    "num_speculative_tokens": 1,
                    "max_tokens": 7,
                    "mode": "dynamic",
                }
            )
        )
        self.assertEqual(
            (dynamic.initial_tokens, dynamic.max_tokens, dynamic.dynamic),
            (1, 7, True),
        )
        with self.assertRaisesRegex(ValueError, "positive token count"):
            _parse_speculative_config('{"method":"mtp","num_speculative_tokens":0}')

    def test_inspect_model_help_names_both_supported_path_shapes(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout), self.assertRaises(SystemExit) as exit_context:
            build_parser().parse_args(["inspect-model", "--help"])

        self.assertEqual(exit_context.exception.code, 0)
        self.assertIn("adapter .py file or directory containing __init__.py", stdout.getvalue())

    def test_serve_max_model_len_defaults_to_auto(self) -> None:
        parser = build_parser()
        for arguments in ([], ["--max-model-len", "auto"], ["--max-model-len", "AUTO"]):
            with self.subTest(arguments=arguments):
                parsed = parser.parse_args(["serve", "/checkpoint", *arguments])
                self.assertIsNone(parsed.max_model_len)
        self.assertEqual(
            parser.parse_args(["serve", "/checkpoint", "--max-model-len", "8192"]).max_model_len,
            8192,
        )

    def test_mamba_ssm_cache_dtype_is_explicit_opt_in(self) -> None:
        parser = build_parser()

        self.assertEqual(
            parser.parse_args(["serve", "/checkpoint"]).mamba_ssm_cache_dtype,
            "bfloat16",
        )
        self.assertEqual(
            parser.parse_args(
                ["serve", "/checkpoint", "--mamba-ssm-cache-dtype", "bfloat16"]
            ).mamba_ssm_cache_dtype,
            "bfloat16",
        )

    def test_recurrent_state_dtype_alias_preserves_complete_runtime_configuration(self) -> None:
        parser = build_parser()
        common = ["serve", "/checkpoint", "--k-cache-dtype", "int4",
                  "--v-cache-dtype", "bfloat16"]
        for dtype in ("auto", "float32", "bfloat16"):
            with self.subTest(dtype=dtype):
                legacy = parser.parse_args(common + ["--mamba-ssm-cache-dtype", dtype])
                current = parser.parse_args(common + ["--recurrent-state-dtype", dtype])
                self.assertEqual(vars(current), vars(legacy))
                self.assertEqual(current.mamba_ssm_cache_dtype, dtype)
                self.assertEqual(current.k_cache_dtype, "int4")
                self.assertEqual(current.v_cache_dtype, "bfloat16")

    def test_recurrent_state_dtype_rejects_unimplemented_storage_types(self) -> None:
        parser = build_parser()
        for dtype in ("int4", "int8", "fp8", "float16"):
            with self.subTest(dtype=dtype), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    parser.parse_args(["serve", "/checkpoint", "--recurrent-state-dtype", dtype])
                self.assertEqual(error.exception.code, 2)

    def test_output_parsers_are_explicit_opt_in(self) -> None:
        parser = build_parser()
        cases = (
            ((), None, None),
            (("--reasoning-parser", "qwen3"), "qwen3", None),
            (("--tool-call-parser", "qwen3_coder"), None, "qwen3_coder"),
            (
                (
                    "--reasoning-parser",
                    "qwen3",
                    "--tool-call-parser",
                    "qwen3_coder",
                ),
                "qwen3",
                "qwen3_coder",
            ),
        )
        for arguments, reasoning, tool in cases:
            with self.subTest(arguments=arguments):
                parsed = parser.parse_args(["serve", "/checkpoint", *arguments])
                self.assertEqual(parsed.reasoning_parser, reasoning)
                self.assertEqual(parsed.tool_call_parser, tool)

        # Parser names belong to the selected model/HF adapter, not a CLI table.
        parsed = parser.parse_args(["serve", "/checkpoint", "--reasoning-parser", "custom"])
        self.assertEqual(parsed.reasoning_parser, "custom")

    def test_inspect_model_is_canonical_and_does_not_build_the_program(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            result = main(["inspect-model", "--model-code", str(TOY_ADAPTER)])

        self.assertEqual(result, 0)
        encoded = stdout.getvalue().strip()
        payload = json.loads(encoded)
        self.assertEqual(
            encoded,
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
        self.assertTrue(payload["model"]["full_bf16"])
        self.assertEqual(payload["cache"]["kv_dtype"], "bf16")
        self.assertEqual(
            {item["name"] for item in payload["modalities"]},
            {"text", "image", "video"},
        )
        self.assertEqual(
            {item["name"]: item["feature_hidden_size"] for item in payload["modalities"]},
            {"text": None, "image": 8, "video": 8},
        )
        self.assertFalse(payload["reference_plan"]["graph_enabled"])

    def test_model_error_is_concise_and_nonzero(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            result = main(["inspect-model", "--model-code", "missing.py"])
        self.assertEqual(result, 2)
        self.assertTrue(stderr.getvalue().startswith("error: "))
        self.assertNotIn("cubic error:", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_performance_serve_rejects_unqualified_cli_shapes(self) -> None:
        cases = (
            (("--pp-size", "3"), "cuda server topology is unsupported"),
            (("--tp-size", "4", "--ep-size", "3"), "EP dividing TP"),
            (
                (
                    "--tp-size",
                    "2",
                    "--pp-size",
                    "2",
                    "--devices",
                    "0,1,2",
                ),
                "requires 4 devices",
            ),
            (
                (
                    "--tp-size",
                    "2",
                    "--pp-size",
                    "2",
                    "--devices",
                    "0,1,1,2",
                ),
                "distinct",
            ),
            (
                (
                    "--tp-size",
                    "2",
                    "--pp-size",
                    "2",
                    "--no-prefix-cache",
                    "--max-model-len",
                    "0",
                ),
                "--max-model-len must be positive",
            ),
            (
                (
                    "--tp-size",
                    "2",
                    "--pp-size",
                    "2",
                    "--no-prefix-cache",
                    "--http-keep-alive-seconds",
                    "0",
                ),
                "--http-keep-alive-seconds must be positive",
            ),
        )
        for arguments, message in cases:
            with self.subTest(arguments=arguments):
                stderr = io.StringIO()
                with (
                    patch("cubic_serve.cli.socket.create_server") as create_server,
                    redirect_stderr(stderr),
                ):
                    result = main(
                        [
                            "serve",
                            "/checkpoint",
                            "--execution-mode",
                            "cuda",
                            *arguments,
                        ]
                    )

                self.assertEqual(result, 2)
                create_server.assert_not_called()
                self.assertIn(message, stderr.getvalue())

    def test_performance_serve_owns_executor_lifecycle_without_reference_engine(self) -> None:
        model_id = "qwen3_5:qwen3.8-27b:text"
        weight_variant = WeightVariantMeta(
            base_model_id=model_id,
            base_revision="b2s256:checkpoint",
        )
        checkpoint_weights = object()
        admission = SimpleNamespace(
            adapter=SimpleNamespace(
                descriptor=SimpleNamespace(
                    model_spec=SimpleNamespace(model_id=model_id),
                    weight_variant=weight_variant,
                    decode_extension=SimpleNamespace(
                        max_proposal_tokens=0,
                    ),
                )
            ),
            model_root=Path("/checkpoint"),
            vocabulary_size=248_320,
            checkpoint_weights=checkpoint_weights,
        )
        codec = SimpleNamespace(
            eos_token_id=151_645, eos_token_ids=(151_645,), admit_parser_recipe=MagicMock()
        )
        processor = object()
        executor = MagicMock()
        executor.max_sequence_tokens = 8192
        executor.kv_capacity_tokens = 7_013_696
        executor.kv_vram_capacity_tokens = 9_200_000
        executor.kv_capacity_shared = False
        server_socket = MagicMock()
        uvicorn_server = MagicMock()
        chat_backend = MagicMock()

        with (
            patch(
                "cubic_serve.models.qwen3_5.serving.admit_reference",
                return_value=admission,
            ) as admit,
            patch("cubic_serve.cli.Path.read_text", return_value='{"model_type":"qwen3_5"}'),
            patch(
                "cubic_serve.models.qwen3_5.output.Qwen3_5OutputCodec",
                return_value=codec,
            ),
            patch(
                "cubic_serve.models.qwen3_5.multimodal.QwenMultimodalProcessor",
                return_value=processor,
            ),
            patch(
                "cubic_serve.cli._load_serving_vision",
                return_value=(None, {4: 4096, 5: 4096}),
            ) as load_vision,
            patch("cubic_serve.generation.xgrammar.load_xgrammar_constraint_factory") as xgrammar,
            patch(
                "cubic_serve.runtime.performance_chat.PerformanceChatBackend",
                return_value=chat_backend,
            ) as build_chat_backend,
            patch(
                "cubic_serve.cli._start_qwen3_5_performance_executor",
                return_value=executor,
            ) as start_executor,
            patch("cubic_serve.runtime.engine.ReferenceEngine") as reference_engine,
            patch("cubic_serve.cli.socket.create_server", return_value=server_socket),
            patch("cubic_serve.cli._LOGGER.info") as info,
            patch("uvicorn.Config", return_value="config") as uvicorn_config,
            patch("cubic_serve.cli._run_http_server", uvicorn_server),
        ):
            result = main(
                [
                    "serve",
                    "/checkpoint",
                    "--execution-mode",
                    "cuda",
                    "--tp-size",
                    "2",
                    "--pp-size",
                    "2",
                    "--devices",
                    "4,5,6,7",
                    "--device",
                    "cuda:5",
                    "--max-model-len",
                    "8192",
                    "--reasoning-parser",
                    "qwen3",
                    "--tool-call-parser",
                    "qwen3_coder",
                    "--served-model-name",
                    "public-model",
                    "compat-name",
                ]
            )

        self.assertEqual(result, 0)
        admit.assert_called_once_with("/checkpoint", verify_payloads=False)
        self.assertEqual(load_vision.call_args.args[1], (4, 5, 6, 7))
        self.assertEqual(load_vision.call_args.args[0].device, "cuda:5")
        start_executor.assert_called_once_with(
            "/checkpoint",
            checkpoint_weights=checkpoint_weights,
            model_id=model_id,
            weight_variant=weight_variant,
            devices=(4, 5, 6, 7),
            eos_token_ids=(151_645,),
            max_sequence_tokens=8192,
            max_running_requests=256,
            tp_size=2,
            pp_size=2,
            ep_size=1,
            num_speculative_tokens=0,
            max_speculative_tokens=0,
            dynamic_max_speculative_tokens=0,
            dynamic_speculation=False,
            use_prefix_cache=True,
            mamba_ssm_cache_dtype="bfloat16",
            prefill_chunk_tokens=None,
            operator_config=None,
            draft_policy=DraftPolicyConfig(),
            device_memory_budgets={},
            input_encoder_reserves={4: 4096, 5: 4096},
            input_encoder=None,
        )
        uvicorn_server.assert_called_once()
        self.assertEqual(uvicorn_server.call_args.args, ("config", server_socket))
        with patch("cubic_serve.cli._LOGGER.info", info):
            uvicorn_server.call_args.kwargs["startup_summary"]()
        self.assertEqual(uvicorn_config.call_args.kwargs["timeout_keep_alive"], 60)
        self.assertFalse(uvicorn_config.call_args.kwargs["access_log"])
        self.assertIsNone(uvicorn_config.call_args.kwargs["log_config"])
        startup = info.call_args_list
        self.assertEqual(startup[0].args[0], "startup args=%s")
        self.assertIsInstance(json.loads(startup[0].args[1]), dict)
        self.assertEqual(startup[1].args[0], "cubic-serve version=%s")
        self.assertEqual(startup[-6].args[0], "serve endpoint: http://%s:%d")
        self.assertEqual(startup[-5].args, ("serve model: %s", "public-model"))
        self.assertEqual(startup[-4].args, ("serve context: %d tokens (%s)", 8192, "explicit"))
        self.assertEqual(startup[-3].args,
                         ("serve KV capacity: %s tokens (VRAM max≈%s)",
                          "7,013,696", "9,200,000"))
        executor.kv_capacity_shared = True
        info.reset_mock()
        with patch("cubic_serve.cli._LOGGER.info", info):
            uvicorn_server.call_args.kwargs["startup_summary"]()
        self.assertEqual(info.call_args_list[-3].args, (
            "serve cache capacity: %s tokens",
            "7,013,696"))
        build_chat_backend.assert_called_once_with(
            executor,
            model_id=model_id,
            weight_variant=weight_variant,
            processor=processor,
            output_codec=codec,
            enable_thinking=True,
            reasoning_parser="qwen3",
            tool_call_parser="qwen3_coder",
        )
        executor.close.assert_called_once_with()
        server_socket.close.assert_called_once_with()
        reference_engine.assert_not_called()
        xgrammar.assert_not_called()

    def test_distributed_eager_is_not_misrepresented_as_public_serving(self) -> None:
        stderr = io.StringIO()
        with (
            patch("cubic_serve.cli.socket.create_server") as create_server,
            redirect_stderr(stderr),
        ):
            result = main([
                "serve", "/checkpoint", "--execution-mode", "eager", "--tp-size", "2",
            ])

        self.assertEqual(result, 2)
        create_server.assert_not_called()
        self.assertIn("TP=PP=EP=1", stderr.getvalue())

    def test_serve_reports_the_actual_bind_failure(self) -> None:
        stderr = io.StringIO()
        bind_error = OSError(98, "Address already in use")
        with (
            patch("cubic_serve.cli.socket.create_server", side_effect=bind_error),
            redirect_stderr(stderr),
        ):
            result = main(
                [
                    "serve",
                    "/checkpoint",
                    "--execution-mode",
                    "cuda",
                    "--tp-size",
                    "4",
                    "--devices",
                    "0,1,2,3",
                    "--host",
                    "0.0.0.0",
                    "--port",
                    "30015",
                ]
            )

        self.assertEqual(result, 2)
        self.assertIn(
            "cannot bind 0.0.0.0:30015: Address already in use",
            stderr.getvalue(),
        )


class ModelAdapterEntryTests(unittest.TestCase):
    def test_default_uses_builtin_selection_without_loading_config_code(self):
        from cubic_serve.cli import _serve_qwen3_5

        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "config.json").write_text(json.dumps({
                "model_type": "qwen3_5",
                "cubic_serve": {"model_adapter": "never.py", "model_code": "never.py"},
            }))
            args = build_parser().parse_args(["serve", directory])
            self.assertIsNone(args.model_adapter)
            with (
                patch("cubic_serve.models.loader.load_model_definition") as load,
                patch("cubic_serve.cli._serve_model") as serve,
            ):
                _serve_qwen3_5(args)
            load.assert_not_called()
            serve.assert_called_once_with(args, model_definition_source=None)

    def test_explicit_source_overrides_builtin_and_cleans_success_or_failure(self):
        from cubic_serve.cli import _serve_qwen3_5

        for fails in (False, True):
            with self.subTest(fails=fails), tempfile.TemporaryDirectory() as directory:
                (Path(directory) / "__init__.py").write_text(
                    "from dataclasses import replace\n"
                    "from cubic_serve.models.qwen3_5 import MODEL_DEFINITION as BASE\n"
                    "MODEL_DEFINITION=replace(BASE,model_type='explicit_marker')\n"
                )
                args = build_parser().parse_args([
                    "serve", "/checkpoint", "--model-adapter", directory,
                ])
                observed = []

                def serve(_args, *, model_definition_source, expected_args=args, fails=fails,
                          observed=observed):
                    self.assertIs(_args, expected_args)
                    self.assertEqual(
                        model_definition_source.definition.model_type, "explicit_marker",
                    )
                    self.assertIn(model_definition_source.module_name, sys.modules)
                    observed.append(model_definition_source.module_name)
                    if fails:
                        raise RuntimeError("private startup failure")

                with patch("cubic_serve.cli._serve_model", side_effect=serve):
                    if fails:
                        with self.assertRaisesRegex(RuntimeError, "private startup failure"):
                            _serve_qwen3_5(args)
                    else:
                        _serve_qwen3_5(args)
                self.assertEqual(len(observed), 1)
                self.assertFalse(any(n == observed[0] or n.startswith(observed[0] + ".")
                                     for n in sys.modules))

    def test_explicit_source_eager_fails_before_code_loading(self):
        from cubic_serve.cli import _serve_qwen3_5

        args = build_parser().parse_args([
            "serve", "/checkpoint", "--model-adapter", "/not-loaded.py",
            "--execution-mode", "eager",
        ])
        with (
            patch("cubic_serve.models.loader.load_model_definition") as load,
            self.assertRaisesRegex(ValueError, "shared cuda or debug"),
        ):
            _serve_qwen3_5(args)
        load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
