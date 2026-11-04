"""Minimal command-line boundary for model inspection and engineering metadata."""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import socket
import sys
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, nullcontext, suppress
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

# CUDA serving uses process-per-rank GPU execution and small host-side batch
# assembly.  Initialize its CPU runtime before any model module imports Torch;
# an explicit operator setting remains authoritative.
if (sys.argv[1:2] == ["serve"]
        and not any(name in os.environ for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"))):
    os.environ["OMP_NUM_THREADS"] = "4"

# Reclaim free pages between graph captures without moving live tensor addresses.
# Set this before Torch is imported and inherit it in rank processes. Either
# explicit allocator spelling remains authoritative, including an empty value.
if (sys.argv[1:2] in (["serve"], ["qualify"])
        and "PYTORCH_ALLOC_CONF" not in os.environ
        and "PYTORCH_CUDA_ALLOC_CONF" not in os.environ):
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

from cubic_serve import __version__
from cubic_serve.contracts.model import WeightVariantMeta
from cubic_serve.models import ModelAdapterLoadError, load_model_adapter
from cubic_serve.ops.strategies import OperatorConfig
from cubic_serve.runtime.speculative import DraftPolicyConfig
from cubic_serve.runtime.prerequisites import (
    RuntimePrerequisiteError,
    require_host_build_prerequisites,
)

_LOGGER = logging.getLogger(__name__)


def _parse_true_false(value: str) -> bool:
    if value not in ("true", "false"):
        raise argparse.ArgumentTypeError("expected true or false")
    return value == "true"


def _kv_cache_storage(args: argparse.Namespace):
    from cubic_serve.runtime.cache.quantization import (
        CacheDType,
        CacheGroupAxis,
        CacheStorageSpec,
        KVCacheStorageSpec,
    )
    return KVCacheStorageSpec(
        CacheStorageSpec(CacheDType(args.k_cache_dtype), CacheGroupAxis.TOKENS),
        CacheStorageSpec(CacheDType(args.v_cache_dtype), CacheGroupAxis.CHANNELS),
        enable_rotation=args.enable_kv_cache_rotation,
    )


@contextmanager
def _serve_interrupts() -> Iterator[None]:
    """One interrupt requests shutdown; subsequent interrupts cannot orphan ranks."""

    signals = (signal.SIGINT, signal.SIGTERM)
    previous = {sig: signal.getsignal(sig) for sig in signals}

    def interrupt(signum: int, frame: Any) -> None:
        for sig in signals:
            signal.signal(sig, signal.SIG_IGN)
        raise KeyboardInterrupt

    for sig in signals:
        signal.signal(sig, interrupt)
    try:
        try:
            yield
        except KeyboardInterrupt:
            _LOGGER.info("Shutdown complete.")
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def _run_http_server(
    config: Any, server_socket: socket.socket, *, startup_summary: Any = None
) -> None:
    import uvicorn

    class Server(uvicorn.Server):
        async def startup(self, sockets: list[socket.socket] | None = None) -> None:
            await super().startup(sockets=sockets)
            if self.started and not self.should_exit:
                if startup_summary is not None:
                    startup_summary()
                from cubic_serve.runtime.logging import finish_log_startup

                finish_log_startup()

        def handle_exit(self, sig: int, frame: Any) -> None:
            if not self.should_exit:
                super().handle_exit(sig, frame)

        async def shutdown(self, sockets: list[socket.socket] | None = None) -> None:
            # Closing transports uses the existing ASGI disconnect/cancel path.
            # Do not wait for a potentially unbounded generation or force-cancel
            # ASGI tasks, which reports normal operator shutdown as an error.
            for connection in tuple(self.server_state.connections):
                connection.transport.close()
            await super().shutdown(sockets=sockets)

    Server(config).run(sockets=[server_socket])


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _model_payload(model_code: str) -> dict[str, Any]:
    loaded = load_model_adapter(model_code)
    descriptor = loaded.descriptor
    model = descriptor.model_spec
    return {
        "adapter": {
            "id": descriptor.adapter_id,
            "revision": descriptor.adapter_revision,
            "source_digest": loaded.source_digest,
            "source_path": str(loaded.source_path),
        },
        "cache": {
            "kv_dtype": model.cache.kv_dtype.value,
            "layout": model.cache.layout,
            "page_size_tokens": model.cache.page_size_tokens,
            "prefix_enabled": model.cache.prefix_enabled,
        },
        "decode_provider": descriptor.decode_extension.provider.value,
        "model": {
            "architecture": model.architecture,
            "full_bf16": model.is_full_bf16,
            "id": model.model_id,
            "layer_count": model.layer_count,
            "revision": model.revision,
            "weight_manifest": model.weights.manifest_digest,
        },
        "modalities": [
            {
                "encoder_id": modality.encoder_id,
                "feature_hidden_size": modality.feature_hidden_size,
                "mime_prefixes": list(modality.mime_prefixes),
                "name": modality.name,
            }
            for modality in model.modalities
        ],
        "reference_plan": {
            "graph_enabled": loaded.reference_plan.graph_enabled,
            "id": loaded.reference_plan.execution_plan_id,
            "mode": loaded.reference_plan.mode.value,
            "signatures": [
                {
                    "accumulator_dtype": binding.signature.accumulator_dtype.value,
                    "cache_form": binding.signature.cache_form,
                    "implementation_id": binding.reference_implementation_id,
                    "input_dtypes": [dtype.value for dtype in binding.signature.input_dtypes],
                    "layout": binding.signature.layout,
                    "output_dtype": binding.signature.output_dtype.value,
                    "phase": binding.signature.phase.value,
                    "semantic_op": binding.signature.semantic_op,
                    "shape": dict(binding.signature.shape_items),
                    "state_form": binding.signature.state_form,
                    "topology_id": binding.signature.topology_id,
                }
                for binding in loaded.reference_plan.bindings
            ],
        },
        "states": [
            {
                "commit_alignment_tokens": state.commit_alignment_tokens,
                "dtype": state.dtype.value,
                "id": state.state_id,
                "layout": state.layout,
                "prefix_replay": state.prefix_replay.value,
                "schema_version": state.schema_version,
                "sharding": state.sharding.value,
            }
            for state in model.states
        ],
    }


def _parse_max_model_len(value: str) -> int | None:
    if value.strip().lower() == "auto":
        return None
    try:
        return int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected an integer or 'auto'") from error


def _parse_graph_max_tokens(value: str) -> int | str:
    parsed = _parse_max_model_len(value)
    if parsed is None:
        return "auto"
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer or 'auto'")
    return parsed


def _parse_gib(value: str) -> int | None:
    if value.strip().lower() == "auto":
        return None
    import math

    try:
        amount = float(value)
        if not math.isfinite(amount) or amount < 0:
            raise ValueError
        return int(amount * (1 << 30))
    except (ValueError, OverflowError) as error:
        raise argparse.ArgumentTypeError("expected finite non-negative GiB or 'auto'") from error


def _parse_positive_gib(value: str) -> int:
    amount = _parse_gib(value)
    if amount is None or amount <= 0:
        raise argparse.ArgumentTypeError("expected positive GiB")
    return amount


def _parse_gpu_gib(value: str) -> int | None:
    return None if value.strip().lower() == "auto" else _parse_positive_gib(value)


def _parse_nonnegative_gib(value: str) -> int:
    amount = _parse_gib(value)
    if amount is None:
        raise argparse.ArgumentTypeError("expected non-negative GiB")
    return amount


def _device_memory_budgets(args, devices):
    from cubic_serve.executor.runner import configure_reference_math
    from cubic_serve.runtime.cache import DeviceMemoryBudget, read_cuda_memory_snapshot

    configure_reference_math()
    budgets = {}
    for device in dict.fromkeys(devices):
        snapshot = read_cuda_memory_snapshot(f"cuda:{device}", initialize_cuda=False)
        budget = DeviceMemoryBudget(snapshot, args.gpu_memory_bytes, args.gpu_reserve_bytes)
        budgets[device] = budget
        limit = snapshot.free_bytes if budget.limit_bytes is None else budget.limit_bytes
        _LOGGER.info("GPU memory budget device=%s limit=%.3f GiB reserve=%.3f GiB available=%.3f GiB",
                     snapshot.device, limit / (1 << 30), budget.reserve_bytes / (1 << 30),
                     snapshot.free_bytes / (1 << 30))
    return budgets


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        self.exit(2, f"error: {message}\n")


def _parse_api_key(value: str) -> str:
    if not value or any(not 33 <= ord(char) <= 126 for char in value):
        raise argparse.ArgumentTypeError("expected a non-empty printable ASCII key without spaces")
    return value


def _parse_served_model_name(value: str) -> str:
    if not value or value.strip() != value:
        raise argparse.ArgumentTypeError("served model names must be non-empty without surrounding whitespace")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(prog="cubic")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect-model",
        help="validate an adapter file or directory without building weights",
    )
    inspect_parser.add_argument(
        "--model-code",
        required=True,
        help="adapter .py file or directory containing __init__.py",
    )

    qwen_parser = subparsers.add_parser(
        "reference-qwen3-5",
        help="run a real qwen3_5 checkpoint through the eager text reference path",
    )
    qwen_parser.add_argument("--model", required=True)
    qwen_parser.add_argument("--input-ids", required=True)
    qwen_parser.add_argument("--max-new-tokens", type=int, default=2)
    qwen_parser.add_argument("--device", default="cuda:0")

    serve_parser = subparsers.add_parser(
        "serve",
        help="serve a supported checkpoint through the public chat APIs",
    )
    serve_parser.add_argument("model")

    server_options = serve_parser.add_argument_group('Server')
    server_options.add_argument(
        "--served-model-name", nargs="+", type=_parse_served_model_name, metavar="NAME",
        help="Public API model name(s); defaults to the checkpoint model ID. "
             "All names are accepted; responses use the first name.",
    )
    server_options.add_argument("--host", default="127.0.0.1")
    server_options.add_argument("--port", type=int, default=8000)
    server_options.add_argument(
        "--api-key", type=_parse_api_key,
        help="Require this API key for /v1 requests; disabled when omitted",
    )
    server_options.add_argument(
        "--http-keep-alive-seconds",
        type=int,
        default=60,
        help="idle HTTP/1.1 connection lifetime; must exceed normal client pool expiry",
    )
    server_options.add_argument(
        "--log-config",
        help='optional file logging JSON, e.g. {"dir":"./logs","rotation":"100 MB","retention":"7 days"}',
    )

    model_options = serve_parser.add_argument_group('Model and generation')
    model_options.add_argument(
        "--compute-mode", choices=("wna16", "wna8"), default="wna16",
        help="Packed compute: wna16 (default) or wna8",
    )
    model_options.add_argument(
        "--dtype",
        choices=("auto", "fp16", "float16", "bf16", "bfloat16"),
        default="auto",
        help="Non-quantized execution dtype; packed weights and format scale dtypes are preserved",
    )
    model_options.add_argument(
        "--max-model-len",
        type=_parse_max_model_len,
        help="optional context cap; auto or omitted uses the largest capacity that fits GPU memory",
    )
    model_options.add_argument("--chat-template")
    model_options.add_argument(
        "--reasoning-parser",
        help="split reasoning from content; omitted means no reasoning parser",
    )
    model_options.add_argument(
        "--tool-call-parser",
        help="decode model tool calls; omitted means no tool-call parser",
    )
    model_options.add_argument(
        "--thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "set the chat-template thinking default (enabled by Qwen); "
            "does not enable a reasoning parser"
        ),
    )
    model_options.add_argument(
        "--structured-outputs-config",
        help='JSON policy, e.g. {"enable_in_reasoning":true,"disable_any_whitespace":true}',
    )
    model_options.add_argument(
        "--speculative-config",
        help='JSON object, e.g. {"method":"mtp","num_speculative_tokens":1}',
    )

    parallel_options = serve_parser.add_argument_group('Devices and parallelism')
    parallel_options.add_argument(
        "--devices",
        default="0,1,2,3",
        help="comma-separated CUDA device indices for cuda mode",
    )
    parallel_options.add_argument(
        "--device", default=None,
        help="Frontend/vision device; defaults to the first declared serving device",
    )
    parallel_options.add_argument("--tp-size", type=int, default=1)
    parallel_options.add_argument("--pp-size", type=int, default=1)
    parallel_options.add_argument("--ep-size", type=int, default=1)

    memory_options = serve_parser.add_argument_group('Memory and cache')
    memory_options.add_argument(
        "--gpu-memory-gib", dest="gpu_memory_bytes", type=_parse_gpu_gib, default=None,
        metavar="GiB|auto",
        help="Per-device application GPU budget in GiB, including weights, encoder/runtime "
        "graphs, scratch, state and KV; auto (default) uses startup available memory",
    )
    memory_options.add_argument(
        "--gpu-reserve-gib", dest="gpu_reserve_bytes", type=_parse_nonnegative_gib, default=1 << 30,
        metavar="GiB",
        help="Unallocated runtime headroom inside each GPU budget, in GiB (default: 1)",
    )
    for side in ("k", "v"):
        memory_options.add_argument(
            f"--{side}-cache-dtype", default="auto",
            choices=("auto", "bfloat16", "float16", "fp8", "int8", "int6", "int4"),
            help=f"{side.upper()} cache storage; auto preserves the model compute dtype",
        )
    memory_options.add_argument(
        "--enable-kv-cache-rotation", type=_parse_true_false, default=False,
        metavar="{true,false}", help="paired KV cache rotation (development default: false)",
    )
    memory_options.add_argument(
        "--recurrent-state-dtype", "--mamba-ssm-cache-dtype",
        dest="mamba_ssm_cache_dtype",
        choices=("auto", "float32", "bfloat16"),
        default="bfloat16",
        help=("GDN/SSM state storage dtype (default: bfloat16); auto follows the checkpoint. "
              "Independent of K/V storage; supported choices depend on the execution path."),
    )
    memory_options.add_argument(
        "--prefix-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    scheduling_options = serve_parser.add_argument_group('Scheduling')
    scheduling_options.add_argument("--max-running-requests", type=int, default=256)
    scheduling_options.add_argument("--token-budget", type=int, default=16)
    scheduling_options.add_argument(
        "--cuda-graph-max-tokens", type=_parse_graph_max_tokens, default="auto",
        metavar="TOKENS|auto",
        help="Physical token-row budget for startup text CUDA graphs (default: auto); "
             "independent of max-running-requests. Includes speculative review and padding; "
             "uses power-of-two anchors at or below the budget.",
    )
    scheduling_options.add_argument(
        "--prefill-chunk-tokens",
        type=int,
        default=None,
        help=(
            "Optional per-request prefill cap; "
            "default uses qualified graph capacity and decode quota"
        ),
    )

    multimodal_options = serve_parser.add_argument_group('Multimodal input')
    multimodal_options.add_argument(
        "--multimodal-memory-gib", dest="multimodal_memory_bytes", type=_parse_positive_gib,
        default=3 << 29, metavar="GiB",
        help="Input-encoder graphs and transient input-work sub-budget in GiB (default: 1.5); "
        "encoder weights also consume the global GPU budget",
    )
    multimodal_options.add_argument(
        "--mm-worker-layout", choices=("colocated", "separate"), default="colocated",
        help="Input encoder shares each first-stage GPU worker, or owns separate GPU workers.",
    )
    multimodal_options.add_argument(
        "--mm-preprocess-workers", type=int, default=2,
        help="Maximum concurrent image/video/audio input preprocessing workers",
    )
    multimodal_options.add_argument(
        "--mm-feature-cache-gib", dest="mm_feature_cache_bytes", type=_parse_nonnegative_gib,
        default=1 << 28, metavar="GiB",
        help="Host-memory encoded-feature cache budget in GiB (default: 0.25; 0 disables caching)",
    )
    multimodal_options.add_argument(
        "--video-fps", type=float, default=2.0,
        help="Video input sampling rate over the complete clip; exceeding the frame/pixel "
        "budget is rejected rather than truncating the clip",
    )
    multimodal_options.add_argument(
        "--video-max-frames", type=int, default=768,
        help="Maximum sampled video frames per input (at least 4); decoded pixel limits also apply",
    )
    multimodal_options.add_argument(
        "--allowed-local-media-path",
        help="directory whose descendants may be read through file:// media URLs",
    )

    debug_options = serve_parser.add_argument_group('Advanced, audit and debug')
    debug_options.add_argument(
        "--execution-mode",
        choices=("cuda", "eager", "debug"),
        default="cuda",
        help="Execution mode: cuda (default, qualified CUDA Graphs), eager "
        "(eager execution), or debug (development debugging and experiments)",
    )
    debug_options.add_argument(
        "--startup-numerical-audit", action="store_true",
        help="Run the full independent numerical and cross-M acceptance suite before "
        "readiness (shared text CUDA Graph path; expensive development check)",
    )
    debug_options.add_argument(
        "--model-adapter", metavar="PATH",
        help="Explicit local MODEL_DEFINITION package or Python file; defaults to built-in support",
    )
    debug_options.add_argument(
        "--operator-config", help="JSON file with startup operator parameters"
    )
    debug_options.add_argument(
        "--activation-dtype",
        choices=("auto", "fp32"),
        default=None,
        help="Activation arithmetic: auto follows model dtype (unsupported BF16 uses FP16); "
        "fp32 forces FP32. Stable softmax reductions remain FP32; storage dtype is unchanged.",
    )
    debug_options.add_argument(
        "--attention-dtype",
        choices=("auto", "fp32"),
        default=None,
        help="Attention/GDN operands: auto uses model dtype with FP32 accumulation (default); "
        "fp32 retains wide products. Softmax precision is independent.",
    )
    debug_options.add_argument(
        "--softmax-dtype",
        choices=("auto", "fp32"),
        default=None,
        help="Softmax arithmetic: auto inherits activation dtype; fp32 overrides only "
        "attention/router softmax (default), retaining activation and storage dtypes.",
    )

    qualify_parser = subparsers.add_parser(
        "qualify", parents=[serve_parser], add_help=False,
        help="explicitly prepare startup evidence and calibrated plans; exits without serving HTTP",
    )
    qualify_parser.set_defaults(startup_qualification=True)
    return parser


def _parse_token_ids(value: str) -> tuple[int, ...]:
    try:
        token_ids = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise ValueError("--input-ids must be comma-separated integers") from error
    if not token_ids or any(token_id < 0 for token_id in token_ids):
        raise ValueError("--input-ids must contain non-negative integers")
    return token_ids


@dataclass(frozen=True, slots=True)
class _SpeculativeOptions:
    initial_tokens: int = 0
    max_tokens: int = 0
    dynamic: bool = False
    policy: DraftPolicyConfig = field(default_factory=DraftPolicyConfig)


def _structured_outputs_kwargs(args: argparse.Namespace) -> dict[str, bool]:
    value = getattr(args, "structured_outputs_config", None)
    if value is None:
        return {}
    try:
        options = json.loads(value)
    except (TypeError, ValueError) as error:
        raise ValueError("--structured-outputs-config must be valid JSON") from error
    names = {"backend", "enable_in_reasoning", "disable_any_whitespace"}
    if not isinstance(options, dict) or set(options) - names:
        raise ValueError("unsupported structured output configuration fields")
    if options.get("backend", "auto") not in ("auto", "xgrammar"):
        raise ValueError("structured output backend must be auto or xgrammar")
    result = {}
    for name, parameter in (
        ("enable_in_reasoning", "structured_output_in_reasoning"),
        ("disable_any_whitespace", "structured_output_disable_whitespace"),
    ):
        enabled = options.get(name, False)
        if type(enabled) is not bool:
            raise ValueError(f"structured output {name} must be boolean")
        if enabled:
            result[parameter] = True
    return result


def _parse_speculative_config(value: str | None) -> _SpeculativeOptions:
    if value is None:
        return _SpeculativeOptions()
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError("--speculative-config must be valid JSON") from error
    if not isinstance(payload, dict) or payload.get("method") != "mtp":
        raise ValueError("speculative config requires method=mtp and a positive token count")
    initial = payload.get("num_speculative_tokens")
    if type(initial) is not int or initial < 1:
        raise ValueError("speculative config requires method=mtp and a positive token count")
    if "mode" not in payload:
        if set(payload) != {"method", "num_speculative_tokens"}:
            raise ValueError("fixed speculative config only accepts method and token count")
        return _SpeculativeOptions(initial, initial, False)
    if (
        set(payload) - {"method", "num_speculative_tokens", "max_tokens", "mode", "policy"}
        or not {"max_tokens", "mode"} <= set(payload)
        or payload["mode"] != "dynamic"
        or type(payload["max_tokens"]) is not int
        or payload["max_tokens"] < initial
    ):
        raise ValueError(
            "dynamic speculative config requires mode=dynamic and max_tokens >= token count"
        )
    return _SpeculativeOptions(
        initial, payload["max_tokens"], True, DraftPolicyConfig.from_dict(payload.get("policy", {}))
    )


def _parse_devices(value: str, world_size: int | None = None) -> tuple[int, ...]:
    try:
        devices = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise ValueError("--devices must be comma-separated CUDA indices") from error
    if not devices or any(device < 0 for device in devices):
        raise ValueError("--devices must contain non-negative CUDA indices")
    if len(set(devices)) != len(devices):
        raise ValueError("--devices must contain distinct CUDA indices")
    if world_size is not None and len(devices) < world_size:
        raise ValueError(f"TP x PP requires {world_size} devices; --devices lists {len(devices)}")
    return devices if world_size is None else devices[:world_size]


def _start_qwen3_5_performance_executor(
    checkpoint: str,
    *,
    checkpoint_weights: Any,
    model_id: str,
    weight_variant: WeightVariantMeta,
    devices: tuple[int, ...],
    eos_token_ids: tuple[int, ...],
    max_sequence_tokens: int | None,
    max_running_requests: int,
    tp_size: int,
    pp_size: int,
    ep_size: int,
    num_speculative_tokens: int,
    max_speculative_tokens: int,
    dynamic_max_speculative_tokens: int,
    dynamic_speculation: bool,
    use_prefix_cache: bool,
    mamba_ssm_cache_dtype: str,
    prefill_chunk_tokens: int | None,
    operator_config: OperatorConfig | None = None,
    draft_policy: DraftPolicyConfig | None = None,
    device_memory_budgets: dict | None = None,
    input_encoder_reserves: dict | None = None,
    input_encoder: Any = None,
) -> Any:
    """Keep the native executor import and provisional ABI in one boundary."""

    from cubic_serve.models.qwen3_5.serving import start_performance_executor

    return start_performance_executor(
        checkpoint,
        checkpoint_weights=checkpoint_weights,
        model_id=model_id,
        weight_variant=weight_variant,
        devices=devices,
        eos_token_ids=eos_token_ids,
        max_sequence_tokens=max_sequence_tokens,
        max_running_requests=max_running_requests,
        tp_size=tp_size,
        pp_size=pp_size,
        ep_size=ep_size,
        num_speculative_tokens=num_speculative_tokens,
        max_speculative_tokens=max_speculative_tokens,
        dynamic_max_speculative_tokens=dynamic_max_speculative_tokens,
        dynamic_speculation=dynamic_speculation,
        use_prefix_cache=use_prefix_cache,
        mamba_ssm_cache_dtype=mamba_ssm_cache_dtype,
        prefill_chunk_tokens=prefill_chunk_tokens,
        operator_config=operator_config,
        draft_policy=draft_policy,
        device_memory_budgets=device_memory_budgets,
        input_encoder_reserves=input_encoder_reserves,
        input_encoder=input_encoder,
    )


def _reference_qwen_payload(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from cubic_serve.executor.runner import ReferenceExecutionContext
    from cubic_serve.models.qwen3_5.model import load_qwen3_5_reference

    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    token_ids = _parse_token_ids(args.input_ids)
    load_started = time.perf_counter()
    model = load_qwen3_5_reference(args.model, device=args.device)
    load_seconds = time.perf_counter() - load_started
    prompt = torch.tensor((token_ids,), dtype=torch.int64, device=model.weights.device)
    generated: list[int] = []
    prefill_seconds = 0.0
    decode_seconds: list[float] = []
    cache = None
    with ReferenceExecutionContext():
        for index in range(args.max_new_tokens):
            step_input = (
                prompt
                if index == 0
                else torch.tensor(
                    ((generated[-1],),),
                    dtype=torch.int64,
                    device=model.weights.device,
                )
            )
            started = time.perf_counter()
            result = model.forward(step_input, cache=cache)
            if model.weights.device.type == "cuda":
                torch.cuda.synchronize(model.weights.device)
            elapsed = time.perf_counter() - started
            if index == 0:
                prefill_seconds = elapsed
            else:
                decode_seconds.append(elapsed)
            generated.append(int(result.logits[:, -1].argmax(dim=-1).item()))
            cache = result.cache
    assert cache is not None
    return {
        "cache": {
            "kv_dtypes": [str(dtype).removeprefix("torch.") for dtype in cache.kv_dtypes],
            "sequence_length": cache.sequence_length,
        },
        "checkpoint": {
            "inventory_digest": model.weights.inventory_digest,
            "semantic_plan_digest": model.weights.semantic_plan_digest,
            "text_tensor_count": model.weights.tensor_count,
        },
        "device": str(model.weights.device),
        "generated_token_ids": generated,
        "mode": "reference-eager",
        "model_type": model.config.identity.model_type,
        "timing_seconds": {
            "decode": decode_seconds,
            "load": load_seconds,
            "prefill": prefill_seconds,
        },
    }


def _quantized_eager_backend(
    args: argparse.Namespace, operator_config: OperatorConfig | None
) -> tuple[Any, Any, str]:
    import torch

    from cubic_serve.contracts.model import WeightVariantMeta
    from cubic_serve.models.builtin import builtin_model_definition, parse_builtin_model_config
    from cubic_serve.ops.cuda.linear.quantized import PackedLinearOperators
    from cubic_serve.runtime.quantized import QuantizedEagerExecutor, load_quantized_model
    from cubic_serve.runtime.token_chat import TokenChatBackend

    raw = json.loads((Path(args.model) / "config.json").read_text())
    dtype = (args.dtype if args.dtype != "auto"
             else parse_builtin_model_config(raw).text_config.dtype)
    state_dtype = {"auto": torch.float32, "float32": torch.float32, "bfloat16": torch.bfloat16}[
        args.mamba_ssm_cache_dtype
    ]
    from cubic_serve.executor.runner import configure_reference_math

    configure_reference_math()
    if torch.device(args.device).type != "cuda":
        raise ValueError("quantized native eager serving requires a CUDA device")
    operators = PackedLinearOperators(
        operator_config, device=torch.cuda.get_device_name(args.device), model=Path(args.model).name
    )
    model, plan = load_quantized_model(
        args.model,
        device=args.device,
        dtype=dtype,
        recurrent_state_dtype=state_dtype,
        operators=operators,
        include_mtp=False,
    )
    model_id = Path(args.model).name
    input_factory, output_factory = builtin_model_definition(model.config).frontend_factories(
        multimodal=False)
    codec = output_factory(args.model, chat_template_path=args.chat_template)
    processor = input_factory(args.model, chat_template_path=args.chat_template)
    executor = QuantizedEagerExecutor(
        model,
        model_root=args.model,
        max_sequence_tokens=args.max_model_len,
        prefill_chunk_tokens=args.prefill_chunk_tokens or args.token_budget,
    )
    semantic_digest = plan.semantic_digest
    if operator_config is not None and operator_config.compute_mode != "wna16":
        import hashlib

        semantic_digest = hashlib.blake2s(
            (plan.semantic_digest + operator_config.digest).encode()
        ).hexdigest()
    try:
        backend = TokenChatBackend(
            executor,
            model_id=model_id,
            weight_variant=WeightVariantMeta(model_id, semantic_digest),
            processor=processor,
            output_codec=codec,
            enable_thinking=args.thinking,
            reasoning_parser=args.reasoning_parser or False,
            tool_call_parser=args.tool_call_parser or False,
            **_structured_outputs_kwargs(args),
        )
    except BaseException:
        executor.close()
        raise
    _LOGGER.info(
        "quantized execution=eager linear=native-packed dtype=%s "
        "state_dtype=%s format=%s operator_config=%s; "
        "text serving, no graph/concurrency qualification",
        model.compute_dtype,
        model.recurrent_state_dtype,
        plan.quantization.method,
        operators.config.digest,
    )
    return backend, executor, model_id


def _load_serving_vision(args, devices, operator_config, memory_budgets, multimodal_policy,
                        checkpoint=None, model_config=None, model_definition_source=None):
    """Shared encoder topology and byte ownership for all checkpoint formats."""
    import torch

    from cubic_serve.inputs.multimodal import EncoderMemoryBudget
    from cubic_serve.models.builtin import (
        builtin_model_definition,
        builtin_model_definition_for_type,
    )
    definition = (model_definition_source.definition if model_definition_source is not None
                  else builtin_model_definition(model_config) if model_config is not None
                  else builtin_model_definition_for_type(
                      json.loads((Path(args.model) / "config.json").read_text()).get("model_type")))
    factory = definition.input_encoder
    if factory is None:
        return None, {}
    source_options = ({} if model_definition_source is None else
                      {"model_definition_source": model_definition_source})
    frontend_device = torch.device(args.device or f"cuda:{devices[0]}")
    frontend_index = (frontend_device.index if frontend_device.index is not None
                      else 0)
    frontend_resources = EncoderMemoryBudget(multimodal_policy)
    colocated = args.mm_worker_layout == "colocated"
    _LOGGER.info("input encoder worker layout=%s", args.mm_worker_layout)
    if args.tp_size > 1 or colocated:
        input_devices = devices[:args.tp_size]
        if frontend_index not in input_devices:
            raise ValueError("--device must belong to the first pipeline stage's encoder TP group")
        encoder_devices = (tuple(input_devices) if colocated else
                           (frontend_index, *(d for d in input_devices if d != frontend_index)))
        _LOGGER.info("module=vision_tower parallelism=tp tp_size=%d devices=%s",
                     args.tp_size, encoder_devices)
        vision = factory(
            args.model, parallel=True, devices=encoder_devices, operator_config=operator_config,
            memory_bytes=args.multimodal_memory_bytes, device_memory_budgets=memory_budgets,
            checkpoint=checkpoint, colocated=colocated, model_config=model_config, **source_options)
        encoder_reserves = vision.input_reserves_by_device
    else:
        _LOGGER.info("module=vision_tower parallelism=tp tp_size=1 devices=(%d,)", frontend_index)
        if frontend_index in memory_budgets:
            memory_budgets[frontend_index].apply_allocator_limit()
        vision = factory(
            args.model, parallel=False, device=args.device or f"cuda:{devices[0]}",
            operator_config=operator_config,
            memory_bytes=args.multimodal_memory_bytes, resources=frontend_resources,
            device_memory_budget=memory_budgets.get(frontend_index), checkpoint=checkpoint,
            model_config=model_config, **source_options)
        encoder_reserves = {frontend_index: vision.peak_input_bytes}
    return vision, encoder_reserves


def _text_graph_backend(
    args: argparse.Namespace, operator_config: OperatorConfig | None, *,
    model_definition_source=None,
) -> tuple[Any, Any, str]:
    """Shared model text graph; each checkpoint binds plain or encoded weights."""
    from cubic_serve.contracts.model import WeightVariantMeta
    from cubic_serve.executor.runner import configure_reference_math
    from cubic_serve.inputs.multimodal import MultimodalInputPolicy
    from cubic_serve.runtime.quantized_graph import TextGraphExecutor
    from cubic_serve.runtime.token_chat import TokenChatBackend

    multimodal_policy = MultimodalInputPolicy(
        encoder_memory_bytes=args.multimodal_memory_bytes,
        preprocess_workers=args.mm_preprocess_workers,
        feature_cache_bytes=args.mm_feature_cache_bytes,
        video_frames_per_second=args.video_fps,
        video_max_frames=args.video_max_frames)
    devices = _parse_devices(args.devices, args.tp_size * args.pp_size)
    import torch
    configure_reference_math()
    frontend_device = torch.device(args.device or f"cuda:{devices[0]}")
    frontend_index = (frontend_device.index if frontend_device.index is not None
                      else 0)
    memory_budgets = _device_memory_budgets(args, (*devices, frontend_index))
    from cubic_serve.models.builtin import (
        admit_builtin_checkpoint,
        admit_model_checkpoint,
        builtin_model_definition,
        builtin_text_config,
        parse_builtin_model_config_json,
        parse_model_config_json,
    )
    config_document = (Path(args.model) / "config.json").read_bytes()
    source_options = ({} if model_definition_source is None else
                      {"model_definition_source": model_definition_source})
    if model_definition_source is None:
        model_config = parse_builtin_model_config_json(config_document)
        definition = builtin_model_definition(model_config)
    else:
        definition = model_definition_source.definition
        model_config = parse_model_config_json(config_document, definition=definition)
    requested_dtype = (builtin_text_config(model_config).dtype
                       if args.dtype == "auto" else args.dtype)
    if requested_dtype not in ("bf16", "bfloat16", "fp16", "float16"):
        raise ValueError("text graphs require an FP16 or BF16 execution dtype")
    multimodal = definition.input_encoder is not None
    input_factory, output_factory = definition.frontend_factories(multimodal=multimodal)
    codec = output_factory(args.model, chat_template_path=args.chat_template)
    if args.reasoning_parser or args.tool_call_parser:
        codec.admit_parser_recipe(
            reasoning_parser=args.reasoning_parser or False,
            tool_call_parser=args.tool_call_parser or False,
        )
    import time
    metadata_started = time.perf_counter()
    _LOGGER.info("checkpoint metadata started modules=vision_tower,main_model")
    checkpoint = (
        admit_builtin_checkpoint(args.model, model_config=model_config)
        if model_definition_source is None else
        admit_model_checkpoint(args.model, model_config=model_config, definition=definition)
    )
    _LOGGER.info("checkpoint metadata complete modules=vision_tower,main_model duration_s=%.3f",
                 time.perf_counter() - metadata_started)
    vision, encoder_reserves = _load_serving_vision(
        args, devices, operator_config, memory_budgets, multimodal_policy, checkpoint,
        model_config=model_config, **source_options)
    try:
        processor_options = dict(chat_template_path=args.chat_template, model_config=model_config)
        if multimodal:
            if vision is None:
                raise ValueError("declared input encoder construction returned no encoder")
            processor_options.update(
                vision=vision, allowed_local_media_path=args.allowed_local_media_path,
                resource_policy=multimodal_policy,
            )
        processor = input_factory(args.model, **processor_options)
        _LOGGER.info("Preparing checkpoint metadata and rank ownership for the text graph")
        speculative = _parse_speculative_config(args.speculative_config)
        executor = TextGraphExecutor(
            args.model, devices=devices, tp=args.tp_size, pp=args.pp_size, ep=args.ep_size,
            max_requests=args.max_running_requests, max_context=args.max_model_len,
            prefill_chunk=args.prefill_chunk_tokens, operator_config=operator_config,
            cuda_graph_max_tokens=args.cuda_graph_max_tokens,
            eos_token_ids=codec.eos_token_ids, use_prefix_cache=args.prefix_cache,
            reasoning_token_boundaries=(
                (tuple(codec.reasoning_start_token_ids), tuple(codec.reasoning_end_token_ids),
                 *tuple(getattr(codec, "reasoning_implicit_end_token_ids", ())))
                if args.reasoning_parser and getattr(codec, "reasoning_start_token_ids", ())
                and getattr(codec, "reasoning_end_token_ids", ()) else ()
            ),
            execution_mode=args.execution_mode,
            compute_dtype=requested_dtype,
            mamba_ssm_cache_dtype=args.mamba_ssm_cache_dtype,
            cache_storage=_kv_cache_storage(args),
            startup_numerical_audit=args.startup_numerical_audit,
            startup_qualification=getattr(args, "startup_qualification", False),
            input_encoder_reserves=encoder_reserves,
            device_memory_budgets=memory_budgets,
            input_encoder=vision,
            checkpoint=checkpoint,
            num_speculative_tokens=speculative.initial_tokens,
            max_speculative_tokens=speculative.max_tokens,
            dynamic_speculation=speculative.dynamic,
            draft_policy=speculative.policy,
            **source_options,
        )
    except BaseException:
        close_encoder = getattr(vision, "close", None)
        if close_encoder is not None:
            close_encoder()
        raise
    model_id = Path(args.model).name
    try:
        backend = TokenChatBackend(
            executor,
            model_id=model_id,
            weight_variant=WeightVariantMeta(model_id, executor.semantic_digest),
            processor=processor,
            output_codec=codec,
            enable_thinking=args.thinking,
            reasoning_parser=args.reasoning_parser or False,
            tool_call_parser=args.tool_call_parser or False,
            multimodal_policy=multimodal_policy,
            **_structured_outputs_kwargs(args),
        )
    except BaseException:
        executor.close()
        raise
    if args.execution_mode == "debug":
        _LOGGER.warning("debug: numerical/performance qualification pending; MTP0")
    return backend, executor, model_id


def _serve_operator_config(args: argparse.Namespace) -> OperatorConfig | None:
    operator_config_path = getattr(args, "operator_config", None)
    operator_config = (
        OperatorConfig.from_file(operator_config_path) if operator_config_path else None
    )
    activation_dtype = getattr(args, "activation_dtype", None)
    softmax_dtype = getattr(args, "softmax_dtype", None)
    attention_dtype = getattr(args, "attention_dtype", None)
    compute_mode = getattr(args, "compute_mode", "wna16")
    if operator_config is None and (
        compute_mode != "wna16"
        or activation_dtype is not None
        or softmax_dtype is not None
        or attention_dtype is not None
    ):
        operator_config = OperatorConfig()
    if operator_config is not None or getattr(args, "compute_mode", "wna16") != "wna16":
        operator_config = replace(
            operator_config or OperatorConfig(), compute_mode=compute_mode,
        )
    if activation_dtype is not None:
        operator_config = replace(
            operator_config or OperatorConfig(), activation_dtype=activation_dtype
        )
    if softmax_dtype is not None:
        operator_config = replace(
            operator_config or OperatorConfig(), softmax_dtype=softmax_dtype
        )
    if attention_dtype is not None:
        operator_config = replace(
            operator_config or OperatorConfig(), attention_dtype=attention_dtype
        )
    return operator_config


def _serve_qwen3_5(args: argparse.Namespace) -> None:
    """Compatibility entry for serve; repository code is loaded only explicitly."""
    from cubic_serve.models.loader import load_model_definition

    _structured_outputs_kwargs(args)

    source_path = getattr(args, "model_adapter", None)
    if source_path is not None and args.execution_mode not in ("cuda", "debug"):
        raise ValueError("explicit model adapters require the shared cuda or debug text path")
    lifetime = nullcontext(None) if source_path is None else load_model_definition(source_path)
    with lifetime as source:
        _serve_model(args, model_definition_source=source)


def _qualify_startup(args: argparse.Namespace) -> None:
    """Use the existing engine to prepare evidence, without opening a socket."""
    from cubic_serve.models.loader import load_model_definition
    from cubic_serve.runtime.logging import configure_logging

    configure_logging()
    if args.execution_mode != "cuda":
        raise ValueError("startup qualification requires the shared CUDA graph engine")
    if args.device is None:
        args.device = f"cuda:{_parse_devices(args.devices, args.tp_size * args.pp_size)[0]}"
    source_path = getattr(args, "model_adapter", None)
    lifetime = nullcontext(None) if source_path is None else load_model_definition(source_path)
    with lifetime as source:
        _, executor, _ = _text_graph_backend(
            args, _serve_operator_config(args), model_definition_source=source)
        executor.close()
    _LOGGER.info("Explicit startup qualification complete; HTTP was not started")


def _serve_model(args: argparse.Namespace, *, model_definition_source=None) -> None:
    """Build the selected fail-closed engine before advertising HTTP readiness."""

    if args.device is None:
        args.device = f"cuda:{_parse_devices(args.devices, args.tp_size * args.pp_size)[0]}"

    startup_args = vars(args).copy()
    if startup_args.get("api_key") is not None:
        startup_args["api_key"] = "<redacted>"
    _LOGGER.info("startup args=%s", _canonical_json(startup_args))
    _LOGGER.info("cubic-serve version=%s", __version__)

    operator_config_path = getattr(args, "operator_config", None)
    operator_config = _serve_operator_config(args)
    topology = (args.tp_size, args.pp_size, args.ep_size)
    if any(type(size) is not int or size <= 0 for size in topology):
        raise ValueError("TP, PP, and EP sizes must be positive integers")
    if args.execution_mode == "eager" and topology != (1, 1, 1):
        raise RuntimeError("the public eager server currently supports only TP=PP=EP=1")
    if args.execution_mode == "cuda" and (
        args.pp_size not in (1, 2) or args.tp_size % args.ep_size
    ):
        raise RuntimeError(
            "the cuda server topology is unsupported: "
            f"requested TP={args.tp_size}, PP={args.pp_size}, EP={args.ep_size} "
            f"with --devices={args.devices}. "
            "This execution path requires PP1/2 and EP dividing TP."
        )
    if not 0 <= args.port <= 65535:
        raise ValueError("--port must be between 0 and 65535")
    if args.http_keep_alive_seconds <= 0:
        raise ValueError("--http-keep-alive-seconds must be positive")
    if args.token_budget <= 0 or (
        args.cuda_graph_max_tokens != "auto" and args.cuda_graph_max_tokens <= 0
    ) or (
        args.prefill_chunk_tokens is not None and args.prefill_chunk_tokens <= 0
    ):
        raise ValueError("scheduler token budgets must be positive")
    if args.max_model_len is not None and args.max_model_len <= 0:
        raise ValueError("--max-model-len must be positive")
    if args.max_running_requests <= 0:
        raise ValueError("--max-running-requests must be positive")
    if args.execution_mode == "cuda":
        # The HTTP event loop and continuous-batch dispatcher intentionally
        # share one process.  Keep the interpreter's default 5 ms GIL quantum
        # from becoming visible as a full decode bubble under many SSE clients.
        sys.setswitchinterval(0.001)
    devices = (
        _parse_devices(args.devices, args.tp_size * args.pp_size)
        if args.execution_mode == "cuda" else None
    )
    speculative = _parse_speculative_config(args.speculative_config)
    num_speculative_tokens = speculative.initial_tokens
    if num_speculative_tokens and (
        args.execution_mode != "cuda" or args.pp_size != 1
    ):
        raise ValueError("MTP serving currently requires cuda mode with PP1")

    try:
        import uvicorn
    except ImportError as error:
        raise RuntimeError("the serving transport requires uvicorn==0.52.1") from error

    from cubic_serve.api import CubicASGIApp
    from cubic_serve.api.server import ChatBackend
    from cubic_serve.executor.runner import configure_reference_math
    from cubic_serve.generation.xgrammar import load_xgrammar_constraint_factory
    from cubic_serve.models.builtin import (
        builtin_model_definition_for_type,
    )

    family = socket.AF_INET6 if ":" in args.host else socket.AF_INET
    try:
        server_socket = socket.create_server(
            (args.host, args.port),
            family=family,
            backlog=2048,
        )
    except OSError as error:
        detail = error.strerror or str(error)
        raise OSError(
            error.errno,
            f"cannot bind {args.host}:{args.port}: {detail}",
        ) from error
    executor = None
    vision = None
    try:
        from cubic_serve.models.quantization import QuantizationConfig

        raw_config = json.loads((Path(args.model) / "config.json").read_text())
        definition = (builtin_model_definition_for_type(raw_config.get("model_type"))
                      if model_definition_source is None else model_definition_source.definition)
        quantized = QuantizationConfig.from_model_config(raw_config) is not None
        # Model declarations select their existing admission capability.
        text_graph = args.execution_mode in ("debug", "cuda") and (
            quantized or definition.reference_admission is None
            or model_definition_source is not None
        )
        if not text_graph and (args.k_cache_dtype != "auto" or args.v_cache_dtype != "auto"
                               or args.enable_kv_cache_rotation):
            raise ValueError("KV cache storage options require the shared text graph runtime")
        if (
            operator_config is not None
            and operator_config.compute_mode != "wna16"
            and not quantized
        ):
            raise ValueError("non-default packed compute requires a packed checkpoint")
        if (
            operator_config_path is not None
            and args.execution_mode != "cuda"
            and not quantized
        ):
            raise ValueError(
                "--operator-config requires cuda mode or quantized eager execution"
            )
        compute_dtype = getattr(args, "dtype", "auto")
        if args.execution_mode == "eager" and args.gpu_memory_bytes is not None:
            raise ValueError("--gpu-memory-gib requires the CUDA/debug serving runtime")
        if args.startup_numerical_audit and (
            not text_graph or args.execution_mode != "cuda"
        ):
            raise ValueError("--startup-numerical-audit requires the shared text cuda path")
        if quantized and not text_graph and num_speculative_tokens:
            raise ValueError(
                "packed eager draft serving is not qualified"
            )
        if not quantized and compute_dtype in ("fp16", "float16"):
            raise ValueError(
                "FP16 execution for the unquantized sealed runtime is not yet qualified"
            )
        if (
            args.execution_mode == "eager"
            and args.mamba_ssm_cache_dtype != "auto"
            and not quantized
        ):
            raise ValueError("--recurrent-state-dtype currently requires cuda mode")
        backend: ChatBackend
        effective_context_tokens: int | None
        if args.execution_mode == "debug" and not text_graph:
            raise ValueError("debug currently requires a quantized checkpoint")
        if quantized or text_graph:
            factory = (
                _text_graph_backend if args.execution_mode in ("debug", "cuda")
                else _quantized_eager_backend
            )
            backend, executor, model_id = factory(
                args, operator_config,
                **({} if model_definition_source is None else
                   {"model_definition_source": model_definition_source}),
            )
            effective_context_tokens = executor.max_sequence_tokens
        else:
            admission_started = time.perf_counter()
            _LOGGER.info("checkpoint metadata started model=%s", args.model)
            if definition.reference_admission is None:
                raise ValueError("model has no reference admission binding")
            admission = definition.reference_admission(args.model, verify_payloads=False)
            _LOGGER.info(
                "checkpoint metadata complete duration_s=%.3f",
                time.perf_counter() - admission_started,
            )
            descriptor = admission.adapter.descriptor
            model_id = descriptor.model_spec.model_id
            weight_variant = descriptor.weight_variant
            input_factory, output_factory = definition.frontend_factories(multimodal=True)
            codec = output_factory(
                admission.model_root,
                chat_template_path=args.chat_template,
            )
            if args.reasoning_parser or args.tool_call_parser:
                codec.admit_parser_recipe(
                    reasoning_parser=args.reasoning_parser or False,
                    tool_call_parser=args.tool_call_parser or False,
                )
            configure_reference_math()
            memory_budgets = {}
            if args.execution_mode == "cuda":
                import torch
                frontend = torch.device(args.device or f"cuda:{devices[0]}")
                frontend_index = (frontend.index if frontend.index is not None
                                  else 0)
                memory_budgets = _device_memory_budgets(args, (*devices, frontend_index))
            from cubic_serve.inputs.multimodal import MultimodalInputPolicy
            multimodal_policy = MultimodalInputPolicy(
                encoder_memory_bytes=args.multimodal_memory_bytes,
                preprocess_workers=args.mm_preprocess_workers,
                feature_cache_bytes=args.mm_feature_cache_bytes,
                video_frames_per_second=args.video_fps,
                video_max_frames=args.video_max_frames)
            encoder_reserves = {}
            if args.execution_mode == "cuda":
                vision, encoder_reserves = _load_serving_vision(
                    args, devices, operator_config, memory_budgets, multimodal_policy)
            else:
                if definition.reference_input_encoder is None:
                    raise ValueError("model has no reference input encoder binding")
                vision = definition.reference_input_encoder(
                    admission.model_root, device=args.device,
                )
            processor = input_factory(
                admission.model_root, vision=vision, chat_template_path=args.chat_template,
                allowed_local_media_path=args.allowed_local_media_path,
                resource_policy=multimodal_policy)
            effective_context_tokens = args.max_model_len
            if args.execution_mode == "eager":
                from cubic_serve.runtime.chat import ReferenceChatBackend
                from cubic_serve.runtime.engine import ReferenceEngine

                effective_context_tokens = min(
                    admission.max_sequence_tokens,
                    args.max_model_len or admission.max_sequence_tokens,
                )

                constraint_factory = load_xgrammar_constraint_factory(
                    processor.encoded_vocabulary(vocab_size=admission.vocabulary_size),
                    vocab_type=getattr(codec, "xgrammar_vocab_type", None),
                    vocab_size=admission.vocabulary_size,
                    stop_token_ids=codec.eos_token_ids,
                    tool_model_format=getattr(codec, "tool_model_format", None),
                    tool_reasoning=getattr(codec, "tool_reasoning", True),
                    special_token_ids=getattr(processor, "constraint_special_token_ids", ()),
                )
                engine = ReferenceEngine(
                    admission.adapter,
                    kv_layout=admission.kv_layout,
                    weight_policy=admission.weight_policy,
                    checkpoint_weights=admission.checkpoint_weights,
                    model_root=admission.model_root,
                    owned_layer_ids=admission.owned_layer_ids,
                    device=args.device,
                    token_budget=args.token_budget,
                    prefill_chunk_tokens=args.prefill_chunk_tokens,
                    eos_token_ids=codec.eos_token_ids,
                    operator_config=operator_config,
                )
                backend = ReferenceChatBackend(
                    engine,
                    max_sequence_tokens=effective_context_tokens,
                    processor=processor,
                    output_codec=codec,
                    enable_thinking=args.thinking,
                    reasoning_parser=args.reasoning_parser or False,
                    tool_call_parser=args.tool_call_parser or False,
                    use_prefix_cache=args.prefix_cache,
                    constraint_factory=constraint_factory,
                    **_structured_outputs_kwargs(args),
                )
            else:
                assert devices is not None
                from cubic_serve.runtime.performance_chat import PerformanceChatBackend

                executor = _start_qwen3_5_performance_executor(
                    args.model,
                    checkpoint_weights=admission.checkpoint_weights,
                    model_id=model_id,
                    weight_variant=weight_variant,
                    devices=devices,
                    eos_token_ids=codec.eos_token_ids,
                    max_sequence_tokens=args.max_model_len,
                    max_running_requests=args.max_running_requests,
                    tp_size=args.tp_size,
                    pp_size=args.pp_size,
                    ep_size=args.ep_size,
                    num_speculative_tokens=num_speculative_tokens,
                    max_speculative_tokens=descriptor.decode_extension.max_proposal_tokens,
                    dynamic_max_speculative_tokens=speculative.max_tokens,
                    dynamic_speculation=speculative.dynamic,
                    use_prefix_cache=args.prefix_cache,
                    mamba_ssm_cache_dtype=args.mamba_ssm_cache_dtype,
                    prefill_chunk_tokens=args.prefill_chunk_tokens,
                    operator_config=operator_config,
                    draft_policy=speculative.policy,
                    device_memory_budgets=memory_budgets,
                    input_encoder_reserves=encoder_reserves,
                    input_encoder=vision,
                )
                effective_context_tokens = executor.max_sequence_tokens
                backend = PerformanceChatBackend(
                    executor,
                    model_id=model_id,
                    weight_variant=weight_variant,
                    processor=processor,
                    output_codec=codec,
                    enable_thinking=args.thinking,
                    reasoning_parser=args.reasoning_parser or False,
                    tool_call_parser=args.tool_call_parser or False,
                    **_structured_outputs_kwargs(args),
                )
        served_model_names = getattr(args, "served_model_name", None)
        app = CubicASGIApp(
            backend, model_id=model_id, served_model_name=served_model_names,
            api_key=getattr(args, "api_key", None),
        )
        public_model_id = served_model_names[0] if served_model_names else model_id
        config = uvicorn.Config(
            app,
            host=args.host,
            port=args.port,
            interface="asgi3",
            log_level="info",
            log_config=None,
            access_log=False,
            timeout_keep_alive=args.http_keep_alive_seconds,
        )
        def startup_summary() -> None:
            _LOGGER.info("=" * 64)
            host = f"[{args.host}]" if ":" in args.host else args.host
            _LOGGER.info("serve endpoint: http://%s:%d", host, args.port)
            _LOGGER.info("serve model: %s", public_model_id)
            if effective_context_tokens is not None:
                context_source = "auto" if args.max_model_len is None else "explicit"
                _LOGGER.info(
                    "serve context: %d tokens (%s)",
                    effective_context_tokens,
                    context_source,
                )
            kv_capacity = getattr(executor, "kv_capacity_tokens", None)
            if kv_capacity is not None:
                vram_capacity = getattr(executor, "kv_vram_capacity_tokens", None)
                if getattr(executor,"kv_capacity_shared",False):
                    _LOGGER.info("serve cache capacity: %s tokens",
                                 f"{kv_capacity:,}")
                elif vram_capacity is None:
                    _LOGGER.info("serve KV capacity: %s tokens", f"{kv_capacity:,}")
                else:
                    _LOGGER.info("serve KV capacity: %s tokens (VRAM max≈%s)",
                                 f"{kv_capacity:,}", f"{vram_capacity:,}")
            else:
                _LOGGER.info("serve KV capacity: not preallocated (eager mode)")
            _LOGGER.info("Application startup complete.")
            _LOGGER.info("=" * 64)

        with suppress(KeyboardInterrupt):
            _run_http_server(config, server_socket, startup_summary=startup_summary)
    finally:
        if executor is not None:
            executor.close()
        close_encoder = getattr(vision, "close", None)
        if close_encoder is not None:
            close_encoder()
        server_socket.close()


def _main(argv: Sequence[str], *, args: argparse.Namespace | None = None) -> int:
    if args is None:
        args = build_parser().parse_args(argv)
    try:
        if args.command in ('serve', 'qualify') and args.execution_mode in ('cuda', 'debug'):
            require_host_build_prerequisites()
        if args.command == "inspect-model":
            print(_canonical_json(_model_payload(args.model_code)))
            return 0
        if args.command == "reference-qwen3-5":
            print(_canonical_json(_reference_qwen_payload(args)))
            return 0
        if args.command == "serve":
            with _serve_interrupts():
                _serve_qwen3_5(args)
            return 0
        if args.command == "qualify":
            with _serve_interrupts():
                _qualify_startup(args)
                return 0
            # The shared shutdown scope consumes interrupts after cleanup.
            # Exiting that scope without finishing is not qualification success.
            return 130
    except RuntimePrerequisiteError as error:
        _LOGGER.error('%s', error)
        return 2
    except (ModelAdapterLoadError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 2


def main(argv: Sequence[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    args = build_parser().parse_args(arguments)
    if args.command in ("serve", "qualify"):
        from cubic_serve.runtime.logging import parse_log_config, serve_logging

        try:
            with serve_logging(parse_log_config(args.log_config)):
                return _main(arguments, args=args)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            print(f"error: {error}", file=sys.stderr)
            return 2
    return _main(arguments, args=args)


if __name__ == "__main__":
    raise SystemExit(main())
