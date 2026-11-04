"""Qwen text and multimodal input recipes with shared visual math."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from importlib.metadata import version
from itertools import groupby
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as functional
from jinja2 import Template, TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer  # type: ignore[import-untyped]
from torch import Tensor

from cubic_serve.contracts.model import WeightVariantMeta
from cubic_serve.contracts.request import (
    CanonicalChatRequest,
    ChatContentPart,
    ChatMessage,
    ChatRole,
    ChatTemplateKwargs,
    ChatToolCall,
    ContentPart,
    DecodePositionPlan,
    EncoderJob,
    InputPlan,
    StructuredOutputKind,
    StructuredOutputMeta,
    TokenPlacement,
    TokenPosition,
)
from cubic_serve.inputs import EncodedFeature, EncodedFeatureBundle, EncodedFeatureCache
from cubic_serve.inputs.chat import (
    ChatProcessorError,
    DecodedMedia,
    MediaResolver,
    ReasoningEffort,
    resolve_chat_template_kwargs,
)
from cubic_serve.inputs.multimodal import MultimodalInputPolicy
from cubic_serve.models.parallel_weights import AxisSpan
from cubic_serve.models.protocol import PreparedInput
from cubic_serve.models.qwen3_5.model import VisionConfig
from cubic_serve.models.qwen3_5.weights import _add_vision

_LOGGER = logging.getLogger(__name__)


_TEXT_RECIPE_KIND = "quanttrio.qwen3_5.text_input.v1"


_REQUIRED_TOKEN_IDS = {
    "<|im_start|>": 248045,
    "<|im_end|>": 248046,
    "<|vision_start|>": 248053,
    "<|vision_end|>": 248054,
    "<|image_pad|>": 248056,
    "<|video_pad|>": 248057,
    "<think>": 248068,
    "</think>": 248069,
}


_ROLES = frozenset(role.value for role in ChatRole)


_MESSAGE_KEYS = frozenset({"role", "content"})
_MESSAGE_METADATA_KEYS = frozenset(
    {"name", "tool_calls", "tool_call_id", "reasoning", "reasoning_content"}
)


def _content_digest_file(path: Path) -> str:
    try:
        with path.open("rb") as file:
            digest = hashlib.file_digest(file, "blake2s").hexdigest()
    except OSError as error:
        raise ChatProcessorError(f"cannot read processor file {path}: {error}") from error
    return f"b2s256:{digest}"


def _raise_template_error(message: str) -> None:
    raise TemplateError(message)


def _to_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        indent=None,
        separators=None,
        sort_keys=False,
    )


def _normalize_messages(
    messages: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    if not messages:
        raise ChatProcessorError("at least one chat message is required")

    normalized: list[dict[str, object]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise ChatProcessorError(f"message {index} must be a mapping")
        unknown = set(message) - _MESSAGE_KEYS - _MESSAGE_METADATA_KEYS
        missing = _MESSAGE_KEYS - set(message)
        if unknown or missing:
            raise ChatProcessorError(
                f"message {index} fields differ from role/content: "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}"
            )

        role = message["role"]
        content = message["content"]
        if not isinstance(role, str) or role not in _ROLES:
            raise ChatProcessorError(
                f"message {index} has unsupported role; expected system, user, or assistant"
            )
        if not isinstance(content, str):
            raise ChatProcessorError(
                f"message {index} uses structured or media content; "
                "the text processor accepts strings only"
            )
        fields: dict[str, object] = {"role": role, "content": content}
        if set(message) & _MESSAGE_METADATA_KEYS or role == "tool":
            try:
                raw_calls = message.get("tool_calls") or []
                if type(raw_calls) is not list:
                    raise ValueError("tool_calls must be an array")
                calls = tuple(
                    ChatToolCall.from_arguments(
                        call["id"],
                        call["function"]["name"],
                        call["function"].get("arguments") or {},
                    )
                    for call in raw_calls
                )
                canonical = ChatMessage(
                    ChatRole(role),
                    (ChatContentPart(0, "text", text=content),),
                    calls,
                    cast(str | None, message.get("tool_call_id")),
                    cast(str | None, message.get("reasoning", message.get("reasoning_content"))),
                    cast(str | None, message.get("name")),
                )
                fields.update(canonical.template_metadata())
            except (TypeError, ValueError, KeyError) as error:
                raise ChatProcessorError(
                    "message has invalid tool or reasoning metadata"
                ) from error
        normalized.append(fields)
    return tuple(normalized)


class Qwen3_5TextProcessor:
    """Render and tokenize local Qwen3.8 text chats without loading model weights."""

    __slots__ = (
        "_chat_template",
        "_file_digests",
        "_runtime_versions",
        "_tokenizer",
    )

    def __init__(
        self,
        model_root: str | Path,
        *,
        chat_template_path: str | Path | None = None,
    ) -> None:
        root = Path(model_root)
        tokenizer_path = root / "tokenizer.json"
        tokenizer_config_path = root / "tokenizer_config.json"
        template_path = (
            Path(chat_template_path)
            if chat_template_path is not None
            else root / "chat_template.jinja"
        )
        for name, path in (
            ("tokenizer", tokenizer_path),
            ("tokenizer config", tokenizer_config_path),
            ("chat template", template_path),
        ):
            if not path.is_file():
                raise ChatProcessorError(f"{name} must be a local file: {path}")

        try:
            tokenizer = Tokenizer.from_file(str(tokenizer_path))
        except Exception as error:
            raise ChatProcessorError(
                f"cannot load tokenizer.json from {tokenizer_path}: {error}"
            ) from error
        for token, expected_id in _REQUIRED_TOKEN_IDS.items():
            observed_id = tokenizer.token_to_id(token)
            if observed_id != expected_id:
                raise ChatProcessorError(
                    f"Qwen3.8 tokenizer token {token!r} is {observed_id}, expected {expected_id}"
                )

        try:
            template_source = template_path.read_bytes().decode("utf-8")
            environment = ImmutableSandboxedEnvironment(
                trim_blocks=True,
                lstrip_blocks=True,
                extensions=("jinja2.ext.loopcontrols",),
            )
            environment.filters["tojson"] = _to_json
            environment.globals["raise_exception"] = _raise_template_error
            chat_template = environment.from_string(template_source)
        except (OSError, UnicodeDecodeError, TemplateError) as error:
            raise ChatProcessorError(
                f"cannot load chat template from {template_path}: {error}"
            ) from error

        self._tokenizer = tokenizer
        self._chat_template: Template = chat_template
        self._file_digests = {
            "tokenizer.json": _content_digest_file(tokenizer_path),
            "tokenizer_config.json": _content_digest_file(tokenizer_config_path),
            "chat_template": _content_digest_file(template_path),
        }
        self._runtime_versions = {
            "jinja2": version("jinja2"),
            "tokenizers": version("tokenizers"),
        }

    @property
    def constraint_special_token_ids(self) -> tuple[int, ...]:
        return tuple(
            i for i, token in self._tokenizer.get_added_tokens_decoder().items() if token.special
        )

    def encoded_vocabulary(self, *, vocab_size: int) -> tuple[str | bytes, ...]:
        """Expose the admitted tokenizer once for request-constraint compilation."""

        vocabulary = cast(dict[str, int], self._tokenizer.get_vocab(with_added_tokens=True))
        if type(vocab_size) is not int or vocab_size <= max(vocabulary.values()):
            raise ChatProcessorError("model vocab_size must cover every tokenizer token id")
        encoded = [""] * vocab_size
        for token, token_id in vocabulary.items():
            encoded[token_id] = token
        return tuple(encoded)

    def render_messages(
        self,
        messages: Sequence[Mapping[str, object]],
        *,
        add_generation_prompt: bool = True,
        enable_thinking: bool = True,
        reasoning_effort: ReasoningEffort = "xhigh",
        preserve_thinking: bool = True,
        structured_output: StructuredOutputMeta | None = None,
        chat_template_kwargs: Mapping[str, object] | None = None,
    ) -> str:
        normalized = _normalize_messages(messages)
        settings = self._settings(
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
            reasoning_effort=reasoning_effort,
            preserve_thinking=preserve_thinking,
            structured_output=structured_output or StructuredOutputMeta.identity(),
            chat_template_kwargs=chat_template_kwargs,
        )
        return self._render(normalized, settings)

    def build_input_plan(
        self,
        *,
        request_id: str,
        model_id: str,
        weight_variant: WeightVariantMeta,
        messages: Sequence[Mapping[str, object]],
        add_generation_prompt: bool = True,
        enable_thinking: bool = True,
        reasoning_effort: ReasoningEffort = "xhigh",
        preserve_thinking: bool = True,
        structured_output: StructuredOutputMeta | None = None,
        chat_template_kwargs: Mapping[str, object] | None = None,
    ) -> InputPlan:
        normalized = _normalize_messages(messages)
        settings = self._settings(
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
            reasoning_effort=reasoning_effort,
            preserve_thinking=preserve_thinking,
            structured_output=structured_output or StructuredOutputMeta.identity(),
            chat_template_kwargs=chat_template_kwargs,
        )
        rendered = self._render(normalized, settings)
        try:
            token_ids = tuple(self._tokenizer.encode(rendered, add_special_tokens=False).ids)
        except Exception as error:
            raise ChatProcessorError(f"tokenizer.json encoding failed: {error}") from error
        if not token_ids:
            raise ChatProcessorError("chat template produced no input tokens")
        positions = tuple(TokenPosition((index, index, index)) for index in range(len(token_ids)))
        first_decode_axis = len(token_ids)
        return InputPlan(
            request_id=request_id,
            model_id=model_id,
            processor_recipe_digest=self._recipe_digest(settings),
            weight_variant=weight_variant,
            content_parts=tuple(
                ContentPart(index, "text", text=message["content"])
                for index, message in enumerate(normalized)
            ),
            token_ids=token_ids,
            positions=positions,
            decode_positions=DecodePositionPlan(
                first=TokenPosition((first_decode_axis, first_decode_axis, first_decode_axis)),
                step_axes=(1, 1, 1),
            ),
        )

    def build_chat_input_plan(
        self,
        request: CanonicalChatRequest,
        *,
        weight_variant: WeightVariantMeta,
        add_generation_prompt: bool = True,
        enable_thinking: bool = True,
        reasoning_effort: ReasoningEffort = "xhigh",
        preserve_thinking: bool = True,
    ) -> InputPlan:
        """Realize one public canonical request through the same text recipe."""

        if not isinstance(request, CanonicalChatRequest):
            raise TypeError("Qwen text processor requires a CanonicalChatRequest")
        template_kwargs, resolved_enable_thinking = resolve_chat_template_kwargs(
            request,
            default_enable_thinking=enable_thinking,
        )
        messages: list[dict[str, object]] = []
        for message in request.messages:
            if any(part.modality != "text" for part in message.content_parts):
                raise ChatProcessorError(
                    "the Qwen text processor cannot realize image or video content"
                )
            messages.append(
                {
                    **message.template_metadata(),
                    "role": message.role.value,
                    "content": "".join(
                        part.text if part.text is not None else "" for part in message.content_parts
                    ),
                }
            )
        return self.build_input_plan(
            request_id=request.request_id,
            model_id=request.model_id,
            weight_variant=weight_variant,
            messages=messages,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=resolved_enable_thinking,
            reasoning_effort=reasoning_effort,
            preserve_thinking=preserve_thinking,
            structured_output=request.structured_output,
            chat_template_kwargs=template_kwargs,
        )

    @staticmethod
    def _settings(
        *,
        add_generation_prompt: bool,
        enable_thinking: bool,
        reasoning_effort: ReasoningEffort,
        preserve_thinking: bool,
        structured_output: StructuredOutputMeta,
        chat_template_kwargs: Mapping[str, object] | None,
    ) -> dict[str, object]:
        booleans = {
            "add_generation_prompt": add_generation_prompt,
            "enable_thinking": enable_thinking,
            "preserve_thinking": preserve_thinking,
        }
        if any(type(value) is not bool for value in booleans.values()):
            raise ChatProcessorError("chat template boolean settings must be bool values")
        settings: dict[str, object] = {
            **booleans,
            "reasoning_effort": reasoning_effort,
            "add_vision_id": False,
            "documents": None,
            "tools": None,
        }
        if chat_template_kwargs is not None:
            if not isinstance(chat_template_kwargs, Mapping) or any(
                type(key) is not str for key in chat_template_kwargs
            ):
                raise ChatProcessorError("chat_template_kwargs must be a mapping with string keys")
            settings.update(chat_template_kwargs)
        if type(settings.get("enable_thinking")) is not bool:
            raise ChatProcessorError("chat template enable_thinking must be a boolean")
        if structured_output.kind is not StructuredOutputKind.IDENTITY:
            settings.update(
                {
                    "structured_output_digest": structured_output.digest,
                    "structured_output_kind": structured_output.kind.value,
                }
            )
        if structured_output.kind is StructuredOutputKind.TOOL_SCHEMA and settings["tools"] is None:
            payload = json.loads(structured_output.canonical_payload)
            settings["tools"] = payload["tools"]
        return settings

    def _render(
        self,
        messages: tuple[dict[str, object], ...],
        settings: Mapping[str, object],
    ) -> str:
        try:
            context = dict(settings)
            context["messages"] = messages
            rendered = self._chat_template.render(**context)
        except TemplateError as error:
            raise ChatProcessorError(f"chat template rejected messages: {error}") from error
        if not rendered:
            raise ChatProcessorError("chat template produced an empty prompt")
        return rendered

    def _recipe_digest(self, settings: Mapping[str, object]) -> str:
        recipe = {
            "files": self._file_digests,
            "kind": _TEXT_RECIPE_KIND,
            "plan": {
                "content_parts": "one_text_part_per_message",
                "decode_step_axes": [1, 1, 1],
                "position_axes": "text_index_repeated_across_three_mrope_axes",
            },
            "renderer": {
                "environment": "jinja2.ImmutableSandboxedEnvironment",
                "lstrip_blocks": True,
                "trim_blocks": True,
            },
            "runtime_versions": self._runtime_versions,
            "settings": dict(settings),
        }
        encoded = json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()
        return f"b2s256:{hashlib.blake2s(encoded).hexdigest()}"


@dataclass(frozen=True, slots=True)
class VisionTensorParallel:
    """Whole attention heads and paired column/row projection ownership."""

    size: int = 1
    rank: int = 0
    row_projections: frozenset[str] = frozenset()

    def __post_init__(self):
        if (type(self.size) is not int or self.size <= 0 or type(self.rank) is not int
                or not 0 <= self.rank < self.size):
            raise ValueError("vision TP requires a positive size and valid rank")


def vision_weight_requests(checkpoint, vision, *, size=1, rank=0):
    """Slice model coordinates; the checkpoint owns packed-code materialization.

    QKV keeps Q/K/V checkpoint order and complete heads. MLP and merger use
    matching column/row ranges. Encoded input-group boundaries guide the cuts,
    including a final partial group; no expert or logical channel is reordered.
    """
    parallel = VisionTensorParallel(size, rank)
    if vision.head_count % size:
        raise ValueError("vision attention heads must divide the requested TP size")
    weights = {w.logical.name: w for w in checkpoint.weights if w.logical.partition == "vision"}
    if size == 1:
        return tuple((w, 0, (AxisSpan(0, w.logical.shape[0]),)) for w in weights.values()), parallel
    owners = {}
    rows = set()

    def paired(column, row, width, alignment=1):
        units = math.ceil(width / alignment)
        if units < size:
            raise ValueError("vision projection has fewer owner units than TP ranks")
        span = AxisSpan(min(width, units * rank // size * alignment),
                        min(width, units * (rank + 1) // size * alignment))
        owners[column + ".weight"] = 0, (span,)
        owners[column + ".bias"] = 0, (span,)
        owners[row + ".weight"] = 1, (span,)
        rows.add(row)

    prefix = "model.visual"
    head_width = vision.hidden_size // size
    for layer in range(vision.layer_count):
        block = f"{prefix}.blocks.{layer}"
        spans = tuple(AxisSpan(part * vision.hidden_size + rank * head_width,
                               part * vision.hidden_size + (rank + 1) * head_width)
                      for part in range(3))
        owners[block + ".attn.qkv.weight"] = 0, spans
        owners[block + ".attn.qkv.bias"] = 0, spans
        owners[block + ".attn.proj.weight"] = 1, (AxisSpan(rank * head_width,
                                                               (rank + 1) * head_width),)
        rows.add(block + ".attn.proj")
        down = weights[block + ".mlp.linear_fc2.weight"]
        alignment = min(vision.intermediate_size, down.encoding.group_in) if down.encoding else 1
        paired(block + ".mlp.linear_fc1", block + ".mlp.linear_fc2",
               vision.intermediate_size, alignment)
    merger = prefix + ".merger"
    width = vision.hidden_size * vision.spatial_merge_size**2
    down = weights[merger + ".linear_fc2.weight"]
    alignment = min(width, down.encoding.group_in) if down.encoding else 1
    paired(merger + ".linear_fc1", merger + ".linear_fc2", width, alignment)
    requests = tuple((w, *owners.get(name, (0, (AxisSpan(0, w.logical.shape[0]),))))
                     for name, w in weights.items())
    return requests, VisionTensorParallel(parallel.size, parallel.rank, frozenset(rows))

if TYPE_CHECKING:
    from cubic_serve.models.builtin import BuiltinModelConfig
    from cubic_serve.models.quantized_checkpoint import QuantizedMatrix


class Qwen3_5VisionLoadError(RuntimeError):
    """The checkpoint cannot become the closed eager vision program."""


class Qwen3_5VisionWeights:
    """Read-only plain or encoded weights for exactly the vision partition."""

    __slots__ = (
        "_tensors",
        "checkpoint_header_digest",
        "device",
        "semantic_plan_digest",
    )

    def __init__(
        self,
        *,
        tensors: dict[str, Tensor | QuantizedMatrix],
        device: torch.device,
        plan,
    ) -> None:
        self._tensors = MappingProxyType(tensors)
        self.device = device
        self.checkpoint_header_digest = plan.checkpoint_header_digest
        self.semantic_plan_digest = plan.semantic_plan_digest

    def __getitem__(self, name: str) -> Tensor | QuantizedMatrix:
        try:
            return self._tensors[name]
        except KeyError as error:
            raise Qwen3_5VisionLoadError(
                f"eager vision graph requested unloaded weight {name!r}"
            ) from error

    @property
    def tensor_count(self) -> int:
        return len(self._tensors)

    def named_tensors(self) -> tuple[tuple[str, Tensor | QuantizedMatrix], ...]:
        return tuple((name, self._tensors[name]) for name in sorted(self._tensors))


def _parse_qwen3_5_family_config(document: bytes):
    """Select model geometry independently of its checkpoint encoding."""

    # Import after model package initialization; builtin imports this module.
    from cubic_serve.models.builtin import parse_builtin_model_config_json

    return parse_builtin_model_config_json(document)


def load_qwen3_5_vision_reference(
    model_root: str | Path,
    *,
    device: str | torch.device = "cuda:0",
    packed_device: str | torch.device | None = None,
    checkpoint=None,
    operators=None,
    tp_size=1,
    tp_rank=0,
    reduce=None,
    progress=None,
    model_config: BuiltinModelConfig | None = None,
) -> Qwen3_5VisionReferenceModel:
    """Validate all headers, then load only the declared vision tensor payloads."""

    root = Path(model_root).resolve(strict=True)
    if not root.is_dir():
        raise Qwen3_5VisionLoadError("Qwen model root must be a directory")
    started = time.perf_counter()
    _LOGGER.info("vision checkpoint metadata started model=%s", root)
    try:
        config = (_parse_qwen3_5_family_config((root / "config.json").read_bytes())
                  if model_config is None else model_config)
        if checkpoint is None:
            from cubic_serve.models.quantized_checkpoint import CheckpointWeights
            from cubic_serve.models.qwen3_5.weights import logical_quantized_weights

            if config.identity.model_type == "qwen4_exp":
                from cubic_serve.models.qwen4_exp.weights import logical_quantized_weights
            elif config.identity.model_type == "qwen3_5_moe":
                from cubic_serve.models.qwen3_5_moe.weights import logical_quantized_weights
            checkpoint = CheckpointWeights.admit(
                root, logical_quantized_weights(config), verify_payloads=False,
                require_quantization=False,
            )
    except (OSError, TypeError, ValueError) as error:
        raise Qwen3_5VisionLoadError("Qwen vision checkpoint admission failed") from error
    _LOGGER.info("vision checkpoint metadata complete duration_s=%.3f",
                 time.perf_counter() - started)

    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise Qwen3_5VisionLoadError("CUDA was requested but is unavailable")
    from types import SimpleNamespace

    requests, parallel = vision_weight_requests(
        checkpoint, getattr(config, "graph", config).vision, size=tp_size, rank=tp_rank)
    started = time.perf_counter()
    tensors = checkpoint.materialize_many(
        requests, device=target, compute_dtype=torch.bfloat16,
        packed_device=packed_device,
        progress=progress)
    _LOGGER.info("vision weight loading complete device=%s duration_s=%.3f",
                 target, time.perf_counter() - started)
    plan = SimpleNamespace(
        checkpoint_header_digest=checkpoint.inventory.inventory_digest,
        semantic_plan_digest=checkpoint.semantic_digest,
    )
    weights = Qwen3_5VisionWeights(tensors=tensors, device=target, plan=plan)
    model = Qwen3_5VisionReferenceModel(config=config, weights=weights, operators=operators,
                                       parallel=parallel, reduce=reduce)
    if target.type == "cuda":
        started = time.perf_counter()
        _LOGGER.info("vision kernel warmup and calibration started device=%s", target)
        with torch.cuda.device(target):
            model.warmup()
        _LOGGER.info("vision kernel warmup complete device=%s duration_s=%.3f",
                     target, time.perf_counter() - started)
    return model


class Qwen3_5VisionReferenceModel:
    """Shared stateless Qwen vision tower and its eager reference traversal."""

    __slots__ = ("_inv_frequency", "config", "operators", "vision_config", "weights",
                 "parallel", "reduce")

    def __init__(self, *, config: BuiltinModelConfig, weights: Qwen3_5VisionWeights,
                 operators=None, parallel=None, reduce=None) -> None:
        graph = getattr(config, "graph", config)
        vision = graph.vision
        if vision.deepstack_layer_indexes:
            raise ValueError("shared reference vision requires a tower without deepstack")
        expected: set[str] = set()
        def add(name, shape, node, partition, dtype="BF16"):
            expected.add(name)
        if config.identity.model_type == "qwen4_exp":
            from cubic_serve.models.qwen4_exp.weights import _add_vision as add_vision
            add_vision(add, config)
        else:
            _add_vision(add, graph, config.architecture_plan())
        if set(weights._tensors) != expected:
            raise ValueError("reference vision tensors differ from the configured tower")
        self.config = config
        self.weights = weights
        self.operators = operators
        self.vision_config = vision
        self.parallel = parallel or VisionTensorParallel()
        self.reduce = reduce
        if vision.head_count % self.parallel.size or (self.parallel.size > 1 and reduce is None):
            raise ValueError("vision TP requires complete heads and a rank reduction")
        rotary_dim = (vision.hidden_size // vision.head_count) // 2
        inv_frequency = 1.0 / (
            10000.0 ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
        )
        self._inv_frequency = inv_frequency.to(weights.device)

    @torch.inference_mode()
    def warmup(self) -> None:
        """Pay device-library initialization before the first public request."""

        vision = self.vision_config
        patch_width = (
            vision.input_channels
            * vision.temporal_patch_size
            * vision.patch_size
            * vision.patch_size
        )
        pixels = torch.zeros((4, patch_width), dtype=torch.float32, device=self.weights.device)
        grid = torch.tensor(((1, 2, 2),), dtype=torch.int64, device=self.weights.device)
        self.forward(pixels, grid)
        torch.cuda.synchronize(self.weights.device)

    @torch.inference_mode()
    def forward(self, pixel_values: Tensor, grid_thw: Tensor) -> Tensor:
        """Return BF16 ``[sum(t*h*w)/merge**2, decoder_hidden]`` feature rows."""

        vision = self.vision_config
        _validate_inputs(pixel_values, grid_thw, vision, self.weights.device)
        positions = _position_ids(grid_thw, vision.spatial_merge_size)
        bilinear_indices, bilinear_weights = _bilinear_position_inputs(
            grid_thw,
            num_grid_per_side=math.isqrt(vision.position_count),
            spatial_merge_size=vision.spatial_merge_size,
        )
        cu_seqlens = _cu_seqlens(grid_thw)

        return self.encode(pixel_values, positions, bilinear_indices, bilinear_weights, cu_seqlens)

    def project(self, inputs: Tensor, name: str) -> Tensor:
        weight = self.weights[name + ".weight"]
        bias = self.weights[name + ".bias"]
        row_parallel = self.parallel.size > 1 and name in self.parallel.row_projections
        if isinstance(weight, Tensor):
            if not row_parallel:
                return functional.linear(inputs, weight, bias)
            output = functional.linear(inputs, weight)
        else:
            if self.operators is None:
                raise ValueError("encoded visual projections require format-owned operators")
            output = self.operators.linear(inputs, weight)
        if row_parallel:
            output = self.reduce(output)
        return output + bias

    def encode(self, pixel_values: Tensor, positions: Tensor, bilinear_indices: Tensor,
               bilinear_weights: Tensor, cu_seqlens: Tensor | None,
               attention_mask: Tensor | None = None) -> Tensor:
        vision = self.vision_config
        prefix = "model.visual"
        hidden = functional.conv3d(
            pixel_values.view(
                -1,
                vision.input_channels,
                vision.temporal_patch_size,
                vision.patch_size,
                vision.patch_size,
            ).to(torch.bfloat16),
            self.weights[f"{prefix}.patch_embed.proj.weight"],
            self.weights[f"{prefix}.patch_embed.proj.bias"],
            stride=(vision.temporal_patch_size, vision.patch_size, vision.patch_size),
        ).view(-1, vision.hidden_size)
        position_weight = self.weights[f"{prefix}.pos_embed.weight"]
        position_rows = (functional.embedding(bilinear_indices, position_weight)
                         if self.operators is None else self.operators.embedding(
                             bilinear_indices, position_weight, dtype=torch.bfloat16))
        position = (position_rows * bilinear_weights[:, :, None]).sum(0)
        hidden = hidden + position.to(hidden.dtype)

        rotary = (positions.unsqueeze(-1) * self._inv_frequency).flatten(1)
        rotary = torch.cat((rotary, rotary), dim=-1)
        cos_sin = (rotary.cos(), rotary.sin())
        for layer_index in range(vision.layer_count):
            hidden = self._block(hidden, cu_seqlens, cos_sin, layer_index, vision, attention_mask)
        return self._merge(hidden, vision)

    def _block(
        self,
        hidden: Tensor,
        cu_seqlens: Tensor,
        cos_sin: tuple[Tensor, Tensor],
        layer_index: int,
        vision: VisionConfig,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        prefix = f"model.visual.blocks.{layer_index}"
        normed = functional.layer_norm(
            hidden,
            (vision.hidden_size,),
            self.weights[f"{prefix}.norm1.weight"],
            self.weights[f"{prefix}.norm1.bias"],
            1e-6,
        )
        hidden = hidden + self._attention(normed, cu_seqlens, cos_sin, prefix, vision,
                                         attention_mask)
        normed = functional.layer_norm(
            hidden,
            (vision.hidden_size,),
            self.weights[f"{prefix}.norm2.weight"],
            self.weights[f"{prefix}.norm2.bias"],
            1e-6,
        )
        activated = functional.gelu(
            self.project(normed, f"{prefix}.mlp.linear_fc1"),
            approximate="tanh",
        )
        return hidden + self.project(activated, f"{prefix}.mlp.linear_fc2")

    def _attention(
        self,
        hidden: Tensor,
        cu_seqlens: Tensor,
        cos_sin: tuple[Tensor, Tensor],
        prefix: str,
        vision: VisionConfig,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        token_count = hidden.shape[0]
        head_dim = vision.hidden_size // vision.head_count
        local_heads = vision.head_count // self.parallel.size
        query, key, value = (
            self.project(hidden, f"{prefix}.attn.qkv")
            .reshape(token_count, 3, local_heads, head_dim)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        query, key = _apply_rotary(query, key, *cos_sin)
        query = query.transpose(0, 1).unsqueeze(0)
        key = key.transpose(0, 1).unsqueeze(0)
        value = value.transpose(0, 1).unsqueeze(0)
        if attention_mask is None:
            lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
            chunks = [torch.split(item, lengths, dim=2) for item in (query, key, value)]
            outputs = [
                _eager_attention(q_chunk, k_chunk, v_chunk, head_dim**-0.5)
                for q_chunk, k_chunk, v_chunk in zip(*chunks, strict=True)
            ]
            attended = torch.cat(outputs, dim=1)
        else:
            # Keep the reference BF16 score and probability rounding while
            # bounding captured attention workspaces by query-row tiles.
            key_transposed = key.transpose(2, 3)
            outputs = []
            for start in range(0, token_count, 512):
                stop = min(start + 512, token_count)
                scores = torch.matmul(query[:, :, start:stop], key_transposed) * head_dim**-0.5
                scores = scores.masked_fill(~attention_mask[start:stop], float("-inf"))
                probabilities = functional.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
                outputs.append(torch.matmul(probabilities, value))
            attended = torch.cat(outputs, dim=2).transpose(1, 2)
        return self.project(attended.reshape(token_count, -1).contiguous(), f"{prefix}.attn.proj")

    def _merge(self, hidden: Tensor, vision: VisionConfig) -> Tensor:
        prefix = "model.visual.merger"
        hidden = functional.layer_norm(
            hidden,
            (vision.hidden_size,),
            self.weights[f"{prefix}.norm.weight"],
            self.weights[f"{prefix}.norm.bias"],
            1e-6,
        ).view(-1, vision.hidden_size * vision.spatial_merge_size**2)
        hidden = functional.gelu(
            self.project(hidden, f"{prefix}.linear_fc1")
        )
        return self.project(hidden, f"{prefix}.linear_fc2")


def _validate_inputs(
    pixel_values: Tensor,
    grid_thw: Tensor,
    vision: VisionConfig,
    device: torch.device,
) -> None:
    patch_width = (
        vision.input_channels * vision.temporal_patch_size * vision.patch_size * vision.patch_size
    )
    if pixel_values.ndim != 2 or pixel_values.shape[1] != patch_width:
        raise ValueError(f"pixel_values must have shape [patches, {patch_width}]")
    if not pixel_values.is_floating_point() or pixel_values.numel() == 0:
        raise TypeError("pixel_values must be a non-empty floating-point tensor")
    if grid_thw.ndim != 2 or grid_thw.shape[1] != 3 or grid_thw.numel() == 0:
        raise ValueError("grid_thw must be non-empty [items, 3]")
    if grid_thw.dtype not in {torch.int32, torch.int64}:
        raise TypeError("grid_thw must use torch.int32 or torch.int64")
    if pixel_values.device != device or grid_thw.device != device:
        raise ValueError("pixel_values, grid_thw, and vision weights must share one device")
    grids = [tuple(int(axis) for axis in item) for item in grid_thw.tolist()]
    if any(
        min(t, h, w) <= 0 or h % vision.spatial_merge_size or w % vision.spatial_merge_size
        for t, h, w in grids
    ):
        raise ValueError("grid_thw axes must be positive and spatially merge-divisible")
    if sum(t * h * w for t, h, w in grids) != pixel_values.shape[0]:
        raise ValueError("pixel patch count differs from grid_thw")


def _cu_seqlens(grid_thw: Tensor) -> Tensor:
    lengths = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])
    cumulative = lengths.cumsum(dim=0, dtype=torch.int32)
    return functional.pad(cumulative, (1, 0), value=0)


def _position_ids(grid_thw: Tensor, merge_size: int) -> Tensor:
    outputs: list[Tensor] = []
    for raw_t, raw_h, raw_w in grid_thw.tolist():
        t, h, w = int(raw_t), int(raw_h), int(raw_w)
        h_ids, w_ids = torch.meshgrid(
            torch.arange(h, device=grid_thw.device),
            torch.arange(w, device=grid_thw.device),
            indexing="ij",
        )
        block_shape = (h // merge_size, merge_size, w // merge_size, merge_size)
        h_ids = h_ids.reshape(block_shape).transpose(1, 2).flatten()
        w_ids = w_ids.reshape(block_shape).transpose(1, 2).flatten()
        outputs.append(torch.stack((h_ids, w_ids), dim=-1).repeat(t, 1))
    return torch.cat(outputs, dim=0)


def _bilinear_position_inputs(
    grid_thw: Tensor,
    *,
    num_grid_per_side: int,
    spatial_merge_size: int,
) -> tuple[Tensor, Tensor]:
    index_parts: list[list[Tensor]] = [[] for _ in range(4)]
    weight_parts: list[list[Tensor]] = [[] for _ in range(4)]
    side = num_grid_per_side
    for raw_t, raw_h, raw_w in grid_thw.tolist():
        t, h, w = int(raw_t), int(raw_h), int(raw_w)
        h_grid = torch.linspace(0, side - 1, h, device=grid_thw.device)
        w_grid = torch.linspace(0, side - 1, w, device=grid_thw.device)
        h_floor, w_floor = h_grid.int(), w_grid.int()
        h_ceil = (h_floor + 1).clamp(max=side - 1)
        w_ceil = (w_floor + 1).clamp(max=side - 1)
        h_fraction, w_fraction = h_grid - h_floor, w_grid - w_floor
        corners = (
            (h_floor[:, None] * side + w_floor[None, :]).flatten(),
            (h_floor[:, None] * side + w_ceil[None, :]).flatten(),
            (h_ceil[:, None] * side + w_floor[None, :]).flatten(),
            (h_ceil[:, None] * side + w_ceil[None, :]).flatten(),
        )
        weights = (
            ((1 - h_fraction)[:, None] * (1 - w_fraction)[None, :]).flatten(),
            ((1 - h_fraction)[:, None] * w_fraction[None, :]).flatten(),
            (h_fraction[:, None] * (1 - w_fraction)[None, :]).flatten(),
            (h_fraction[:, None] * w_fraction[None, :]).flatten(),
        )
        h_index = torch.arange(h, device=grid_thw.device).view(
            h // spatial_merge_size, spatial_merge_size
        )
        w_index = torch.arange(w, device=grid_thw.device).view(
            w // spatial_merge_size, spatial_merge_size
        )
        reorder = (
            (h_index[:, :, None, None] * w + w_index[None, None, :, :])
            .transpose(1, 2)
            .flatten()
            .repeat(t)
        )
        for corner in range(4):
            index_parts[corner].append(corners[corner][reorder])
            weight_parts[corner].append(weights[corner][reorder])
    return (
        torch.stack([torch.cat(part) for part in index_parts]),
        torch.stack([torch.cat(part) for part in weight_parts]),
    )


def _rotate_half(hidden: Tensor) -> Tensor:
    first, second = hidden.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _apply_rotary(
    query: Tensor,
    key: Tensor,
    cosine: Tensor,
    sine: Tensor,
) -> tuple[Tensor, Tensor]:
    query_dtype, key_dtype = query.dtype, key.dtype
    query, key = query.float(), key.float()
    cosine, sine = cosine.unsqueeze(-2).float(), sine.unsqueeze(-2).float()
    query = (query * cosine) + (_rotate_half(query) * sine)
    key = (key * cosine) + (_rotate_half(key) * sine)
    return query.to(query_dtype), key.to(key_dtype)


def _eager_attention(query: Tensor, key: Tensor, value: Tensor, scale: float) -> Tensor:
    scores = torch.matmul(query, key.transpose(2, 3)) * scale
    probabilities = functional.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    probabilities = functional.dropout(probabilities, p=0.0, training=False)
    return torch.matmul(probabilities, value).transpose(1, 2).contiguous()


_RECIPE_KIND = "quanttrio.qwen3_5.multimodal_input.v1"
_PLACEHOLDER = {
    "image": "<|vision_start|><|image_pad|><|vision_end|>",
    "video": "<|vision_start|><|video_pad|><|vision_end|>",
}


class Qwen3_5MultimodalError(ChatProcessorError):
    """Trusted patches cannot produce one closed Qwen3.8 input program."""


@dataclass(frozen=True, slots=True)
class Qwen3_5VideoMetadata:
    """Trusted sampling facts needed to reproduce Qwen frame timestamps."""

    frame_indices: tuple[int, ...]
    frames_per_second: float
    frame_timestamps_seconds: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if (
            not self.frame_indices
            or any(type(index) is not int or index < 0 for index in self.frame_indices)
            or tuple(sorted(self.frame_indices)) != self.frame_indices
        ):
            raise ValueError("video frame indices must be non-negative and ordered")
        if (
            isinstance(self.frames_per_second, bool)
            or not isinstance(self.frames_per_second, int | float)
            or not math.isfinite(self.frames_per_second)
            or self.frames_per_second <= 0
        ):
            raise ValueError("video frames_per_second must be finite and positive")
        if self.frame_timestamps_seconds and (
            len(self.frame_timestamps_seconds) != len(self.frame_indices)
            or any(not math.isfinite(t) or t < 0 for t in self.frame_timestamps_seconds)
            or tuple(sorted(self.frame_timestamps_seconds)) != self.frame_timestamps_seconds
        ):
            raise ValueError("video source timestamps must be finite, ordered and match sampled frames")

    def timestamps(self, temporal_patch_size: int) -> tuple[float, ...]:
        remainder = len(self.frame_indices) % temporal_patch_size
        seconds = list(self.frame_timestamps_seconds) if self.frame_timestamps_seconds else [
            index / self.frames_per_second for index in self.frame_indices]
        if remainder:
            seconds.extend(seconds[-1] for _ in range(temporal_patch_size - remainder))
        return tuple(
            (seconds[start] + seconds[start + temporal_patch_size - 1]) / 2
            for start in range(0, len(seconds), temporal_patch_size)
        )


@dataclass(frozen=True, slots=True)
class Qwen3_5MediaPatches:
    """One already-decoded and already-patchified trusted media item."""

    content_part_ordinal: int
    modality: str
    pixel_values: torch.Tensor = field(compare=False, repr=False)
    grid_thw: tuple[int, int, int]
    video_metadata: Qwen3_5VideoMetadata | None = None

    def __post_init__(self) -> None:
        if self.content_part_ordinal < 0 or self.modality not in _PLACEHOLDER:
            raise ValueError(
                "media patches require an image/video modality and non-negative ordinal"
            )
        if (
            not isinstance(self.pixel_values, torch.Tensor)
            or self.pixel_values.dtype is not torch.float32
            or self.pixel_values.layout is not torch.strided
            or not self.pixel_values.is_contiguous()
            or self.pixel_values.ndim != 2
            or min(self.pixel_values.shape) <= 0
            or self.pixel_values.device.type == "meta"
            or self.pixel_values.requires_grad
        ):
            raise ValueError("pixel_values must be a realized contiguous float32 patch matrix")
        if not bool(torch.isfinite(self.pixel_values).all().item()):
            raise ValueError("pixel_values must contain only finite values")
        if (
            type(self.grid_thw) is not tuple
            or len(self.grid_thw) != 3
            or any(type(axis) is not int or axis <= 0 for axis in self.grid_thw)
        ):
            raise ValueError("grid_thw must be a positive integer (t, h, w) tuple")
        if self.modality == "image":
            if self.grid_thw[0] != 1 or self.video_metadata is not None:
                raise ValueError("image patches require t=1 and no video metadata")
        elif self.video_metadata is None:
            raise ValueError("video patches require trusted sampling metadata")


@dataclass(frozen=True, slots=True)
class Qwen3_5TrustedMessage:
    """Role and trusted canonical content, preserving exact part order."""

    role: ChatRole
    content_parts: tuple[ContentPart, ...]
    template_metadata: ChatTemplateKwargs = field(default_factory=ChatTemplateKwargs)

    def __post_init__(self) -> None:
        if not isinstance(self.role, ChatRole) or not self.content_parts:
            raise ValueError("trusted message requires a supported role and content")
        if not isinstance(self.template_metadata, ChatTemplateKwargs):
            raise TypeError("trusted message metadata must be immutable")
        if any(not isinstance(part, ContentPart) for part in self.content_parts):
            raise TypeError("trusted message accepts only realized ContentPart values")
        if self.role is ChatRole.SYSTEM and any(
            part.modality != "text" for part in self.content_parts
        ):
            raise ValueError("system messages cannot contain image or video patches")


def _smart_resize(
    height: int,
    width: int,
    *,
    factor: int,
    min_pixels: int,
    max_pixels: int,
    frames: int = 1,
) -> tuple[int, int]:
    """Qwen2/3-VL factor-aligned resize rule used by the official processors."""

    if min(height, width, factor, frames) <= 0 or max(height, width) / min(height, width) > 200:
        raise Qwen3_5MultimodalError("media dimensions are outside the Qwen processor domain")
    height = max(height, factor)
    width = max(width, factor)
    resized_height = round(height / factor) * factor
    resized_width = round(width / factor) * factor
    pixels = frames * resized_height * resized_width
    if pixels > max_pixels:
        scale = math.sqrt((frames * height * width) / max_pixels)
        resized_height = max(factor, math.floor(height / scale / factor) * factor)
        resized_width = max(factor, math.floor(width / scale / factor) * factor)
    elif pixels < min_pixels:
        scale = math.sqrt(min_pixels / (frames * height * width))
        resized_height = math.ceil(height * scale / factor) * factor
        resized_width = math.ceil(width * scale / factor) * factor
    return resized_height, resized_width


class Qwen3_5MultimodalProcessor:
    """Resolve public image/video parts into the existing trusted Qwen bridge."""

    __slots__ = ("_bridge", "_image_pixels", "_resolver", "_text", "_video_pixels")

    def __init__(
        self,
        model_root: str | Path,
        *,
        vision: Qwen3_5VisionReferenceModel,
        chat_template_path: str | Path | None = None,
        allowed_local_media_path: str | Path | None = None,
        resource_policy: MultimodalInputPolicy | None = None,
        model_config: BuiltinModelConfig | None = None,
    ) -> None:
        root = Path(model_root).resolve(strict=True)
        self._text = Qwen3_5TextProcessor(root, chat_template_path=chat_template_path)
        self._bridge = Qwen3_5MultimodalBridge(
            root, vision=vision, chat_template_path=chat_template_path,
            resource_policy=resource_policy, model_config=model_config,
        )
        policy = resource_policy or MultimodalInputPolicy()
        self._resolver = MediaResolver(
            allowed_local_media_path=allowed_local_media_path,
            video_fps=policy.video_frames_per_second, max_frames=policy.video_max_frames)
        try:
            image = json.loads((root / "preprocessor_config.json").read_bytes())
            video = json.loads((root / "video_preprocessor_config.json").read_bytes())
            self._image_pixels = (
                int(image["size"]["shortest_edge"]),
                int(image["size"]["longest_edge"]),
            )
            self._video_pixels = (
                int(video["size"]["shortest_edge"]),
                int(video["size"]["longest_edge"]),
            )
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise Qwen3_5MultimodalError("Qwen media processor config is invalid") from error

    @property
    def constraint_special_token_ids(self) -> tuple[int, ...]:
        return self._text.constraint_special_token_ids

    def encoded_vocabulary(self, *, vocab_size: int) -> tuple[str | bytes, ...]:
        return self._text.encoded_vocabulary(vocab_size=vocab_size)

    def build_chat_input_plan(
        self,
        request: CanonicalChatRequest,
        *,
        weight_variant: WeightVariantMeta,
        add_generation_prompt: bool = True,
        enable_thinking: bool = True,
        reasoning_effort: ReasoningEffort = "xhigh",
        preserve_thinking: bool = True,
    ) -> InputPlan:
        prepared = self.build_chat_prepared_input(
            request,
            weight_variant=weight_variant,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
            reasoning_effort=reasoning_effort,
            preserve_thinking=preserve_thinking,
        )
        if prepared.feature_bundle is not None:
            raise Qwen3_5MultimodalError(
                "multimodal input requires the PreparedInput execution boundary"
            )
        return prepared.input_plan

    def build_chat_prepared_input(
        self,
        request: CanonicalChatRequest,
        *,
        weight_variant: WeightVariantMeta,
        add_generation_prompt: bool = True,
        enable_thinking: bool = True,
        reasoning_effort: ReasoningEffort = "xhigh",
        preserve_thinking: bool = True,
    ) -> PreparedInput:
        if not isinstance(request, CanonicalChatRequest):
            raise TypeError("Qwen multimodal processor requires a CanonicalChatRequest")
        if any(
            part.modality == "audio"
            for message in request.messages
            for part in message.content_parts
        ):
            raise Qwen3_5MultimodalError("Qwen3.8 does not support audio input")
        has_media = any(
            part.modality != "text"
            for message in request.messages
            for part in message.content_parts
        )
        if not has_media:
            return PreparedInput(
                self._text.build_chat_input_plan(
                    request,
                    weight_variant=weight_variant,
                    add_generation_prompt=add_generation_prompt,
                    enable_thinking=enable_thinking,
                    reasoning_effort=reasoning_effort,
                    preserve_thinking=preserve_thinking,
                )
            )

        realized = self._resolver.realize(request)
        realized_by_ordinal = {item.part.ordinal: item for item in realized}
        messages = tuple(
            Qwen3_5TrustedMessage(
                message.role,
                tuple(
                    ContentPart(part.ordinal, "text", text=part.text)
                    if part.modality == "text"
                    else realized_by_ordinal[part.ordinal].part
                    for part in message.content_parts
                ),
                ChatTemplateKwargs.from_value(message.template_metadata()),
            )
            for message in request.messages
        )
        template_kwargs, resolved_thinking = resolve_chat_template_kwargs(
            request,
            default_enable_thinking=enable_thinking,
        )
        return self._bridge.prepare(
            request_id=request.request_id,
            model_id=request.model_id,
            weight_variant=weight_variant,
            messages=messages,
            media_patches=tuple(self._patchify(item) for item in realized),
            add_generation_prompt=add_generation_prompt,
            enable_thinking=resolved_thinking,
            reasoning_effort=reasoning_effort,
            preserve_thinking=preserve_thinking,
            structured_output=request.structured_output,
            chat_template_kwargs=template_kwargs,
        )

    def _patchify(self, media: DecodedMedia) -> Qwen3_5MediaPatches:
        vision = self._bridge.vision.vision_config
        patch = vision.patch_size
        temporal = vision.temporal_patch_size
        merge = vision.spatial_merge_size
        frames = media.frames.permute(0, 3, 1, 2).to(torch.float32)
        frame_count, _, height, width = frames.shape
        minimum, maximum = (
            self._image_pixels if media.part.modality == "image" else self._video_pixels
        )
        resized_height, resized_width = _smart_resize(
            height,
            width,
            factor=patch * merge,
            min_pixels=minimum,
            max_pixels=maximum,
            frames=frame_count,
        )
        frames = (
            functional.interpolate(
                frames,
                size=(resized_height, resized_width),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
            .round_()
            .clamp_(0.0, 255.0)
            .div_(255.0)
            .sub_(0.5)
            .div_(0.5)
        )
        grid_h, grid_w = resized_height // patch, resized_width // patch
        if media.part.modality == "image":
            patches = (
                frames[:1]
                .reshape(1, 3, grid_h // merge, merge, patch, grid_w // merge, merge, patch)
                .permute(0, 2, 5, 3, 6, 1, 4, 7)
            )
            flattened = (
                patches.unsqueeze(6)
                .expand(-1, -1, -1, -1, -1, -1, temporal, -1, -1)
                .reshape(grid_h * grid_w, 3 * temporal * patch * patch)
            )
            return Qwen3_5MediaPatches(
                media.part.ordinal,
                "image",
                flattened.contiguous(),
                (1, grid_h, grid_w),
            )

        remainder = frame_count % temporal
        if remainder:
            frames = torch.cat((frames, frames[-1:].expand(temporal - remainder, -1, -1, -1)))
            frame_count = int(frames.shape[0])
        grid_t = frame_count // temporal
        flattened = (
            frames.reshape(
                1,
                grid_t,
                temporal,
                3,
                grid_h // merge,
                merge,
                patch,
                grid_w // merge,
                merge,
                patch,
            )
            .permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
            .reshape(grid_t * grid_h * grid_w, 3 * temporal * patch * patch)
        )
        return Qwen3_5MediaPatches(
            media.part.ordinal,
            "video",
            flattened.contiguous(),
            (grid_t, grid_h, grid_w),
            Qwen3_5VideoMetadata(media.frame_indices, media.frames_per_second,
                                media.frame_timestamps_seconds),
        )


class Qwen3_5MultimodalBridge:
    """Build an InputPlan and its exact eager EncodedFeatureBundle."""

    __slots__ = ("_config_digest", "_feature_cache", "_text", "_tokenizer", "_tokens", "vision")

    def __init__(
        self,
        model_root: str | Path,
        *,
        vision: Qwen3_5VisionReferenceModel,
        chat_template_path: str | Path | None = None,
        resource_policy: MultimodalInputPolicy | None = None,
        model_config: BuiltinModelConfig | None = None,
    ) -> None:
        root = Path(model_root).resolve(strict=True)
        try:
            config_document = (root / "config.json").read_bytes()
            config = (_parse_qwen3_5_family_config(config_document)
                      if model_config is None else model_config)
            tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
        except (OSError, RuntimeError, ValueError) as error:
            raise Qwen3_5MultimodalError("local multimodal processor files are invalid") from error
        if config != vision.config:
            raise Qwen3_5MultimodalError("vision program and processor config differ")
        self._text = Qwen3_5TextProcessor(root, chat_template_path=chat_template_path)
        self._tokens = getattr(config, "graph", config).modality_tokens
        self._tokenizer = tokenizer
        self._config_digest = f"b2s256:{hashlib.blake2s(config_document).hexdigest()}"
        self._feature_cache = EncodedFeatureCache(
            (resource_policy or MultimodalInputPolicy()).feature_cache_bytes)
        self.vision = vision

    def prepare(
        self,
        *,
        request_id: str,
        model_id: str,
        weight_variant: WeightVariantMeta,
        messages: tuple[Qwen3_5TrustedMessage, ...],
        media_patches: tuple[Qwen3_5MediaPatches, ...],
        add_generation_prompt: bool = True,
        enable_thinking: bool = True,
        reasoning_effort: ReasoningEffort = "xhigh",
        preserve_thinking: bool = True,
        structured_output: StructuredOutputMeta | None = None,
        chat_template_kwargs: Mapping[str, object] | None = None,
    ) -> PreparedInput:
        parts, ordered_media = self._validate_request(messages, media_patches)
        rendered_messages = tuple(
            {
                **message.template_metadata.as_dict(),
                "role": message.role.value,
                "content": "".join(
                    (part.text or "") if part.modality == "text" else _PLACEHOLDER[part.modality]
                    for part in message.content_parts
                ),
            }
            for message in messages
        )
        base_plan = self._text.build_input_plan(
            request_id=request_id,
            model_id=model_id,
            weight_variant=weight_variant,
            messages=rendered_messages,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
            reasoning_effort=reasoning_effort,
            preserve_thinking=preserve_thinking,
            structured_output=structured_output,
            chat_template_kwargs=chat_template_kwargs,
        )
        recipe_digest = self._recipe_digest(base_plan.processor_recipe_digest)
        token_ids, token_types, placements = self._expand_prompt(base_plan.token_ids, ordered_media)
        positions = self._mrope_positions(token_types, ordered_media)
        next_position = max(axis for position in positions for axis in position.axes) + 1
        jobs = tuple(
            EncoderJob(
                job_id=f"qwen3_5.vision.part.{media.content_part_ordinal}",
                content_part_ordinal=media.content_part_ordinal,
                encoder_id=f"qwen.vision@{self.vision.weights.semantic_plan_digest}",
                processor_id=recipe_digest,
                cache_key=self._cache_key(media, parts[media.content_part_ordinal], recipe_digest),
            )
            for media in ordered_media
        )
        input_plan = InputPlan(
            request_id=request_id,
            model_id=model_id,
            processor_recipe_digest=recipe_digest,
            weight_variant=weight_variant,
            content_parts=parts,
            token_ids=token_ids,
            positions=positions,
            decode_positions=DecodePositionPlan(
                first=TokenPosition((next_position, next_position, next_position)),
                step_axes=(1, 1, 1),
            ),
            placements=placements,
            encoder_jobs=jobs,
        )
        features = self._encode(ordered_media, jobs, parts)
        bundle = EncodedFeatureBundle.from_input_plan(input_plan, features)
        return PreparedInput(input_plan, bundle)

    def _validate_request(
        self,
        messages: tuple[Qwen3_5TrustedMessage, ...],
        media_patches: tuple[Qwen3_5MediaPatches, ...],
    ) -> tuple[tuple[ContentPart, ...], tuple[Qwen3_5MediaPatches, ...]]:
        if not messages or any(
            not isinstance(message, Qwen3_5TrustedMessage) for message in messages
        ):
            raise Qwen3_5MultimodalError("messages must be a non-empty trusted message tuple")
        parts = tuple(part for message in messages for part in message.content_parts)
        if tuple(part.ordinal for part in parts) != tuple(range(len(parts))):
            raise Qwen3_5MultimodalError("content part ordinals must be globally contiguous")
        non_text = tuple(part for part in parts if part.modality != "text")
        if any(part.modality not in _PLACEHOLDER for part in non_text):
            raise Qwen3_5MultimodalError("only image and video trusted media are supported")
        if tuple(item.content_part_ordinal for item in media_patches) != tuple(
            part.ordinal for part in non_text
        ):
            raise Qwen3_5MultimodalError(
                "trusted patch inputs must exactly follow media part order"
            )

        vision = self.vision.vision_config
        patch_width = (
            vision.input_channels
            * vision.temporal_patch_size
            * vision.patch_size
            * vision.patch_size
        )
        for part, patches in zip(non_text, media_patches, strict=True):
            if part.media is None or part.modality != patches.modality:
                raise Qwen3_5MultimodalError("trusted patch modality differs from its content part")
            if not part.media.mime_type.startswith(f"{part.modality}/"):
                raise Qwen3_5MultimodalError("trusted media MIME type differs from its modality")
            t, h, w = patches.grid_thw
            if (
                patches.pixel_values.shape != (t * h * w, patch_width)
                or h % vision.spatial_merge_size
                or w % vision.spatial_merge_size
            ):
                raise Qwen3_5MultimodalError("patch tensor and grid do not match the vision config")
            if patches.video_metadata is not None:
                timestamps = patches.video_metadata.timestamps(vision.temporal_patch_size)
                if len(timestamps) != t:
                    raise Qwen3_5MultimodalError(
                        "video temporal grid differs from sampling metadata"
                    )
                if part.media.frame_count is not None and any(
                    index >= part.media.frame_count
                    for index in patches.video_metadata.frame_indices
                ):
                    raise Qwen3_5MultimodalError(
                        "sampled video frame exceeds its trusted descriptor"
                    )
        return parts, media_patches

    def _expand_prompt(
        self,
        base_ids: tuple[int, ...],
        media_inputs: tuple[Qwen3_5MediaPatches, ...],
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[TokenPlacement, ...]]:
        output: list[int] = []
        token_types: list[int] = []
        placements: list[TokenPlacement] = []
        media_index = 0
        consumer_layers = (0,)
        merge = self.vision.vision_config.spatial_merge_size
        temporal_patch = self.vision.vision_config.temporal_patch_size

        for token_id in base_ids:
            if token_id not in {self._tokens.image, self._tokens.video}:
                output.append(token_id)
                token_types.append(0)
                continue
            if media_index >= len(media_inputs):
                raise Qwen3_5MultimodalError("rendered prompt contains an untrusted media token")
            media = media_inputs[media_index]
            expected = self._tokens.image if media.modality == "image" else self._tokens.video
            if token_id != expected:
                raise Qwen3_5MultimodalError(
                    "rendered media token order differs from trusted parts"
                )
            _, height, width = media.grid_thw
            frame_rows = (height // merge) * (width // merge)
            if media.modality == "image":
                start = len(output)
                output.extend([token_id] * frame_rows)
                token_types.extend([1] * frame_rows)
                placements.append(
                    TokenPlacement(
                        media.content_part_ordinal,
                        start,
                        frame_rows,
                        0,
                        frame_rows,
                        consumer_layers,
                    )
                )
            else:
                assert media.video_metadata is not None
                for frame, timestamp in enumerate(media.video_metadata.timestamps(temporal_patch)):
                    timestamp_ids = self._tokenizer.encode(
                        f"<{timestamp:.1f} seconds>", add_special_tokens=False
                    ).ids
                    output.extend(timestamp_ids)
                    token_types.extend([0] * len(timestamp_ids))
                    output.append(self._tokens.vision_start)
                    token_types.append(0)
                    start = len(output)
                    output.extend([token_id] * frame_rows)
                    token_types.extend([2] * frame_rows)
                    placements.append(
                        TokenPlacement(
                            media.content_part_ordinal,
                            start,
                            frame_rows,
                            frame * frame_rows,
                            frame_rows,
                            consumer_layers,
                        )
                    )
                    output.append(self._tokens.vision_end)
                    token_types.append(0)
            media_index += 1
        if media_index != len(media_inputs):
            raise Qwen3_5MultimodalError("chat template did not render every trusted media part")
        return tuple(output), tuple(token_types), tuple(placements)

    def _mrope_positions(
        self,
        token_types: tuple[int, ...],
        media_inputs: tuple[Qwen3_5MediaPatches, ...],
    ) -> tuple[TokenPosition, ...]:
        merge = self.vision.vision_config.spatial_merge_size
        grids: dict[int, list[tuple[int, int, int]]] = {1: [], 2: []}
        for media in media_inputs:
            t, h, w = media.grid_thw
            if media.modality == "image":
                grids[1].append((t, h, w))
            else:
                grids[2].extend((1, h, w) for _ in range(t))
        grid_offsets = {1: 0, 2: 0}
        positions: list[TokenPosition] = []
        current = 0
        for modality, grouped in groupby(token_types):
            count = sum(1 for _ in grouped)
            if modality == 0:
                positions.extend(TokenPosition((current + index,) * 3) for index in range(count))
                current += count
                continue
            try:
                grid = grids[modality][grid_offsets[modality]]
            except (KeyError, IndexError) as error:
                raise Qwen3_5MultimodalError("media token groups exceed trusted grids") from error
            grid_offsets[modality] += 1
            t, h, w = grid
            llm_h, llm_w = h // merge, w // merge
            if count != t * llm_h * llm_w:
                raise Qwen3_5MultimodalError("media token group length differs from its grid")
            positions.extend(
                TokenPosition((current + temporal, current + height, current + width))
                for temporal in range(t)
                for height in range(llm_h)
                for width in range(llm_w)
            )
            current += max(h, w) // merge
        if any(grid_offsets[key] != len(grids[key]) for key in (1, 2)):
            raise Qwen3_5MultimodalError("trusted grids exceed rendered media token groups")
        return tuple(positions)

    def _encode(
        self,
        media_inputs: tuple[Qwen3_5MediaPatches, ...],
        jobs: tuple[EncoderJob, ...],
        parts: tuple[ContentPart, ...],
    ) -> tuple[EncodedFeature, ...]:
        features: list[EncodedFeature] = []
        device = self.vision.weights.device
        for media, job in zip(media_inputs, jobs, strict=True):
            descriptor = parts[media.content_part_ordinal].media
            assert descriptor is not None
            cache_key = f"{job.job_id}\0{job.cache_key}"
            cached = self._feature_cache.get(cache_key)
            if cached is not None:
                if (
                    cached.content_part_ordinal != job.content_part_ordinal
                    or cached.encoder_id != job.encoder_id
                    or cached.processor_id != job.processor_id
                    or cached.source_media_digest != descriptor.content_digest
                ):
                    raise Qwen3_5MultimodalError("cached vision feature identity differs")
                features.append(cached)
                continue
            # Captured encoders own upload through immutable host-result completion.
            # Eager/reference encoders retain their existing device-input contract.
            input_device = getattr(self.vision, "input_device", device)
            pixels = media.pixel_values.to(device=input_device)
            grid = torch.tensor((media.grid_thw,), dtype=torch.int64,
                                device=input_device)
            tensor = self.vision.forward(pixels, grid).to("cpu").contiguous()
            features.append(
                self._feature_cache.put(
                    cache_key,
                    EncodedFeature(
                        job_id=job.job_id,
                        content_part_ordinal=media.content_part_ordinal,
                        encoder_id=job.encoder_id,
                        processor_id=job.processor_id,
                        source_media_digest=descriptor.content_digest,
                        tensor=tensor,
                    ),
                )
            )
        return tuple(features)

    def _recipe_digest(self, text_recipe_digest: str) -> str:
        return _digest(
            {
                "config": self._config_digest,
                "feature_fusion": "decoder.input@layer:0;identity_rows",
                "kind": _RECIPE_KIND,
                "placeholder_ids": {
                    "image": self._tokens.image,
                    "video": self._tokens.video,
                    "vision_end": self._tokens.vision_end,
                    "vision_start": self._tokens.vision_start,
                },
                "position_recipe": "transformers.qwen3_5.get_rope_index:three-axis",
                "text_recipe": text_recipe_digest,
                "trusted_input": "float32_patch_rows+grid_thw+sampled_frame_indices+fps",
                "video_layout": "timestamp+vision_start+frame_rows+vision_end",
            }
        )

    def _cache_key(
        self,
        media: Qwen3_5MediaPatches,
        part: ContentPart,
        recipe_digest: str,
    ) -> str:
        assert part.media is not None
        timestamps = (
            ()
            if media.video_metadata is None
            else tuple(
                f"{value:.17g}"
                for value in media.video_metadata.timestamps(
                    self.vision.vision_config.temporal_patch_size
                )
            )
        )
        return _digest(
            {
                "grid_thw": media.grid_thw,
                "media": part.media.content_digest,
                "modality": media.modality,
                "patches": _tensor_digest(media.pixel_values),
                "processor": recipe_digest,
                "timestamps": timestamps,
                "vision": self.vision.weights.semantic_plan_digest,
            }
        )


def _digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return f"b2s256:{hashlib.blake2s(payload).hexdigest()}"


def _tensor_digest(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy()
    return _digest(
        {"bytes": hashlib.blake2s(memoryview(raw)).hexdigest(), "shape": tuple(tensor.shape)}
    )


__all__ = (
    "Qwen3_5MediaPatches",
    "Qwen3_5MultimodalBridge",
    "Qwen3_5MultimodalError",
    "Qwen3_5MultimodalProcessor",
    "Qwen3_5TrustedMessage",
    "Qwen3_5VideoMetadata",
    "Qwen3_5VisionLoadError",
    "Qwen3_5VisionReferenceModel",
    "Qwen3_5VisionWeights",
    "load_qwen3_5_vision_reference",
)


# Existing names remain import-compatible; model families share these contracts.
QwenVisionReferenceModel = Qwen3_5VisionReferenceModel
QwenMultimodalProcessor = Qwen3_5MultimodalProcessor
load_qwen_vision_reference = load_qwen3_5_vision_reference
