"""Deterministic path-loaded 3-in/1-out model used by the first vertical slice."""

from __future__ import annotations

import torch

from cubic_serve.contracts.execution import (
    BatchPhase,
    DecodeExtensionMeta,
    OpSignature,
    TensorSpec,
)
from cubic_serve.contracts.model import (
    CacheSpec,
    ModalitySpec,
    ModelSpec,
    PrefixReplayPolicy,
    ScalarDType,
    StateSpec,
    WeightVariantMeta,
)
from cubic_serve.executor.runner import require_reference_execution
from cubic_serve.inputs import FeatureReader
from cubic_serve.models import (
    ModelAdapterDescriptor,
    ModelBuildContext,
    ModelSegment,
    ModelStateIO,
    ModelStepResult,
    ProgramExecutionRecipe,
    SemanticOpExecutor,
    StateInitialization,
)
from cubic_serve.models.weights import FullBF16WeightPolicy, collect_full_bf16_weight_evidence
from cubic_serve.ops.catalog import ReferenceImplementation

CUBIC_MODEL_ADAPTER_API_VERSION = 1

_HIDDEN = 8
_VOCAB = 32
_MAX_PREFILL_M = 4


def _fixture_encode(value: torch.Tensor, offset: int) -> torch.Tensor:
    require_reference_execution()
    if value.dtype is not torch.bfloat16:
        raise TypeError("toy fixture encoders require BF16 input")
    return (value.to(torch.float32) + offset).to(torch.bfloat16)


def encode_text(value: torch.Tensor) -> torch.Tensor:
    return _fixture_encode(value, 1)


def encode_media(value: torch.Tensor) -> torch.Tensor:
    return _fixture_encode(value, 2)


def _signature(
    semantic_op: str,
    phase: BatchPhase,
    m: int,
    *,
    layout: str,
    cache_form: str | None = None,
) -> OpSignature:
    shape_items: tuple[tuple[str, int], ...] = (("hidden", _HIDDEN), ("m", m))
    inputs: tuple[TensorSpec, ...]
    outputs: tuple[TensorSpec, ...]
    if semantic_op.startswith("fixture.input."):
        inputs = (TensorSpec("value", ScalarDType.BF16, ("hidden",)),)
        outputs = (TensorSpec("encoded", ScalarDType.BF16, ("hidden",)),)
    elif semantic_op == "linear" and layout == "hidden_hidden":
        inputs = (
            TensorSpec("inputs", ScalarDType.BF16, ("m", "hidden")),
            TensorSpec("weight", ScalarDType.BF16, ("hidden", "hidden")),
        )
        outputs = (TensorSpec("result", ScalarDType.BF16, ("m", "hidden")),)
    elif semantic_op == "linear" and layout == "hidden_vocab":
        shape_items = (("hidden", _HIDDEN), ("m", m), ("vocab", _VOCAB))
        inputs = (
            TensorSpec("inputs", ScalarDType.BF16, ("m", "hidden")),
            TensorSpec("weight", ScalarDType.BF16, ("vocab", "hidden")),
        )
        outputs = (TensorSpec("result", ScalarDType.BF16, ("m", "vocab")),)
    elif semantic_op == "attention.full":
        inputs = (
            TensorSpec("query", ScalarDType.BF16, (1, 1, "m", "hidden")),
            TensorSpec("key", ScalarDType.BF16, (1, 1, "context", "hidden")),
            TensorSpec("value", ScalarDType.BF16, (1, 1, "context", "hidden")),
            TensorSpec("query_ordinals", ScalarDType.INT64, (1, "m")),
            TensorSpec("key_ordinals", ScalarDType.INT64, (1, "context")),
        )
        outputs = (TensorSpec("attended", ScalarDType.BF16, (1, 1, "m", "hidden")),)
    else:
        raise ValueError(f"toy signature has no tensor ABI for {semantic_op!r}/{layout!r}")
    return OpSignature(
        semantic_op=semantic_op,
        phase=phase,
        inputs=inputs,
        outputs=outputs,
        accumulator_dtype=ScalarDType.FP32,
        layout=layout,
        shape_items=shape_items,
        cache_form=cache_form,
        state_form=None,
        topology_id="single",
    )


def _required_signatures() -> tuple[OpSignature, ...]:
    signatures: list[OpSignature] = []
    for modality in ("text", "media"):
        signatures.append(
            _signature(
                f"fixture.input.{modality}",
                BatchPhase.PREFILL,
                1,
                layout="token_hidden",
            )
        )
    signatures.append(
        _signature(
            "fixture.input.text",
            BatchPhase.DECODE,
            1,
            layout="token_hidden",
        )
    )
    for m in range(1, _MAX_PREFILL_M + 1):
        signatures.extend(
            (
                _signature("linear", BatchPhase.PREFILL, m, layout="hidden_hidden"),
                _signature(
                    "attention.full",
                    BatchPhase.PREFILL,
                    m,
                    layout="bhsd",
                    cache_form="paged_bf16",
                ),
            )
        )
    signatures.extend(
        (
            _signature("linear", BatchPhase.DECODE, 1, layout="hidden_hidden"),
            _signature(
                "attention.full",
                BatchPhase.DECODE,
                1,
                layout="bhsd",
                cache_form="paged_bf16",
            ),
            _signature("linear", BatchPhase.PREFILL, 1, layout="hidden_vocab"),
            _signature("linear", BatchPhase.DECODE, 1, layout="hidden_vocab"),
        )
    )
    return tuple(signatures)


def _descriptor() -> ModelAdapterDescriptor:
    program = ToyProgram(ModelBuildContext(
        device=torch.device("cpu"), rank_id=0, world_size=1, owned_layer_ids=(0,),
    ))
    tensors = program.named_weight_tensors()
    evidence = collect_full_bf16_weight_evidence(
        tensors, policy=FullBF16WeightPolicy(expected_names=tuple(name for name, _ in tensors)),
    )
    model = ModelSpec(
        model_id="quanttrio/toy-3i1o",
        architecture="toy_dense_multimodal",
        revision="fixture-v4",
        weights=evidence.manifest,
        cache=CacheSpec(
            kv_dtype=ScalarDType.BF16,
            page_size_tokens=4,
            layout="layer_token_kv_head_dim",
            prefix_enabled=True,
        ),
        modalities=(
            ModalitySpec(name="text"),
            ModalitySpec(
                name="image",
                mime_prefixes=("image/",),
                encoder_id="toy.vision",
                feature_hidden_size=8,
            ),
            ModalitySpec(
                name="video",
                mime_prefixes=("video/",),
                encoder_id="toy.vision",
                feature_hidden_size=8,
            ),
        ),
        states=(
            StateSpec(
                state_id="toy.running_hidden",
                dtype=ScalarDType.FP32,
                layout="hidden",
                shape_symbols=("hidden",),
                commit_alignment_tokens=3,
                prefix_replay=PrefixReplayPolicy.FROM_LAST_ALIGNED_COMMIT,
            ),
        ),
        layer_count=1,
    )
    references = (
        ReferenceImplementation("fixture.input.text", "toy.reference.text.v1", encode_text),
        ReferenceImplementation("fixture.input.media", "toy.reference.media.v1", encode_media),
    )
    return ModelAdapterDescriptor(
        adapter_id="quanttrio.toy-3i1o",
        adapter_revision="4",
        model_spec=model,
        weight_variant=WeightVariantMeta(
            base_model_id=model.model_id,
            base_revision=model.revision,
        ),
        decode_extension=DecodeExtensionMeta(),
        required_ops=_required_signatures(),
        reference_implementations=references,
    )


class ToyProgram:
    def __init__(self, context: ModelBuildContext) -> None:
        if context.rank_id != 0 or context.world_size != 1:
            raise ValueError("the toy program is intentionally single-rank")
        if context.owned_layer_ids != (0,):
            raise ValueError("the toy program owns exactly decoder layer 0")
        self.device = context.device
        values = torch.arange(_VOCAB * _HIDDEN, device=self.device, dtype=torch.float32)
        self.embedding = (((values % 17) - 8) / 16).reshape(_VOCAB, _HIDDEN).to(torch.bfloat16)
        identity = torch.eye(_HIDDEN, device=self.device, dtype=torch.bfloat16)
        self.q_weight = identity
        self.k_weight = torch.roll(identity, shifts=1, dims=0)
        self.v_weight = torch.flip(identity, dims=(0,))
        self.out_weight = torch.roll(identity, shifts=2, dims=0)
        lm_values = torch.arange(_VOCAB * _HIDDEN, device=self.device, dtype=torch.float32)
        self.lm_weight = (((lm_values % 11) - 5) / 8).reshape(_VOCAB, _HIDDEN).to(torch.bfloat16)

    def named_weight_tensors(self) -> tuple[tuple[str, torch.Tensor], ...]:
        return (
            ("embedding", self.embedding),
            ("k_weight", self.k_weight),
            ("lm_weight", self.lm_weight),
            ("out_weight", self.out_weight),
            ("q_weight", self.q_weight),
            ("v_weight", self.v_weight),
        )

    def execution_recipe(self) -> ProgramExecutionRecipe:
        return ProgramExecutionRecipe(
            recipe_id="cubic.toy-dense-multimodal.reference",
            revision="2",
            settings=(
                ("attention", "full"),
                ("hidden_size", str(_HIDDEN)),
                ("max_prefill_m", str(_MAX_PREFILL_M)),
                ("vocab_size", str(_VOCAB)),
            ),
        )

    def initialize_request(self) -> tuple[StateInitialization, ...]:
        return (
            StateInitialization(
                state_id="toy.running_hidden",
                value=torch.zeros((_HIDDEN,), dtype=torch.float32, device=self.device),
            ),
        )

    @staticmethod
    def _position_row(axes: tuple[int, ...], device: torch.device) -> torch.Tensor:
        row = torch.zeros((_HIDDEN,), dtype=torch.float32, device=device)
        for axis_index, axis in enumerate(axes):
            row[axis_index % _HIDDEN] += float((axis_index + 1) * axis)
        return row

    def execute_segment(
        self,
        segment: ModelSegment,
        ops: SemanticOpExecutor,
        state: ModelStateIO,
        features: FeatureReader,
    ) -> ModelStepResult:
        require_reference_execution()
        phase = segment.phase
        m = segment.token_count
        if phase is BatchPhase.PREFILL and m > _MAX_PREFILL_M:
            raise ValueError("toy prefill M exceeds its sealed signature domain")

        media_rows: dict[int, torch.Tensor] = {}
        for span in features.spans("decoder.input", 0):
            if span.rows.shape[1] != _HIDDEN:
                raise ValueError("toy encoded feature hidden size must equal 8")
            for relative_index, feature_row in enumerate(span.rows):
                token_offset = span.token_offset + relative_index
                if token_offset in media_rows:
                    raise ValueError("toy feature spans cannot overlap")
                media_rows[token_offset] = feature_row

        rows: list[torch.Tensor] = []
        for token_offset, (token_id, position) in enumerate(zip(
            segment.token_ids,
            segment.positions,
            strict=True,
        )):
            if token_id >= _VOCAB:
                raise ValueError("toy token id exceeds its vocabulary")
            row = self.embedding[token_id].to(torch.float32)
            row = row + self._position_row(position.axes, self.device)
            feature_row = media_rows.get(token_offset)
            modality = "media" if feature_row is not None else "text"
            if feature_row is not None:
                row = row + feature_row.to(torch.float32)
            input_signature = _signature(
                f"fixture.input.{modality}", phase, 1, layout="token_hidden"
            )
            rows.append(ops.call(input_signature, row.to(torch.bfloat16)))
        hidden = torch.stack(rows)
        running = state.read_recurrent_state("toy.running_hidden")
        stateful_rows: list[torch.Tensor] = []
        for row in hidden:
            running = running + row.to(torch.float32)
            stateful_rows.append((row.to(torch.float32) + running).to(torch.bfloat16))
        hidden = torch.stack(stateful_rows)
        state.write_recurrent_state(
            "toy.running_hidden",
            running,
            target_position=segment.token_start + m,
        )

        linear_signature = _signature("linear", phase, m, layout="hidden_hidden")
        query = ops.call(linear_signature, hidden, self.q_weight)
        key = ops.call(linear_signature, hidden, self.k_weight)
        value = ops.call(linear_signature, hidden, self.v_weight)
        ordinals = tuple(range(segment.token_start, segment.token_start + m))
        state.write_kv(
            0,
            ordinals,
            key.reshape(m, 1, _HIDDEN),
            value.reshape(m, 1, _HIDDEN),
        )

        context_ordinals = tuple(range(segment.token_start + m))
        cached_key, cached_value = state.read_kv(0, context_ordinals)
        query_tensor = query.reshape(m, 1, _HIDDEN).transpose(0, 1).unsqueeze(0)
        key_tensor = cached_key.transpose(0, 1).unsqueeze(0)
        value_tensor = cached_value.transpose(0, 1).unsqueeze(0)
        attention_signature = _signature(
            "attention.full",
            phase,
            m,
            layout="bhsd",
            cache_form="paged_bf16",
        )
        attended = ops.call(
            attention_signature,
            query_tensor,
            key_tensor,
            value_tensor,
            query_ordinals=torch.tensor(ordinals, dtype=torch.int64, device=self.device).unsqueeze(
                0
            ),
            key_ordinals=torch.tensor(
                context_ordinals, dtype=torch.int64, device=self.device
            ).unsqueeze(0),
            output_dtype=torch.bfloat16,
        )[0, 0]
        transformed = ops.call(linear_signature, attended, self.out_weight)
        hidden = (hidden.to(torch.float32) + transformed.to(torch.float32)).to(torch.bfloat16)

        logits_signature = _signature("linear", phase, 1, layout="hidden_vocab")
        logits = ops.call(logits_signature, hidden[-1:], self.lm_weight)[0]
        return ModelStepResult(logits=logits)


class ToyAdapter:
    def __init__(self) -> None:
        self._descriptor = _descriptor()

    def describe(self) -> ModelAdapterDescriptor:
        return self._descriptor

    def build(self, context: ModelBuildContext) -> ToyProgram:
        return ToyProgram(context)


def create_model_adapter() -> ToyAdapter:
    return ToyAdapter()
