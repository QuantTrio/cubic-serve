"""Narrow ABI between Cubic Serve and a path-loaded model adapter."""

from __future__ import annotations

from typing_extensions import Generic, TypeVar

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Protocol, runtime_checkable

from cubic_serve.contracts.execution import DecodeExtensionMeta, OpSignature
from cubic_serve.contracts.model import ModelSpec, WeightVariantMeta
from cubic_serve.models.protocol import ModelBuildContext, ModelProgram, ProgramExecutionRecipe
from cubic_serve.ops.catalog import ReferenceImplementation

if TYPE_CHECKING:
    from cubic_serve.contracts.execution import ParallelTopology
    from cubic_serve.executor.draft import DraftProvider
    from cubic_serve.generation.output import ChatOutputCodec
    from cubic_serve.inputs.chat import ChatInputProcessor
    from cubic_serve.models.parallel_weights import LogicalWeightSlice
    from cubic_serve.models.quantized_checkpoint import LogicalWeight

MODEL_ADAPTER_API_VERSION: Final[int] = 1
MODEL_DEFINITION_API_VERSION: Final[int] = 1


Config = TypeVar("Config")


class WeightOwnershipPlanner(Protocol[Config]):
    def __call__(
        self,
        config: Config,
        topology: ParallelTopology,
        rank: int,
        *,
        include_mtp: bool = False,
        logical_weights: tuple[LogicalWeight, ...] | None = None,
    ) -> tuple[LogicalWeightSlice, ...]: ...


class DraftProviderFactory(Protocol[Config]):
    def __call__(self, config: Config, *, method: str, max_depth: int) -> DraftProvider: ...


@dataclass(frozen=True, slots=True)
class TextGraphDefinition:
    """Startup binding to an existing stage and its model-owned configuration.

    The engine retains allocation, capture, qualification, leases and replay.
    ``named_state_fields`` selects the existing named-state execution contract.
    """

    stage_type: Callable[..., Any]
    named_state_fields: bool
    stage_options: Mapping[str, Any]
    required_operations: tuple[str, ...]
    block_size_tokens: int
    recurrent_state_dtype: str
    state_audit_steps: int = 0

    def __post_init__(self) -> None:
        if (type(self.state_audit_steps) is not int or self.state_audit_steps < 0
                or (self.named_state_fields and self.state_audit_steps == 0)):
            raise ValueError("named-state model graphs require a positive audit horizon")


class TextGraphFactory(Protocol[Config]):
    def __call__(
        self, config: Config, topology: ParallelTopology, *, block_size: int,
        recurrent_state_dtype: str,
        definition: ModelDefinition[Config] | None = None,
    ) -> TextGraphDefinition: ...


@dataclass(frozen=True, slots=True)
class ModelDefinition(Generic[Config]):
    """Reusable startup declarations, without runtime or kernel authority.

    Keep original parser/planner functions and native config objects. A derivative
    may use dataclasses.replace to override only its changed semantics. Checkpoint
    I/O, compute policy, execution and qualification remain common engine work.
    """

    model_type: str
    parse_config: Callable[[Mapping[str, object]], Config]
    logical_weights: Callable[[Config], tuple[LogicalWeight, ...]]
    owner_slices: WeightOwnershipPlanner[Config]
    api_version: int = MODEL_DEFINITION_API_VERSION
    text_input_processor: Callable[..., ChatInputProcessor] | None = None
    multimodal_input_processor: Callable[..., ChatInputProcessor] | None = None
    output_codec: Callable[..., ChatOutputCodec] | None = None
    draft_provider: DraftProviderFactory[Config] | None = None
    text_model: Callable[..., Any] | None = None
    text_graph: TextGraphFactory[Config] | None = None
    compute_config: Callable[[Mapping[str, object], str], Mapping[str, object]] | None = None
    program_recipe: Callable[[Config], ProgramExecutionRecipe] | None = None
    reference_admission: Callable[..., Any] | None = None
    input_encoder: Callable[..., Any] | None = None
    reference_input_encoder: Callable[..., Any] | None = None
    # Load-time declarations only. Container readers validate and materialize
    # them into the existing logical plan; model execution never sees a format.
    checkpoint_imports: tuple[tuple[str, Callable[..., Any]], ...] = ()

    def __post_init__(self) -> None:
        if type(self.api_version) is not int or self.api_version != MODEL_DEFINITION_API_VERSION:
            raise ValueError("unsupported model definition API version")
        if not isinstance(self.model_type, str) or not self.model_type.strip():
            raise ValueError("model definition requires a model_type")
        if self.model_type != self.model_type.strip():
            raise ValueError("model definition model_type must be canonical")
        if not all(
            callable(item)
            for item in (self.parse_config, self.logical_weights, self.owner_slices)
        ):
            raise TypeError("model definition requires callable parser and weight planners")

        if any(
            item is not None and not callable(item)
            for item in (
                self.text_input_processor, self.multimodal_input_processor, self.output_codec
            )
        ):
            raise TypeError("model frontend constructors must be callable when supplied")

        if self.draft_provider is not None and not callable(self.draft_provider):
            raise TypeError("model draft provider must be callable when supplied")

        if self.text_model is not None and not callable(self.text_model):
            raise TypeError("model text constructor must be callable when supplied")

        if self.text_graph is not None and not callable(self.text_graph):
            raise TypeError("model text graph binding must be callable when supplied")

        if self.compute_config is not None and not callable(self.compute_config):
            raise TypeError("model compute config transform must be callable when supplied")

        if self.program_recipe is not None and not callable(self.program_recipe):
            raise TypeError("model program recipe must be callable when supplied")

        if any(item is not None and not callable(item) for item in (
            self.reference_admission, self.input_encoder, self.reference_input_encoder,
        )):
            raise TypeError("model admission and input encoder bindings must be callable")
        if (type(self.checkpoint_imports) is not tuple
                or any(type(item) is not tuple or len(item) != 2 or not item[0]
                       or not isinstance(item[0], str) or not callable(item[1])
                       for item in self.checkpoint_imports)
                or len({item[0] for item in self.checkpoint_imports})
                != len(self.checkpoint_imports)):
            raise TypeError("checkpoint imports require unique container names and declarations")

    def frontend_factories(
        self, *, multimodal: bool
    ) -> tuple[Callable[..., ChatInputProcessor], Callable[..., ChatOutputCodec]]:
        processor = self.multimodal_input_processor if multimodal else self.text_input_processor
        if processor is None or self.output_codec is None:
            raise ValueError("model has no declared chat frontend construction binding")
        return processor, self.output_codec


@dataclass(frozen=True, slots=True)
class ModelAdapterDescriptor:
    """Complete, immutable declaration needed to admit one model instance."""

    adapter_id: str
    adapter_revision: str
    model_spec: ModelSpec
    weight_variant: WeightVariantMeta
    decode_extension: DecodeExtensionMeta
    required_ops: tuple[OpSignature, ...]
    reference_implementations: tuple[ReferenceImplementation, ...] = ()

    def __post_init__(self) -> None:
        if not self.adapter_id or not self.adapter_revision:
            raise ValueError("adapter identity and revision are required")
        if not isinstance(self.model_spec, ModelSpec):
            raise TypeError("model_spec must be a ModelSpec")
        if not isinstance(self.weight_variant, WeightVariantMeta):
            raise TypeError("weight_variant must be a WeightVariantMeta")
        if not isinstance(self.decode_extension, DecodeExtensionMeta):
            raise TypeError("decode_extension must be a DecodeExtensionMeta")
        if not isinstance(self.required_ops, tuple) or not self.required_ops:
            raise ValueError("required_ops must be a non-empty tuple")
        if any(not isinstance(item, OpSignature) for item in self.required_ops):
            raise TypeError("required_ops must contain only OpSignature values")
        if not isinstance(self.reference_implementations, tuple):
            raise TypeError("reference_implementations must be a tuple")
        if any(
            not isinstance(item, ReferenceImplementation) for item in self.reference_implementations
        ):
            raise TypeError(
                "reference_implementations must contain only ReferenceImplementation values"
            )


@runtime_checkable
class ModelAdapter(Protocol):
    """The only behavior required from an external model adapter instance."""

    def describe(self) -> ModelAdapterDescriptor:
        """Return this instance's immutable model and execution declaration."""

        ...

    def build(self, context: ModelBuildContext) -> ModelProgram:
        """Construct one local-rank model program without selecting kernels."""

        ...
