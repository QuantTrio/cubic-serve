"""Reference execution policy and runner primitives.

Reference operators are deliberately unusable outside :class:`ReferenceExecutionContext`.
That makes ambient autocast and graph capture fail closed instead of silently changing the
declared arithmetic.
"""

from __future__ import annotations

import inspect
import os
from contextlib import ExitStack
from contextvars import ContextVar, Token
from dataclasses import dataclass
from types import TracebackType
from typing import Any, cast

import torch

from cubic_serve.contracts.execution import (
    ExecutionMode,
    ExecutionPlan,
    OpSignature,
    PlanSegment,
    TensorSpec,
)
from cubic_serve.contracts.model import ScalarDType
from cubic_serve.ops.catalog import ReferenceCatalog, resolve_reference_binding


@dataclass(frozen=True, slots=True)
class ReferenceMathPolicy:
    cublas_workspace_config: str
    deterministic_algorithms: bool
    fp32_precision: str
    tf32: bool
    reduced_precision_reduction: bool
    fp16_accumulation: bool
    cudnn_benchmark: bool
    autocast: bool
    compiled: bool
    custom_candidates: bool
    cuda_graph: bool
    autotune: bool
    async_overlap: bool


@dataclass(frozen=True, slots=True)
class ReferenceSegmentIdentity:
    """Exact immutable BatchPlan identity behind one revocable capability set."""

    plan_id: str
    topology_id: str
    partition_id: str
    segment: PlanSegment

    def __post_init__(self) -> None:
        if not self.plan_id or not self.topology_id or not self.partition_id:
            raise ValueError("reference segment plan/topology/partition identity is required")
        if not isinstance(self.segment, PlanSegment):
            raise TypeError("reference segment identity requires a PlanSegment")


_REFERENCE_DEPTH: ContextVar[int] = ContextVar("cubic_reference_depth", default=0)


def _is_compiling() -> bool:
    compiler = getattr(torch, "compiler", None)
    predicate = getattr(compiler, "is_compiling", None)
    return bool(predicate is not None and predicate())


def _active_autocast_devices() -> tuple[str, ...]:
    return tuple(
        device_type for device_type in ("cpu", "cuda") if torch.is_autocast_enabled(device_type)
    )


def _cuda_graph_capture_active() -> bool:
    if not torch.cuda.is_initialized():  # type: ignore[no-untyped-call]
        return False
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except RuntimeError as error:
        raise RuntimeError("reference mode could not verify CUDA graph capture state") from error


def _assert_reference_math_policy() -> None:
    violations: list[str] = []
    if _is_compiling():
        violations.append("graph compilation is active")
    if _cuda_graph_capture_active():
        violations.append("CUDA graph capture is active")
    active_devices = _active_autocast_devices()
    if active_devices:
        violations.append("autocast is active for " + ", ".join(active_devices))
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        violations.append("CUBLAS_WORKSPACE_CONFIG is not :4096:8")
    if not torch.are_deterministic_algorithms_enabled():
        violations.append("deterministic algorithms are disabled")
    if torch.is_deterministic_algorithms_warn_only_enabled():
        violations.append("deterministic algorithms are warn-only")
    backends = cast(Any, torch.backends)
    precision_settings = (
        backends.fp32_precision,
        torch.backends.cuda.matmul.fp32_precision,
        torch.backends.cudnn.fp32_precision,
    )
    if precision_settings != ("ieee", "ieee", "ieee"):
        violations.append("FP32 backend precision is not uniformly IEEE")
    if torch.get_float32_matmul_precision() != "highest":
        violations.append("float32 matmul precision is not highest")
    if (
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        or torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
    ):
        violations.append("reduced-precision accumulation is enabled")
    if torch.backends.cuda.matmul.allow_fp16_accumulation:
        violations.append("FP16 accumulation is enabled")
    if torch.backends.cudnn.benchmark:
        violations.append("cuDNN benchmark selection is enabled")
    if not torch.backends.cudnn.deterministic:
        violations.append("cuDNN deterministic mode is disabled")
    if violations:
        raise RuntimeError("reference execution policy drift: " + "; ".join(violations))


def require_reference_execution() -> None:
    """Reject a reference op call whose process/thread policy is not sealed."""

    if _REFERENCE_DEPTH.get() <= 0:
        raise RuntimeError("reference operators require ReferenceExecutionContext")
    _assert_reference_math_policy()


def configure_reference_math() -> ReferenceMathPolicy:
    """Seal the process-wide eager math policy before the first CUDA allocation."""

    if _is_compiling():
        raise RuntimeError("reference math cannot be configured under graph compilation")

    required_workspace = ":4096:8"
    configured_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if configured_workspace not in (None, required_workspace):
        raise RuntimeError(
            "reference mode requires CUBLAS_WORKSPACE_CONFIG=:4096:8; "
            f"found {configured_workspace!r}"
        )
    if configured_workspace is None:
        if torch.cuda.is_initialized():  # type: ignore[no-untyped-call]
            raise RuntimeError("reference math must be configured before the first CUDA allocation")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = required_workspace

    torch.use_deterministic_algorithms(True, warn_only=False)
    # Torch 2.13's hierarchical precision API avoids the ambiguous legacy
    # cuDNN allow_tf32 flag (conv and RNN can otherwise disagree).
    backends = cast(Any, torch.backends)
    backends.fp32_precision = "ieee"
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    return ReferenceMathPolicy(
        cublas_workspace_config=required_workspace,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        fp32_precision="ieee",
        tf32=False,
        reduced_precision_reduction=False,
        fp16_accumulation=False,
        cudnn_benchmark=False,
        autocast=False,
        compiled=False,
        custom_candidates=False,
        cuda_graph=False,
        autotune=False,
        async_overlap=False,
    )


class ReferenceExecutionContext:
    """A fail-closed eager execution domain for trusted reference semantics."""

    def __init__(self) -> None:
        self.policy: ReferenceMathPolicy | None = None
        self._stack: ExitStack | None = None
        self._depth_token: Token[int] | None = None

    def __enter__(self) -> ReferenceMathPolicy:
        if self._stack is not None:
            raise RuntimeError("ReferenceExecutionContext instances cannot be re-entered")
        if _is_compiling():
            raise RuntimeError("reference execution cannot enter under graph compilation")
        active_devices = _active_autocast_devices()
        if active_devices:
            raise RuntimeError(
                "reference execution refuses ambient autocast for " + ", ".join(active_devices)
            )

        self.policy = configure_reference_math()
        stack = ExitStack()
        try:
            stack.enter_context(torch.autocast(device_type="cpu", enabled=False))
            stack.enter_context(torch.autocast(device_type="cuda", enabled=False))
            self._depth_token = _REFERENCE_DEPTH.set(_REFERENCE_DEPTH.get() + 1)
            _assert_reference_math_policy()
        except BaseException:
            if self._depth_token is not None:
                _REFERENCE_DEPTH.reset(self._depth_token)
                self._depth_token = None
            stack.close()
            raise
        self._stack = stack
        return self.policy

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._stack is None or self._depth_token is None:
            raise RuntimeError("ReferenceExecutionContext was not entered")
        policy_error: RuntimeError | None = None
        try:
            _assert_reference_math_policy()
        except RuntimeError as error:
            policy_error = error
        finally:
            _REFERENCE_DEPTH.reset(self._depth_token)
            self._stack.close()
            self._stack = None
            self._depth_token = None
        if policy_error is not None:
            configure_reference_math()
            if exception is not None:
                exception.add_note(str(policy_error))
            else:
                raise policy_error


class ReferenceOpExecutor:
    """Invoke only exact signatures sealed into one reference ExecutionPlan."""

    def __init__(self, plan: ExecutionPlan, catalog: ReferenceCatalog) -> None:
        if plan.mode is not ExecutionMode.REFERENCE or plan.graph_enabled:
            raise ValueError("ReferenceOpExecutor requires an eager reference plan")
        self._plan = plan
        self._catalog = catalog
        self._bindings = {binding.signature: binding for binding in plan.bindings}
        self._functions: dict[OpSignature, Any] = {}
        self._call_signatures: dict[OpSignature, inspect.Signature] = {}
        if len(self._bindings) != len(plan.bindings):
            raise ValueError("reference plan contains duplicate operator signatures")
        for binding in plan.bindings:
            function = resolve_reference_binding(binding, catalog)
            try:
                call_signature = inspect.signature(function)
            except (TypeError, ValueError) as error:
                raise TypeError("reference implementations must have an inspectable ABI") from error
            parameters = call_signature.parameters
            accepts_arbitrary_keywords = any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
            )
            missing_names = tuple(
                tensor.name
                for tensor in binding.signature.inputs
                if tensor.name not in parameters and not accepts_arbitrary_keywords
            )
            if missing_names:
                raise TypeError(
                    "reference implementation is missing declared tensor operands: "
                    + ", ".join(missing_names)
                )
            self._functions[binding.signature] = function
            self._call_signatures[binding.signature] = call_signature

    @property
    def signatures(self) -> tuple[OpSignature, ...]:
        return tuple(binding.signature for binding in self._plan.bindings)

    def bind_segment(self, identity: ReferenceSegmentIdentity) -> ReferenceOpCapability:
        """Mint the only operator authority exposed to one model invocation."""

        require_reference_execution()
        if not isinstance(identity, ReferenceSegmentIdentity):
            raise TypeError("reference operator capability requires a segment identity")
        return ReferenceOpCapability(self, identity)

    def call(self, signature: OpSignature, /, *args: Any, **kwargs: Any) -> Any:
        require_reference_execution()
        try:
            function = self._functions[signature]
            call_signature = self._call_signatures[signature]
        except KeyError as error:
            raise LookupError(
                "model requested an operator signature absent from the sealed plan"
            ) from error

        call_kwargs = dict(kwargs)
        try:
            partial = call_signature.bind_partial(*args, **call_kwargs)
        except TypeError as error:
            raise TypeError("operator call does not match the sealed callable ABI") from error
        _seal_dtype_control(
            "accumulator_dtype",
            signature.accumulator_dtype,
            call_signature,
            partial.arguments,
            call_kwargs,
        )
        if len(signature.outputs) == 1:
            _seal_dtype_control(
                "output_dtype",
                signature.outputs[0].dtype,
                call_signature,
                partial.arguments,
                call_kwargs,
            )
        try:
            bound = call_signature.bind(*args, **call_kwargs)
        except TypeError as error:
            raise TypeError("operator call does not satisfy the sealed callable ABI") from error

        declared_names = {tensor.name for tensor in signature.inputs}
        for name, value in bound.arguments.items():
            if name not in declared_names and _contains_tensor(value):
                raise TypeError(f"operator call supplied undeclared tensor operand {name!r}")

        dimensions = dict(signature.shape_items)
        for tensor in signature.inputs:
            if tensor.name not in bound.arguments:
                raise TypeError(f"operator call omitted tensor operand {tensor.name!r}")
            _validate_tensor(
                bound.arguments[tensor.name],
                tensor,
                dimensions,
                kind="input",
            )

        result = function(*args, **call_kwargs)
        values = _output_values(result, len(signature.outputs))
        for tensor, value in zip(signature.outputs, values, strict=True):
            _validate_tensor(value, tensor, dimensions, kind="output")
        return result


class ReferenceOpCapability:
    """Revocable phase/topology-bound view of a sealed reference dispatcher.

    The model program never receives the engine-wide dispatcher. Retaining this
    object after its segment callback returns is harmless: every later call
    fails before an implementation can run.
    """

    __slots__ = ("_active", "_call_count", "_executor", "_identity")

    def __init__(
        self,
        executor: ReferenceOpExecutor,
        identity: ReferenceSegmentIdentity,
    ) -> None:
        self._executor = executor
        self._identity = identity
        self._active = True
        self._call_count = 0

    def call(self, signature: OpSignature, /, *args: Any, **kwargs: Any) -> Any:
        require_reference_execution()
        if not self._active:
            raise RuntimeError("reference operator capability was revoked")
        if signature.phase is not self._identity.segment.phase:
            raise ValueError("operator signature phase differs from its active segment")
        if signature.topology_id != self._identity.topology_id:
            raise ValueError("operator signature topology differs from its active segment")
        result = self._executor.call(signature, *args, **kwargs)
        self._call_count += 1
        return result

    def verify_used(self) -> None:
        require_reference_execution()
        if not self._active:
            raise RuntimeError("reference operator capability was revoked")
        if self._call_count == 0:
            raise RuntimeError("model segment bypassed the sealed reference operator catalog")

    def revoke(self) -> None:
        self._active = False


def _seal_dtype_control(
    name: str,
    expected_dtype: ScalarDType,
    call_signature: inspect.Signature,
    bound_arguments: dict[str, Any],
    call_kwargs: dict[str, Any],
) -> None:
    expected = _torch_dtype(expected_dtype)
    parameter = call_signature.parameters.get(name)
    if parameter is None:
        return
    if name in bound_arguments:
        if bound_arguments[name] is not expected:
            raise TypeError(f"operator {name} differs from the sealed signature")
        return
    if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
        raise TypeError(f"operator {name} must be explicit because it is positional-only")
    call_kwargs[name] = expected


def _validate_tensor(
    value: Any,
    spec: TensorSpec,
    dimensions: dict[str, int],
    *,
    kind: str,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"operator {kind} {spec.name!r} must be a torch.Tensor")
    if value.layout is not torch.strided:
        raise TypeError(f"operator {kind} {spec.name!r} must use strided tensor storage")
    expected_dtype = _torch_dtype(spec.dtype)
    if value.dtype is not expected_dtype:
        raise TypeError(
            f"operator {kind} {spec.name!r} dtype must be {expected_dtype}; found {value.dtype}"
        )
    if value.ndim != len(spec.shape):
        raise ValueError(
            f"operator {kind} {spec.name!r} rank must be {len(spec.shape)}; found {value.ndim}"
        )
    for axis, (constraint, actual) in enumerate(zip(spec.shape, value.shape, strict=True)):
        if actual <= 0:
            raise ValueError(f"operator {kind} {spec.name!r} axis {axis} cannot be empty")
        if type(constraint) is int:
            expected = constraint
        else:
            assert isinstance(constraint, str)
            expected = dimensions.setdefault(constraint, actual)
        if actual != expected:
            raise ValueError(
                f"operator {kind} {spec.name!r} axis {axis} must be {expected}; found {actual}"
            )


def _output_values(result: Any, output_count: int) -> tuple[Any, ...]:
    if output_count == 1:
        if not isinstance(result, torch.Tensor):
            raise TypeError("single-output operator must return one torch.Tensor")
        return (result,)
    if type(result) is not tuple or len(result) != output_count:
        raise TypeError(f"operator must return an exact {output_count}-tensor tuple")
    return result


def _contains_tensor(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, dict):
        return any(_contains_tensor(item) for item in value.values())
    if isinstance(value, list | tuple):
        return any(_contains_tensor(item) for item in value)
    return False


def _torch_dtype(dtype: ScalarDType) -> torch.dtype:
    mapping = {
        ScalarDType.BF16: torch.bfloat16,
        ScalarDType.FP16: torch.float16,
        ScalarDType.FP32: torch.float32,
        ScalarDType.FP8_E4M3FN: torch.float8_e4m3fn,
        ScalarDType.FP8_E5M2: torch.float8_e5m2,
        ScalarDType.INT8: torch.int8,
        ScalarDType.INT64: torch.int64,
    }
    try:
        return mapping[dtype]
    except KeyError as error:
        raise ValueError(
            f"tensor dtype {dtype.value!r} has no unambiguous torch.Tensor storage ABI"
        ) from error
