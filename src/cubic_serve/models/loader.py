"""Load one model adapter from one explicitly supplied Python source path."""

from __future__ import annotations

import ast
import hashlib
import inspect
import pickle
import sys
from contextlib import suppress
from copy import copy
from dataclasses import asdict, dataclass
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec
from importlib.util import module_from_spec, spec_from_file_location, spec_from_loader
from os import PathLike
from pathlib import Path
from types import ModuleType
from typing import Any
from uuid import uuid4

from cubic_serve.contracts.execution import DecodeProvider, ExecutionPlan
from cubic_serve.executor.runner import require_reference_execution
from cubic_serve.models.adapter import (
    MODEL_ADAPTER_API_VERSION,
    ModelAdapter,
    ModelAdapterDescriptor,
    ModelDefinition,
)
from cubic_serve.models.protocol import ModelBuildContext, ModelProgram, ProgramExecutionRecipe
from cubic_serve.ops.catalog import (
    ReferenceCatalog,
    builtin_reference_catalog,
    seal_reference_plan,
)

_API_ATTRIBUTE = "CUBIC_MODEL_ADAPTER_API_VERSION"
_FACTORY_ATTRIBUTE = "create_model_adapter"
_MODEL_ATTRIBUTE = "MODEL"


class ModelAdapterLoadError(RuntimeError):
    """Base failure for explicit model adapter loading."""


class ModelAdapterABIError(ModelAdapterLoadError):
    """The external module does not implement the fixed adapter ABI."""


class ModelAdapterValidationError(ModelAdapterLoadError):
    """The adapter declaration is valid Python but is not admissible."""


class ModelAdapterSourcePolicyError(ModelAdapterValidationError):
    """The adapter source requests execution modes outside the eager boundary."""


_TORCH_NAMESPACE = "torch"
_AOT_AUTOGRAD = "aot_" + "autograd"
_FORBIDDEN_COMPILE_API_PREFIXES = (
    "functorch." + "compile",
    f"{_TORCH_NAMESPACE}._" + "dynamo",
    f"{_TORCH_NAMESPACE}._export",
    f"{_TORCH_NAMESPACE}._functorch._{_AOT_AUTOGRAD}",
    f"{_TORCH_NAMESPACE}._functorch.{_AOT_AUTOGRAD}",
    f"{_TORCH_NAMESPACE}._functorch.compilers",
    f"{_TORCH_NAMESPACE}._" + "inductor",
    f"{_TORCH_NAMESPACE}." + "compile",
    f"{_TORCH_NAMESPACE}.compiler",
    f"{_TORCH_NAMESPACE}.export",
    f"{_TORCH_NAMESPACE}.jit",
)
_FORBIDDEN_COMPILE_APIS = frozenset(
    {
        "torch.fx.experimental.proxy_tensor.make_fx",
        "torch.fx.symbolic_trace",
    }
)
_FORBIDDEN_REFERENCE_API_PREFIXES: tuple[tuple[str, str], ...] = (
    ("flash_attn", "third-party fused kernels"),
    ("triton", "custom Triton kernels and autotune"),
    ("xformers.ops", "third-party fused kernels"),
    ("torch.autocast", "adapter-controlled autocast"),
    ("torch.backends.cudnn.benchmark", "adapter-controlled autotune"),
    ("torch.cuda.CUDAGraph", "CUDA Graph"),
    ("torch.cuda.graph", "CUDA Graph"),
    ("torch.cuda.graphs", "CUDA Graph"),
    ("torch.cuda.make_graphed_callables", "CUDA Graph"),
    ("torch.cuda.amp", "adapter-controlled autocast"),
    ("torch.library", "custom operator registration"),
    ("torch.ops", "custom/fused operator dispatch"),
    ("torch.utils.cpp_extension", "runtime native-extension loading"),
)
_FORBIDDEN_REFERENCE_APIS = {
    "torch.nn.functional.scaled_dot_product_attention": "fused attention dispatch",
}


def _is_forbidden_compile_api(qualified_name: str) -> bool:
    if qualified_name in _FORBIDDEN_COMPILE_APIS:
        return True
    return any(
        qualified_name == prefix or qualified_name.startswith(f"{prefix}.")
        for prefix in _FORBIDDEN_COMPILE_API_PREFIXES
    )


def _forbidden_source_policy(qualified_name: str) -> str | None:
    if _is_forbidden_compile_api(qualified_name):
        return "compilation"
    exact = _FORBIDDEN_REFERENCE_APIS.get(qualified_name)
    if exact is not None:
        return exact
    for prefix, policy in _FORBIDDEN_REFERENCE_API_PREFIXES:
        if qualified_name == prefix or qualified_name.startswith(f"{prefix}."):
            return policy
    return None


class _EagerOnlySourceAuditor(ast.NodeVisitor):
    """Reject statically resolvable APIs incompatible with eager reference mode.

    This is an admission boundary for trusted adapter source, not a sandbox. In
    particular, Python can construct names through dynamic reflection that an AST
    audit cannot prove safe. Runtime isolation is a separate concern.
    """

    def __init__(self, source: Path) -> None:
        self._source = source
        self._scopes: list[dict[str, str | None]] = [{}]

    def _reject(self, node: ast.AST, qualified_name: str, policy: str) -> None:
        line = getattr(node, "lineno", 1)
        column = getattr(node, "col_offset", 0) + 1
        if policy == "compilation":
            detail = "torch compilation is unsupported"
        else:
            detail = f"{policy} is unavailable in reference execution"
        raise ModelAdapterSourcePolicyError(
            "model adapter source requests unsupported eager reference API "
            f"{qualified_name!r} at {self._source}:{line}:{column}; "
            f"{detail} and adapters must remain eager-only"
        )

    def _bind(self, name: str, qualified_name: str | None) -> None:
        self._scopes[-1][name] = qualified_name

    def _resolve_name(self, name: str) -> str | None:
        for scope in reversed(self._scopes):
            if name in scope:
                return scope[name]
        return None

    def _resolve_expression(self, node: ast.expr) -> str | None:
        if isinstance(node, ast.Name):
            return self._resolve_name(node.id)
        if isinstance(node, ast.Attribute):
            owner = self._resolve_expression(node.value)
            if owner is not None:
                return f"{owner}.{node.attr}"
        return None

    def _bind_target(self, target: ast.expr, qualified_name: str | None = None) -> None:
        if isinstance(target, ast.Name):
            self._bind(target.id, qualified_name)
            return
        if isinstance(target, ast.List | ast.Tuple):
            for element in target.elts:
                self._bind_target(element)

    def _push_function_scope(self, arguments: ast.arguments) -> None:
        self._scopes.append({})
        positional = (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs)
        for argument in positional:
            self._bind(argument.arg, None)
        if arguments.vararg is not None:
            self._bind(arguments.vararg.arg, None)
        if arguments.kwarg is not None:
            self._bind(arguments.kwarg.arg, None)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            policy = _forbidden_source_policy(alias.name)
            if policy is not None:
                self._reject(node, alias.name, policy)
            local_name = alias.asname or alias.name.partition(".")[0]
            qualified_name = alias.name if alias.asname else local_name
            self._bind(local_name, qualified_name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module_name = node.module or ""
        policy = _forbidden_source_policy(module_name)
        if policy is not None:
            self._reject(node, module_name, policy)
        for alias in node.names:
            if alias.name == "*":
                if module_name in {"functorch", "torch"}:
                    self._reject(node, f"{module_name}.*", "compilation")
                continue
            qualified_name = f"{module_name}.{alias.name}" if module_name else alias.name
            policy = _forbidden_source_policy(qualified_name)
            if policy is not None:
                self._reject(node, qualified_name, policy)
            self._bind(alias.asname or alias.name, qualified_name)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        qualified_name = self._resolve_expression(node)
        policy = None if qualified_name is None else _forbidden_source_policy(qualified_name)
        if qualified_name is not None and policy is not None:
            self._reject(node, qualified_name, policy)
        self.visit(node.value)

    def visit_Name(self, node: ast.Name) -> None:
        if not isinstance(node.ctx, ast.Load):
            return
        qualified_name = self._resolve_name(node.id)
        policy = None if qualified_name is None else _forbidden_source_policy(qualified_name)
        if qualified_name is not None and policy is not None:
            self._reject(node, qualified_name, policy)

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        qualified_name = self._resolve_expression(node.value)
        for target in node.targets:
            self._bind_target(target, qualified_name)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self.visit(node.annotation)
        if node.value is not None:
            self.visit(node.value)
            qualified_name = self._resolve_expression(node.value)
        else:
            qualified_name = None
        self._bind_target(node.target, qualified_name)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.visit(node.value)
        self._bind_target(node.target, self._resolve_expression(node.value))

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        for default in (*node.args.defaults, *node.args.kw_defaults):
            if default is not None:
                self.visit(default)
        if node.returns is not None:
            self.visit(node.returns)
        self._bind(node.name, None)
        self._push_function_scope(node.args)
        try:
            for statement in node.body:
                self.visit(statement)
        finally:
            self._scopes.pop()

    def visit_Lambda(self, node: ast.Lambda) -> None:
        for default in (*node.args.defaults, *node.args.kw_defaults):
            if default is not None:
                self.visit(default)
        self._push_function_scope(node.args)
        try:
            self.visit(node.body)
        finally:
            self._scopes.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        for base in node.bases:
            self.visit(base)
        for keyword in node.keywords:
            self.visit(keyword.value)
        self._bind(node.name, None)
        self._scopes.append({})
        try:
            for statement in node.body:
                self.visit(statement)
        finally:
            self._scopes.pop()


def _audit_eager_only_source(source: Path, source_bytes: bytes) -> None:
    try:
        tree = ast.parse(source_bytes, filename=str(source), mode="exec")
    except SyntaxError as error:
        raise ModelAdapterLoadError(f"model adapter source is invalid Python: {source}") from error
    _EagerOnlySourceAuditor(source).visit(tree)


@dataclass(frozen=True, slots=True)
class LoadedModelAdapter:
    """One isolated adapter instance and its sealed reference execution plan."""

    source_path: Path
    source_digest: str
    module_name: str
    adapter: ModelAdapter
    descriptor: ModelAdapterDescriptor
    reference_catalog: ReferenceCatalog
    reference_plan: ExecutionPlan

    def build_program(self, context: ModelBuildContext) -> ModelProgram:
        # This public boundary fails closed even if a future caller forgets the
        # engine's surrounding context manager.
        require_reference_execution()
        try:
            program = self.adapter.build(context)
        except Exception as error:
            raise ModelAdapterLoadError("model adapter build() raised an exception") from error
        require_reference_execution()
        if not isinstance(program, ModelProgram):
            raise ModelAdapterABIError("ModelAdapter.build() must return a ModelProgram")
        return program


def _resolve_source(path: str | PathLike[str]) -> Path:
    source = Path(path)
    try:
        source = source.resolve(strict=True)
    except FileNotFoundError as error:
        raise ModelAdapterLoadError(f"model adapter path does not exist: {source}") from error
    if source.is_dir():
        source = source / "__init__.py"
        if not source.is_file():
            raise ModelAdapterLoadError("model adapter directory must contain __init__.py")
    if source.suffix != ".py":
        raise ModelAdapterLoadError("model adapter path must name a directory or .py file")
    if not source.is_file():
        raise ModelAdapterLoadError(f"model adapter path is not a file: {source}")
    return source


def _execute_isolated(source: Path) -> tuple[ModuleType, str, str]:
    package = source.name == "__init__.py"
    sources = tuple(sorted(source.parent.glob("*.py"))) if package else (source,)
    digest = hashlib.blake2s()
    for item in sources:
        source_bytes = item.read_bytes()
        _audit_eager_only_source(item, source_bytes)
        digest.update(item.name.encode())
        digest.update(b"\0")
        digest.update(source_bytes)
        digest.update(b"\0")
    source_bytes = source.read_bytes()
    source_digest = f"b2s256:{digest.hexdigest()}"
    path_tag = hashlib.blake2s(str(source).encode()).hexdigest()[:12]
    module_name = f"_cubic_serve_adapter_{path_tag}_{uuid4().hex}"
    spec = spec_from_file_location(
        module_name,
        source,
        submodule_search_locations=[str(source.parent)] if package else None,
    )
    if spec is None:
        raise ModelAdapterLoadError(f"cannot construct a module spec for {source}")
    module = module_from_spec(spec)

    # A transient entry supports ordinary Python features such as dataclasses and
    # self-import during execution. It is always removed: this is not a registry.
    sys.modules[module_name] = module
    try:
        code = compile(source_bytes, str(source), "exec", dont_inherit=True)
        exec(code, module.__dict__)
    except Exception as error:
        raise ModelAdapterLoadError(f"model adapter module failed to load: {source}") from error
    finally:
        for loaded_name in tuple(sys.modules):
            if loaded_name == module_name or loaded_name.startswith(module_name + "."):
                sys.modules.pop(loaded_name, None)
    return module, module_name, source_digest


def _create_adapter(module: ModuleType) -> ModelAdapter:
    version = getattr(module, _API_ATTRIBUTE, None)
    if type(version) is not int or version != MODEL_ADAPTER_API_VERSION:
        raise ModelAdapterABIError(
            f"{_API_ATTRIBUTE} must equal {MODEL_ADAPTER_API_VERSION}; found {version!r}"
        )

    declared = getattr(module, _MODEL_ATTRIBUTE, None)
    factory = getattr(module, _FACTORY_ATTRIBUTE, None)
    if declared is None and not callable(factory):
        raise ModelAdapterABIError(
            f"adapter module must export {_MODEL_ATTRIBUTE} or callable {_FACTORY_ATTRIBUTE}()"
        )
    if declared is not None and not isinstance(declared, ModelAdapter):
        raise ModelAdapterABIError(
            f"{_MODEL_ATTRIBUTE} must implement ModelAdapter.describe() and build()"
        )
    if factory is None:
        assert isinstance(declared, ModelAdapter)
        return declared
    if not callable(factory):
        raise ModelAdapterABIError(f"{_FACTORY_ATTRIBUTE} must be callable when exported")
    if inspect.iscoroutinefunction(factory):
        raise ModelAdapterABIError("adapter factory must be synchronous")
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError) as error:
        raise ModelAdapterABIError(
            "adapter factory must have an inspectable zero-argument ABI"
        ) from error
    if signature.parameters:
        raise ModelAdapterABIError("adapter factory ABI is exactly create_model_adapter()")

    try:
        instance = factory()
    except Exception as error:
        raise ModelAdapterLoadError("model adapter factory raised an exception") from error
    if inspect.isawaitable(instance):
        raise ModelAdapterABIError("adapter factory must return synchronously")
    if not isinstance(instance, ModelAdapter):
        raise ModelAdapterABIError(
            "adapter factory result must implement ModelAdapter.describe() and build()"
        )
    if declared is not None and instance is not declared:
        raise ModelAdapterABIError("create_model_adapter() must return the exported MODEL")
    return instance


def _validate_and_seal(
    adapter: ModelAdapter,
) -> tuple[ModelAdapterDescriptor, ReferenceCatalog, ExecutionPlan]:
    try:
        descriptor = adapter.describe()
    except Exception as error:
        raise ModelAdapterLoadError("model adapter describe() raised an exception") from error
    if not isinstance(descriptor, ModelAdapterDescriptor):
        raise ModelAdapterABIError("ModelAdapter.describe() must return ModelAdapterDescriptor")

    try:
        descriptor.model_spec.require_full_bf16()
        if not descriptor.weight_variant.is_base:
            raise ValueError("v0.1 rejects non-base weight variants (including LoRA)")
        catalog = builtin_reference_catalog().extend(descriptor.reference_implementations)
        plan = seal_reference_plan(
            execution_plan_id=(f"reference:{descriptor.adapter_id}:{descriptor.adapter_revision}"),
            signatures=descriptor.required_ops,
            catalog=catalog,
        )
    except (LookupError, TypeError, ValueError) as error:
        raise ModelAdapterValidationError(
            f"model adapter declaration cannot be admitted: {error}"
        ) from error
    return descriptor, catalog, plan


def seal_builtin_model_adapter(
    adapter: ModelAdapter,
    *,
    source_path: str | PathLike[str],
) -> LoadedModelAdapter:
    """Seal one explicitly constructed in-tree adapter without a registry lookup."""

    if not isinstance(adapter, ModelAdapter):
        raise ModelAdapterABIError("built-in adapter must implement the ModelAdapter ABI")
    source = _resolve_source(source_path)
    source_bytes = source.read_bytes()
    _audit_eager_only_source(source, source_bytes)
    descriptor, catalog, plan = _validate_and_seal(adapter)
    return LoadedModelAdapter(
        source_path=source,
        source_digest=f"b2s256:{hashlib.blake2s(source_bytes).hexdigest()}",
        module_name=type(adapter).__module__,
        adapter=adapter,
        descriptor=descriptor,
        reference_catalog=catalog,
        reference_plan=plan,
    )


def load_model_adapter(path: str | PathLike[str]) -> LoadedModelAdapter:
    """Load, validate, and seal a fresh adapter instance from an explicit path.

    The path may name one ``.py`` source or one adapter directory containing
    ``__init__.py`` and may be absolute or relative to the caller's working directory. No
    adapter registry, import cache, or ``sys.path`` mutation is used. Before any
    source execution, a conservative AST admission check rejects statically
    resolvable compilation, graph, autotune, and custom/fused-kernel APIs. This
    is a policy boundary for trusted external code, not a sandbox, transitive
    dependency audit, or protection against dynamic reflection.
    """

    source = _resolve_source(path)
    module, module_name, source_digest = _execute_isolated(source)
    adapter = _create_adapter(module)
    descriptor, catalog, plan = _validate_and_seal(adapter)
    return LoadedModelAdapter(
        source_path=source,
        source_digest=source_digest,
        module_name=module_name,
        adapter=adapter,
        descriptor=descriptor,
        reference_catalog=catalog,
        reference_plan=plan,
    )


class _DefinitionSourceAuditor(_EagerOnlySourceAuditor):
    """Keep implicit Torch compilation out of trusted native model definitions.

    Reference ABI1 has a separate eager-only policy. Native definition modules
    may bind shared format operators and native helpers; importing those APIs
    grants no execution, CUDA Graph, resource or qualification authority.
    Like the original auditor, this static check is not a Python sandbox.
    """

    def _reject(self, node: ast.AST, qualified_name: str, policy: str) -> None:
        if policy == "compilation":
            raise ModelAdapterSourcePolicyError(
                "model definition source requests unsupported torch compilation "
                f"{qualified_name!r} at {self._source}:{getattr(node, 'lineno', 1)}"
            )


def _resolve_definition_source(path: str | PathLike[str]) -> Path:
    source = Path(path)
    if source.is_dir():
        source = source / "__init__.py"
    try:
        # Preserve the Python filename of HF snapshot links into extensionless blobs.
        source = source.parent.resolve(strict=True) / source.name
        if source.suffix != ".py" or not source.is_file():
            raise ModelAdapterLoadError("model definition requires a directory or Python file")
    except OSError as error:
        raise ModelAdapterLoadError("model definition source is unavailable") from error
    return source


def _definition_sources(
    source: Path,
) -> tuple[dict[str, tuple[Path, bytes | None, bool]], str]:
    root = source.parent
    digest = hashlib.blake2s()
    sources: dict[str, tuple[Path, bytes | None, bool]] = {}
    for path in sorted(root.rglob("*.py")):
        data = path.read_bytes()
        try:
            _DefinitionSourceAuditor(path).visit(ast.parse(data, filename=str(path)))
        except SyntaxError as error:
            raise ModelAdapterSourcePolicyError(
                "model definition contains invalid Python"
            ) from error
        relative = path.relative_to(root)
        digest.update(relative.as_posix().encode() + b"\0" + data + b"\0")
        package = path == source or (path.name == "__init__.py" and path.parent != root)
        parts = (relative.parent.parts if package else relative.with_suffix("").parts)
        if path == source:
            key = ""
        elif all(part.isidentifier() for part in parts):
            key = ".".join(parts)
        else:
            continue  # Attested non-module Python filenames are never imported implicitly.
        prior = sources.get(key)
        # Match Python's package-before-module resolution for a shared basename.
        if prior is None or package:
            sources[key] = path, data, package
    if "" not in sources:
        raise ModelAdapterLoadError("model definition entry disappeared during admission")
    for key in tuple(sources):
        parts = key.split(".")
        for depth in range(1, len(parts)):
            parent = ".".join(parts[:depth])
            sources.setdefault(parent, (root.joinpath(*parts[:depth]), None, True))
    return sources, f"b2s256:{digest.hexdigest()}"


class _DefinitionNamespace(MetaPathFinder, Loader):
    """One owned package, executing admitted source bytes, never cached pyc."""

    def __init__(self, source: Path) -> None:
        self.sources, self.digest = _definition_sources(source)
        self.name = f"_cubic_serve_definition_{uuid4().hex}"
        self.closed = False

    def find_spec(self, fullname: str, path: object = None, target: ModuleType | None = None):
        if self.closed or (fullname != self.name and not fullname.startswith(self.name + ".")):
            return None
        key = "" if fullname == self.name else fullname.removeprefix(self.name + ".")
        entry = self.sources.get(key)
        if entry is None:
            return None
        source, data, package = entry
        spec = spec_from_loader(
            fullname, self, origin=str(source) if data is not None else None, is_package=package
        )
        assert spec is not None
        spec.has_location = data is not None
        return spec

    def create_module(self, spec: ModuleSpec) -> ModuleType | None:
        return None

    def exec_module(self, module: ModuleType) -> None:
        key = "" if module.__name__ == self.name else module.__name__.removeprefix(self.name + ".")
        source, data, _package = self.sources[key]
        if data is not None:
            exec(compile(data, str(source), "exec", dont_inherit=True), module.__dict__)

    def close(self) -> None:
        self.closed = True
        with suppress(ValueError):
            sys.meta_path.remove(self)
        for name in tuple(sys.modules):
            if name == self.name or name.startswith(self.name + "."):
                sys.modules.pop(name, None)
        self.sources.clear()


@dataclass(frozen=True, slots=True)
class LoadedModelDefinition:
    """Owned startup declaration; close after its admitted instance shuts down.

    No execution-mode override or transitive dependency attestation is granted.
    Relative imports use the immutable package snapshot for the whole lifetime.
    """

    source_path: Path
    source_digest: str
    module_name: str
    definition: ModelDefinition[Any]
    _namespace: _DefinitionNamespace

    def verify_source(self) -> None:
        if self._namespace.closed:
            raise ModelAdapterLoadError("model definition is closed")
        try:
            _, digest = _definition_sources(self.source_path)
        except OSError as error:
            raise ModelAdapterLoadError("model definition source is unavailable") from error
        if digest != self.source_digest:
            raise ModelAdapterLoadError("model definition source changed during admission")

    def close(self) -> None:
        self._namespace.close()

    def execution_recipe(self, config: Any) -> ProgramExecutionRecipe:
        """Reuse the existing recipe contract for declared semantic dependencies."""
        if self._namespace.closed:
            raise ModelAdapterLoadError("model definition is closed")
        factory = self.definition.program_recipe
        if factory is None:
            from cubic_serve.guardrail.compare import stable_digest

            try:
                config_digest = stable_digest(asdict(config))
            except (TypeError, ValueError) as error:
                raise ModelAdapterValidationError(
                    "noncanonical model config requires a declared program recipe"
                ) from error
            return ProgramExecutionRecipe(
                recipe_id="cubic.model-definition",
                revision=self.source_digest,
                settings=(("config", config_digest), ("model_type", self.definition.model_type)),
            )
        recipe = factory(config)
        if not isinstance(recipe, ProgramExecutionRecipe):
            raise ModelAdapterValidationError("model program recipe has an invalid type")
        return recipe

    def program_identity(self, base_digest: str, recipe: ProgramExecutionRecipe) -> str:
        """Bind owned source and semantic recipe without changing checkpoint identity."""
        from cubic_serve.guardrail.compare import stable_digest

        if self._namespace.closed:
            raise ModelAdapterLoadError("model definition is closed")
        if not isinstance(recipe, ProgramExecutionRecipe):
            raise ModelAdapterValidationError("model program recipe has an invalid type")
        return stable_digest({
            "base": base_digest, "model_definition": self.source_digest,
            "program_recipe": asdict(recipe),
        })

    def __reduce__(self):
        self.verify_source()
        return _restore_model_definition, (self.source_path, self.source_digest, self.module_name)

    def bind_metadata(self, value: Any) -> _DefinitionBoundMetadata:
        """Bootstrap source before restoring controller-owned admission metadata."""
        attached = getattr(value, "model_definition_source", None)
        if attached is not None:
            if attached is not self:
                raise ModelAdapterLoadError("metadata belongs to a different model definition")
            value = copy(value)
            del value.model_definition_source
        return _DefinitionBoundMetadata(self, value)

    def __enter__(self) -> LoadedModelDefinition:
        try:
            self.verify_source()
        except BaseException:
            self.close()
            raise
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def load_model_definition(path: str | PathLike[str]) -> LoadedModelDefinition:
    """Load an explicitly supplied MODEL_DEFINITION; never discover repository code.

    Each owned namespace supports package-relative imports from admitted Python
    bytes without changing sys.path or registering a global model override. This
    admits trusted code, not a Python sandbox. Legacy reference ABI1 is unchanged.
    """
    source = _resolve_definition_source(path)
    namespace = _DefinitionNamespace(source)
    return _load_definition_namespace(source, namespace)


def _load_definition_namespace(
    source: Path, namespace: _DefinitionNamespace
) -> LoadedModelDefinition:
    sys.meta_path.insert(0, namespace)
    try:
        spec = namespace.find_spec(namespace.name)
        assert spec is not None
        module = module_from_spec(spec)
        sys.modules[namespace.name] = module
        namespace.exec_module(module)
        definition = getattr(module, "MODEL_DEFINITION", None)
        if not isinstance(definition, ModelDefinition):
            raise ModelAdapterABIError("adapter must export a ModelDefinition as MODEL_DEFINITION")
        loaded = LoadedModelDefinition(
            source, namespace.digest, namespace.name, definition, namespace
        )
        loaded.verify_source()
        return loaded
    except BaseException as error:
        namespace.close()
        if isinstance(error, Exception) and not isinstance(error, ModelAdapterLoadError):
            raise ModelAdapterLoadError("model definition module failed to load") from error
        raise


def _restore_model_definition(
    source_path: Path, expected_digest: str, module_name: str
) -> LoadedModelDefinition:
    source = _resolve_definition_source(source_path)
    if (
        not isinstance(module_name, str)
        or not module_name.startswith("_cubic_serve_definition_")
        or not module_name.isidentifier()
    ):
        raise ModelAdapterLoadError("invalid owned definition namespace")
    if any(name == module_name or name.startswith(module_name + ".") for name in sys.modules):
        raise ModelAdapterLoadError("owned definition namespace is already in use")
    namespace = _DefinitionNamespace(source)
    if namespace.digest != expected_digest:
        namespace.close()
        raise ModelAdapterLoadError("model definition source changed before worker admission")
    namespace.name = module_name
    return _load_definition_namespace(source, namespace)


@dataclass(frozen=True, slots=True)
class _DefinitionBoundMetadata:
    owner: LoadedModelDefinition
    value: Any

    def __reduce__(self):
        # Nested bytes defer custom-class lookup until the owned source exists.
        # This is only the controller's private local metadata, never a checkpoint
        # or a public/user-supplied pickle.
        return _restore_definition_metadata, (
            self.owner, pickle.dumps(self.value, protocol=pickle.HIGHEST_PROTOCOL),
        )


def _restore_definition_metadata(owner: LoadedModelDefinition, payload: bytes) -> Any:
    try:
        value = pickle.loads(payload)
        value.model_definition_source = owner
        return value
    except BaseException:
        owner.close()
        raise
