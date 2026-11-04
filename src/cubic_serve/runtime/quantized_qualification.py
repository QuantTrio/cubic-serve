"""Live execution checks for the packed-weight graph portfolio.

These witnesses establish capture, padding and cache/state identity. They do
not replace the independent original-weight numerical oracle and cannot alone
issue performance admission. The actual checks use the ordinary serving ABI,
shared with unquantized execution.
"""

from __future__ import annotations

import ctypes
import hashlib
import inspect
import logging
import json
import os
import struct
import sys
import time
import tempfile
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import asdict, fields, is_dataclass
from functools import cache, partial
from pathlib import Path
from types import CodeType, SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist

from cubic_serve.contracts.execution import RankReplayResources
from cubic_serve.contracts.implementation import implementation_source, implementation_sources
from cubic_serve.executor.text_serving import FlashNextDecodeStage
from cubic_serve.guardrail.compare import stable_digest
from cubic_serve.runtime.logging import diagnostic_event, startup_cache_loaded
from cubic_serve.runtime.performance import (
    _active_snapshot,
    _bounded_audit_logical_ms,
    _cold_start_graph_variant_audit,
    _cold_start_prefix_cache_audit,
    _logical_m_values,
    _prepare_audit,
    _run_bucket_migration_audit,
    _startup_check_phase,
)

_LOGGER = logging.getLogger(__name__)


def _receipt_pack(value):
    """JSON transport that preserves tuple keys in learned cost histories."""
    if isinstance(value, dict):
        return {"dict": [[_receipt_pack(k), _receipt_pack(v)] for k, v in value.items()]}
    if isinstance(value, tuple):
        return {"tuple": [_receipt_pack(v) for v in value]}
    if isinstance(value, list):
        return {"list": [_receipt_pack(v) for v in value]}
    if value is None or type(value) in (str, bool, int, float):
        return value
    raise TypeError(f"nonportable startup receipt value: {type(value).__name__}")


def _receipt_unpack(value):
    if not isinstance(value, dict):
        return value
    if set(value) == {"dict"}:
        return {_receipt_unpack(k): _receipt_unpack(v) for k, v in value["dict"]}
    if set(value) == {"tuple"}:
        return tuple(_receipt_unpack(v) for v in value["tuple"])
    if set(value) == {"list"}:
        return [_receipt_unpack(v) for v in value["list"]]
    raise ValueError("invalid startup receipt value")


class StartupQualification:
    """Restore completed engineering evidence; normal startup never produces it.

    Graphs, live addresses and resources are still created by their original
    owners. Only portable evidence and measured priors cross process lifetimes.
    A miss fails collectively instead of turning serve into a test runner.
    """

    engineering = True

    def __init__(self, key, rank, *, recording=False, root=None, consensus=None):
        from cubic_serve.runtime.performance import _qualification_cache_root

        self.rank, self.recording = rank, recording
        self.key = key
        self.path = (root or _qualification_cache_root()) / f"startup-rank{rank}-{key.removeprefix('b2s256:')}.json"
        self.consensus = consensus or (lambda value: value)
        self.records = {}
        self.counts = {}
        self.hits = 0
        self.failed = False
        self._plan_publications = []
        self.receipt_set = None
        self._prior_records = {}
        self._can_reuse = False
        self._reused = []
        try:
            raw = json.loads(self.path.read_text())
            body = raw["body"]
            valid = (raw["digest"] == stable_digest(body)
                     and body["schema"] == "cubic.startup-evidence.v1"
                     and body["key"] == key and body["rank"] == rank
                     and isinstance(body["records"], dict)
                     and isinstance(body["receipt_set"], str))
        except (OSError, ValueError, KeyError, TypeError):
            valid = False
        if not self.consensus(valid):
            if recording:
                return
            raise RuntimeError("startup evidence is missing or invalid; run the explicit 'cubic qualify' command for this configuration; serve will not run engineering tests")
        cohort = [None] * dist.get_world_size()
        dist.all_gather_object(cohort, (rank, body["receipt_set"], stable_digest(body["records"])))
        complete = (all(isinstance(item, tuple) and len(item) == 3
                        and item[0] == index for index, item in enumerate(cohort))
                    and all(item[1] == body["receipt_set"] for item in cohort)
                    and body["receipt_set"] == stable_digest([item[2] for item in cohort]))
        if not self.consensus(complete):
            if recording:
                return
            raise RuntimeError("startup evidence ranks belong to an incomplete qualification cohort")
        self.receipt_set = body["receipt_set"]
        if recording:
            # Only records actually requested by this workflow enter its new
            # receipt. A complete prior cohort supplies individual exact-input
            # proofs; it never supplies a missing operation or numerical audit.
            self._prior_records = body["records"]
            self._can_reuse = True
        else:
            self.records = body["records"]

    @classmethod
    def for_admission(cls, admission, rank, device):
        import triton
        from cuda.bindings import driver
        from cubic_serve.ops.cuda.launch_cost import _launch_calibration_device
        from cubic_serve.ops.cuda.native import _check

        identity = execution_identity(admission)
        devices = [None] * dist.get_world_size()
        dist.all_gather_object(devices, _launch_calibration_device(device))
        storage = getattr(admission, "cache_storage", None)
        key = stable_digest({
            "schema": "cubic.startup-plan.v2", "implementation": _composition_source_identity(),
            "inventory": identity.inventory.inventory_digest,
            "weights": identity.parallel_plan.weight_plan_digest,
            "semantic_plan": identity.parallel_plan.semantic_plan_digest,
            "topology": identity.sealed.topology_digest,
            "partition": identity.sealed.partition_digest,
            "operator_config": admission.operator_config.digest,
            "devices": devices, "capacities": admission.graph_capacities,
            "runtime": (sys.version, str(torch.__version__), str(torch.version.cuda),
                        triton.__version__, int(_check(driver.cuDriverGetVersion())[0])),
            "context": admission.max_sequence_tokens,
            "owners": admission.max_running_requests,
            "cache_storage": None if storage is None else storage.identity,
            "state_dtype": admission.mamba_ssm_cache_dtype,
            "draft": (admission.num_speculative_tokens,
                      admission.dynamic_max_speculative_tokens, admission.dynamic_speculation),
            "reasoning": (getattr(admission, "reasoning_token_boundaries", ()),
                          getattr(admission, "constraint_stop_token_ids", ())),
            "prefix": admission.use_prefix_cache,
        })

        def consensus(value):
            flags = torch.tensor([int(value)], dtype=torch.int32, device=device)
            dist.all_reduce(flags, op=dist.ReduceOp.MIN)
            return bool(flags.item())

        session = cls(key, rank, recording=(getattr(admission, "startup_qualification", False)
                                           or admission.startup_numerical_audit),
                      consensus=consensus)
        from cubic_serve.runtime.preparation import StartupPreparation
        topology = admission.parallel_plan.topology
        session.preparation = StartupPreparation(rank, device,
            topology=(topology.tp_size, topology.pp_size, topology.ep_size))
        return session

    def run(self, label, identity, action, *, reuse=True):
        ordinal = self.counts.get(label, 0)
        self.counts[label] = ordinal + 1
        name = f"{label}:{ordinal}"
        digest = stable_digest(identity)
        reuse = reuse and identity is not None
        if reuse and (not self.recording or self._can_reuse):
            source = self._prior_records if self.recording else self.records
            record = source.get(name)
            valid = isinstance(record, dict) and record.get("identity") == digest and "value" in record
            value = None
            if valid:
                try:
                    value = _receipt_unpack(record["value"])
                except (ValueError, TypeError, KeyError):
                    valid = False
            if self.consensus(valid):
                self.hits += 1
                if self.recording:
                    self.records[name] = record
                    self._reused.append(name)
                return value
        if not self.recording:
            self.failed = True
            raise RuntimeError(f"startup evidence does not cover bound operation {label}; run explicit qualification")
        if self.recording:
            try:
                value = action()
                record = {"identity": digest, "value": _receipt_pack(value)}
            except BaseException:
                self.failed = True
                raise
            self.records[name] = record
            return value

    def component(self, owner, identity, *, method="qualify", args=(), attributes=("qualified",)):
        def action():
            getattr(owner, method)(*args)
            return {name: getattr(owner, name) for name in attributes if hasattr(owner, name)}

        program = _qualification_program_identity(owner)
        state = self.run(f"{type(owner).__name__}.{method}", (identity, program), action,
                         reuse=program is not None)
        for name, value in state.items():
            setattr(owner, name, value)
        # The parent's evidence includes the independently checked candidate
        # producer; this is metadata restoration, never a candidate execution.
        candidate = getattr(owner, "_candidate_preparation", None)
        if candidate is not None and "evidence" in state:
            candidate.evidence = state["evidence"].get("local_candidates")

    def window(self, window, *, segments=None):
        executor = window.e
        identity = (_bound_composition_plan(executor.body), executor.captured_m,
                    executor.max_sequence_tokens, window.stochastic,
                    getattr(window, "draft_maintenance", None) is not None,
                    None if segments is None else tuple(segments))
        method = "qualify" if segments is None else "qualify_segments"
        attributes = ("qualified", "program_qualified", "segmented_program_qualified")
        def action():
            getattr(window, method)(*(() if segments is None else (segments,)))
            return ({name: getattr(window, name) for name in attributes if hasattr(window, name)},
                    executor._bound_active_rows)
        program = _qualification_program_identity(window)
        flags, active_rows = self.run(f"{type(window).__name__}.{method}",
            (identity, program), action, reuse=program is not None)
        for name, value in flags.items():
            setattr(window, name, value)
        executor._bound_active_rows = active_rows

    def compaction(self, owner, identity, action):
        # This PyTorch graph exposes geometry, but no portable identity for its
        # actual executable images. Re-prove each fresh instance until that
        # dependency can be bound; a geometry-only receipt cannot supply it.
        self.run("resident-compaction", identity, action, reuse=False)
        owner.qualified = True

    def costs(self, label, executors, action, *, entry=()):
        resource = SimpleNamespace(program=tuple(
            (executor.body, getattr(executor, "_sampling", None)) for executor in executors.values()))
        program = _qualification_program_identity(resource)
        values = self.run(label, (tuple(executors), entry, program), lambda: action(executors),
                          reuse=program is not None)
        if (preparation := getattr(self, "preparation", None)) is not None:
            self._plan_publications.append(
                lambda: preparation.publish_costs(label, executors, values, entry=entry))
        return values

    def draft_verifier(self, drafts, targets):
        from cubic_serve.runtime.performance import _qualify_draft_verification

        def action():
            _qualify_draft_verification(drafts, targets)
            return {capacity: draft._verifier_qualified for capacity, draft in drafts.items()}
        program = _qualification_program_identity(SimpleNamespace(
            program=(*drafts.values(), *targets.values())))
        flags = self.run("draft-verifier", (tuple(drafts), tuple(targets), program), action,
                         reuse=program is not None)
        if set(flags) != set(drafts) or any(value is not True for value in flags.values()):
            raise RuntimeError("startup evidence does not qualify every draft verifier")
        for capacity, qualified in flags.items():
            drafts[capacity]._verifier_qualified = qualified

    def draft_rounds(self, rounds, providers, targets, hidden, admission, *,
                     draft_provider=None, groups=None, **kwargs):
        attributes = ("target_constraints_qualified", "constraint_continuation_qualified",
                      "grammar_qualified", "grammar_continuation_qualified", "budget_qualified",
                      "_startup_round_costs", "_startup_constraint_round_costs")
        windows = ("_resident_window", "_candidate_window", "_budget_resident_window",
                   "_budget_candidate_window")
        numerical = bool(getattr(admission, "startup_numerical_audit", False))
        progress = kwargs.pop("progress", None)
        cases, native_count = (), 0
        if numerical:
            from cubic_serve.guardrail.quantized_graph.speculative import draft_numerical_case_plan
            if groups is None:
                raise ValueError("independent draft audit requires its bound process groups")
            cases = draft_numerical_case_plan(rounds, admission, draft_provider)
            native_count = sum(len(_draft_window_case_plan(
                key[0], round_, tuple(sorted(providers)), tuple(sorted(targets)), False))
                for key, round_ in rounds.items())
            def native_progress(completed, total):
                if total != native_count:
                    raise RuntimeError("native draft proof schedule differs")
                if progress is not None:
                    progress(completed, native_count + 3 * len(cases))
            kwargs["progress"] = native_progress
        elif progress is not None:
            kwargs["progress"] = progress

        def action():
            witness = qualify_draft_windows(rounds, providers, targets, hidden, admission,
                                            reuse_receipts=False, initialize_cold_fixture=True,
                                            **kwargs)
            values = {}
            for key, round_ in rounds.items():
                values[key] = (dict((name, getattr(round_, name)) for name in attributes
                                    if hasattr(round_, name)),
                               tuple((name, getattr(getattr(round_, name), "qualified"))
                                     for name in windows if getattr(round_, name, None) is not None))
            return witness, values

        witness, values = self.run("draft-composition",
                                  _draft_composition_receipt_key(rounds, providers, targets, admission),
                                  action)
        if set(values) != set(rounds):
            raise RuntimeError("startup evidence has a different draft portfolio")
        for key, (state, flags) in values.items():
            for name, value in state.items():
                setattr(rounds[key], name, value)
            for name, qualified in flags:
                getattr(rounds[key], name).qualified = qualified
        if numerical:
            from cubic_serve.guardrail.quantized_graph.speculative import audit_draft_portfolio_numerics
            total = native_count + 3 * len(cases)
            if progress is not None:
                progress(native_count, total)
            def numerical_progress(completed, expected):
                if expected != 3 * len(cases):
                    raise RuntimeError("independent draft proof schedule differs")
                if progress is not None:
                    progress(native_count + completed, total)
            numerical_witness = self.run("draft-independent-numerics",
                (tuple(sorted(rounds)), "greedy-stochastic-fixed-rng-v1"),
                lambda: audit_draft_portfolio_numerics(
                    rounds, providers, targets, hidden, admission,
                    draft_provider=draft_provider, groups=groups,
                    rank=kwargs["rank"], sampler_rank=kwargs["sampler_rank"], cases=cases,
                    completed_round=numerical_progress), reuse=False)
            witness = stable_digest({"native": witness, "independent": numerical_witness})
        if (preparation := getattr(self, "preparation", None)) is not None:
            self._plan_publications.append(lambda: preparation.publish_draft_costs(rounds))
        return witness

    def _publish_plans(self):
        for publish in self._plan_publications:
            publish()
        self._plan_publications.clear()

    def finish(self):
        if self.failed:
            raise RuntimeError("failed startup qualification cannot publish evidence")
        if not self.recording:
            if not self.consensus(self.hits == len(self.records) == sum(self.counts.values())):
                raise RuntimeError("startup evidence contains an unmatched execution schedule")
            self._publish_plans()
            startup_cache_loaded(self.rank, self.key)
            return
        complete = (len(self.records) == sum(self.counts.values())
                    and (not self._can_reuse or self._prior_records.keys() <= self.records.keys()))
        if not self.consensus(complete):
            self.failed = True
            raise RuntimeError("startup qualification cannot publish an unmatched prior execution schedule")
        digests = [None] * dist.get_world_size()
        dist.all_gather_object(digests, stable_digest(self.records))
        body = dict(schema="cubic.startup-evidence.v1", key=self.key, rank=self.rank,
                    records=self.records, receipt_set=stable_digest(digests))
        if self._reused:
            body["reused_from"] = dict(receipt_set=self.receipt_set, operations=self._reused)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", dir=self.path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(dict(body=body, digest=stable_digest(body)), stream, allow_nan=False)
        try:
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)
        self._publish_plans()
        self.receipt_set = body["receipt_set"]
        diagnostic_event(_LOGGER, 'quantized_qualification.startup_evidence_prepared_rank_d_operations_d',
            self.rank,
            len(self.records),
        )


@cache
def _composition_callback_source(path):
    return stable_digest(implementation_source(path))


@cache
def _composition_callback_code(code):
    # marshal's reference/intern flags change with harmless process references.
    # Canonical code structure binds loaded semantics without those identities,
    # filenames or debug line tables. Code objects retain no closure/buffers.
    def constant(value):
        if isinstance(value, CodeType):
            return ("code", _composition_callback_code(value))
        if value is None or type(value) in (bool, int, str):
            return (type(value).__name__, value)
        if isinstance(value, bytes):
            return ("bytes", value.hex())
        if isinstance(value, float):
            return ("float", struct.pack(">d", value).hex())
        if isinstance(value, complex):
            return ("complex", constant(value.real), constant(value.imag))
        if isinstance(value, (tuple, frozenset)):
            values = tuple(constant(item) for item in value)
            return (type(value).__name__, tuple(sorted(values, key=repr))
                    if isinstance(value, frozenset) else values)
        if value is Ellipsis:
            return ("ellipsis",)
        raise TypeError("execution code has an unsupported constant")
    return stable_digest((code.co_code.hex(), code.co_argcount, code.co_posonlyargcount,
                          code.co_kwonlyargcount, code.co_nlocals, code.co_stacksize,
                          code.co_flags, code.co_names, code.co_varnames, code.co_freevars,
                          code.co_cellvars, code.co_exceptiontable.hex(),
                          tuple(constant(value) for value in code.co_consts)))


@cache
def _composition_source_identity():
    return stable_digest(implementation_sources(
        "cubic_serve.runtime.quantized_graph",
        "cubic_serve.runtime.quantized_qualification",
    ))


def _bound_composition_plan(stage, *, kernels_only=False, workload=False):
    """Read retained launch identities, never addresses, weights or GPU values.

    Follow the bound call DAG, including closures and composed routed operators.
    Native/packed bindings own their complete compiled-artifact identities;
    fixed primitive signatures own their declared geometry/reduction metadata.
    An unrecognized external callable disables receipt reuse, not qualification.
    """
    # A captured body's selected launches are immutable. Cache on its owner,
    # without retaining a stage or any GPU storage in a global memo table.
    sealed = bool(getattr(stage, "_captured_once", False))
    if workload and not kernels_only:
        raise ValueError("portable workload identity requires kernel-only inspection")
    cache_attribute = ("_sealed_workload_digest" if workload else
                       "_sealed_kernel_digest" if kernels_only else "_sealed_launch_plan_digest")
    cached = getattr(stage, cache_attribute, None) if sealed else None
    if cached is not None:
        return cached

    from triton.compiler.compiler import CompiledKernel
    from cubic_serve.models.parallel_weights import RankLocalTensors
    from cubic_serve.models.quantized_checkpoint import CheckpointWeights, QuantizedMatrix
    from cubic_serve.ops.cuda.compiled_artifact import TensorBinding
    from cubic_serve.ops.cuda.linear.quantized import PackedLinearOperators
    from cubic_serve.runtime.cache import CacheArena
    from cubic_serve.runtime.quantized import QuantizedWeights

    # Program properties may construct short-lived tuples. Retain visited
    # objects until traversal ends so allocator ID reuse cannot skip an edge.
    seen, records, images = {}, [], []

    def image_identity(artifact):
        image = (artifact.backend_binary_hashes if artifact.backend == "torch-cublas"
                 else dict(artifact.stage_hashes)["cubin"])
        images.append(stable_digest(image))
        if workload:
            specialization = json.loads(artifact.specialization_json)
            # Vendor geometry declares the physical CUDA ordinal for live ABI
            # checks. It is not part of a portable invocation's work.
            if artifact.backend == "torch-cublas":
                specialization.pop("device", None)
            return image, artifact.grid, specialization
        return image

    def visit(value, *, constant=False):
        if value is None or type(value) in (bool, int, float, str):
            if not kernels_only and (constant or type(value) is bool):
                records.append((type(value).__name__, value))
            return
        if (value is None or isinstance(value, (str, int, float, bool, torch.Tensor,
                                               torch.dtype, torch.device, TensorBinding,
                                               QuantizedMatrix, QuantizedWeights,
                                               RankLocalTensors, CheckpointWeights,
                                               PackedLinearOperators, CacheArena))):
            # Source weights/factories and allocator bookkeeping are not
            # executable edges. Checkpoint/ownership and arena geometry are
            # separately attested in the enclosing key; follow their retained
            # bound operators, not millions of inventory/pool records.
            return
        identity = id(value), constant
        if identity in seen:
            return
        seen[identity] = value
        resource = getattr(value, "workload_resource_identity", None)
        if resource is not None and (workload or not kernels_only):
            # Storage changes price this invocation, not its CUDA image/device
            # namespace. The owner declares only immutable resource geometry.
            records.append(("workload-resource", resource))
        if isinstance(value, CompiledKernel):
            image = hashlib.blake2s(value.asm["cubin"]).hexdigest()
            images.append(stable_digest("b2s256:" + image))
            records.append(("cuda-image", image) if kernels_only else ("triton", value.hash, image))
            return
        projector = getattr(value, "project_artifacts", None)
        if callable(projector):
            records.append(("cuda-images" if kernels_only else type(value).__qualname__, tuple(
                (image_identity(artifact) if kernels_only else artifact.artifact_digest)
                for artifact in projector())))
            return
        projector = getattr(value, "project_artifact", None)
        if callable(projector):
            # Slot-backed primitives also own an actual compiled image. Their
            # geometry-only launch signature cannot identify that image.
            artifact = projector()
            records.append(("cuda-image" if kernels_only else type(value).__qualname__, (
                image_identity(artifact) if kernels_only else artifact.artifact_digest,)))
            return
        signature = getattr(value, "launch_signature", None)
        if isinstance(signature, str) and (not kernels_only or workload):
            records.append((type(value).__module__, type(value).__qualname__, signature))
        packed_signature = getattr(value, "signature", None)
        if isinstance(packed_signature, str) and not kernels_only:
            records.append((type(value).__qualname__, packed_signature,
                            getattr(value, "grid", None)))
        if isinstance(value, partial):
            visit(value.func)
            visit(value.args, constant=True)
            visit(value.keywords, constant=True)
        elif inspect.ismethod(value):
            if not kernels_only:
                source = inspect.getsourcefile(value.__func__)
                if source is None:
                    raise TypeError("bound callback has no implementation source")
                records.append((value.__func__.__qualname__, _composition_callback_source(source),
                                _composition_callback_code(value.__func__.__code__)))
            visit(value.__self__)
        elif inspect.isfunction(value):
            if not kernels_only:
                source = inspect.getsourcefile(value)
                if source is None:
                    raise TypeError("execution callback has no implementation source")
                records.append((value.__qualname__, _composition_callback_source(source),
                                _composition_callback_code(value.__code__)))
            visit(value.__defaults__, constant=True)
            visit(value.__kwdefaults__, constant=True)
            for cell in value.__closure__ or ():
                visit(cell.cell_contents, constant=True)
        elif inspect.isbuiltin(value):
            owner = getattr(value, "__self__", None)
            module = value.__module__ or type(owner).__module__
            if module not in ("builtins", "_operator") and not module.startswith("torch"):
                raise TypeError(f"external builtin lacks an artifact identity: {module}.{value.__qualname__}")
            # Python/Torch versions are in the key; tensor/storage ownership
            # remains an instance obligation checked by the live witnesses.
            if not kernels_only:
                records.append(("builtin", module, value.__qualname__))
            visit(owner)
        elif isinstance(value, dict):
            for name, item in value.items():
                if constant:
                    visit(name, constant=True)
                visit(item, constant=constant)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item, constant=constant)
        elif type(value).__module__.startswith("cubic_serve."):
            program = getattr(value, "program", None)
            if program is not None and program is not value:
                visit(program, constant=constant)
            elif is_dataclass(value):
                for field in fields(value):
                    visit(getattr(value, field.name), constant=constant)
            elif hasattr(value, "__dict__"):
                for item in vars(value).values():
                    visit(item)
        elif callable(value):
            raise TypeError(f"unbound external execution callable: {type(value).__module__}."
                            f"{getattr(value, '__qualname__', type(value).__qualname__)}")

    visit(getattr(stage, "program", stage))
    if not records:
        raise TypeError("captured body has no portable bound launch plan")
    digest = stable_digest(sorted(set(images)) if kernels_only and not workload else records)
    if sealed:
        try:
            setattr(stage, cache_attribute, digest)
        except (AttributeError, TypeError):
            pass
    return digest


def _qualification_program_identity(resource):
    """Unknown executable edges disable reuse, never qualify a new image."""
    try:
        return _bound_composition_plan(resource, kernels_only=True)
    except TypeError:
        return None


def _draft_composition_receipt_key(rounds, providers, targets, admission):
    """All ranks bind one exact implementation and selected topology plan."""
    from cuda.bindings import driver
    import triton

    from cubic_serve.ops.cuda.native import _check

    device = next(iter(targets.values())).body.hidden.device
    local = None
    try:
        properties = torch.cuda.get_device_properties(device)
        plans = tuple((kind, capacity, _bound_composition_plan(executor.body))
                      for kind, portfolio in (("target", targets), ("provider", providers))
                      for capacity, executor in sorted(portfolio.items()))
        arenas = tuple((kind, arena.spec.page_capacity, arena.spec.storage_page_capacity,
                        arena.spec.state_slot_capacity, arena.spec.scratch_capacity,
                        arena.spec.scratch_arena_ids,
                        tuple((role, name, tuple(t.shape), tuple(t.stride()), str(t.dtype))
                              for role, tensors in (
                                  ("page", {"key": arena.key, "value": arena.value,
                                            "written": arena.written, **arena.page_fields}),
                                  ("state", arena.states))
                              for name, t in sorted(tensors.items())))
                       for kind, arena in (("target", next(iter(targets.values())).body.arena),
                                           ("provider", next(iter(providers.values())).body.arena)))
        programs = tuple((key, _qualification_program_identity(round_))
                         for key, round_ in sorted(rounds.items()))
        if any(program is None for _, program in programs):
            raise TypeError("draft composition lacks a bound executable image")
        local = (plans, programs, arenas, str(properties.uuid), properties.multi_processor_count,
                 properties.L2_cache_size, torch.cuda.get_device_capability(device),
                 int(_check(driver.cuDriverGetVersion())[0]))
    except (AttributeError, TypeError, ValueError, OSError) as error:
        diagnostic_event(_LOGGER, 'quantized_qualification.mtp_composition_receipt_unavailable_s',
            error,
        )
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local)
    if any(plan is None for plan in gathered):
        return None
    identity = execution_identity(admission)
    key = stable_digest({
        "schema": "cubic.mtp-composition-key.v3",
        "implementation": _composition_source_identity(), "plans": gathered,
        "inventory": identity.inventory.inventory_digest,
        "semantic_plan": identity.parallel_plan.semantic_plan_digest,
        "topology": identity.sealed.topology_digest,
        "partition": identity.sealed.partition_digest,
        "operator_config": admission.operator_config.digest,
        "precision": (admission.operator_config.activation_dtype,
                      admission.operator_config.attention_dtype,
                      admission.operator_config.softmax_dtype, admission.mamba_ssm_cache_dtype),
        "max_context": admission.max_sequence_tokens,
        "reasoning_sampling": (
            tuple(getattr(admission, "reasoning_token_boundaries", ())),
            tuple(getattr(admission, "constraint_stop_token_ids", ())),
        ),
        "runtime": (sys.version, str(torch.__version__), str(torch.version.cuda), triton.__version__),
        "cases": tuple((capacity, round_.depth, round_.stride,
                         _draft_window_audit_owner_maps(round_.C))
                        for (capacity, _depth), round_ in sorted(rounds.items())),
    })
    keys = [None] * dist.get_world_size()
    dist.all_gather_object(keys, key)
    return key if len(set(keys)) == 1 else None

try:
    _CPU_MEMCMP = ctypes.CDLL(None).memcmp
    _CPU_MEMCMP.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t)
    _CPU_MEMCMP.restype = ctypes.c_int
except (AttributeError, OSError):
    _CPU_MEMCMP = None


def _has_typed_boundary(stage: Any) -> bool:
    return isinstance(stage, FlashNextDecodeStage) or getattr(stage, "typed_state_boundary", False)


def warm_reference_runtime(device: torch.device, width: int) -> None:
    """Materialize the oracle's BLAS context before sizing the retained KV pool.

    Native/Triton serving need not initialize BLAS. The independent oracle
    does, and its library context/workspace survives ordinary empty_cache().
    Use bounded synthetic operands, never expanded checkpoint weights. This
    is resource initialization, not numerical qualification.
    """
    from cubic_serve.executor.runner import ReferenceExecutionContext
    from cubic_serve.ops.eager.linear import linear

    with ReferenceExecutionContext(), torch.inference_mode():
        inputs = torch.ones((1, width), device=device, dtype=torch.bfloat16)
        weights = torch.ones((2, width), device=device, dtype=torch.bfloat16)
        linear(inputs, weights)
    torch.cuda.synchronize(device)


def execution_identity(admission: Any) -> Any:
    """Adapt only identity fields consumed by the shared execution witnesses.

    The legacy shared helpers call the topology/partition identity ``sealed``;
    this adapter is not a numerical seal and is never public admission evidence.
    """
    topology = asdict(admission.parallel_plan.topology)
    ownership = tuple(
        tuple(asdict(binding) for binding in rank) for rank in admission.parallel_plan.bindings
    )
    plan = {
        "checkpoint": admission.checkpoint.semantic_digest, "ownership": ownership,
    }
    weight_plan_digest = stable_digest(plan)
    cache_storage = getattr(admission, "cache_storage", None)
    if cache_storage is not None:
        plan["cache_storage"] = cache_storage.identity
    plan_digest = stable_digest(plan)
    definition_source = getattr(admission, "model_definition_source", None)
    if definition_source is not None:
        plan_digest = definition_source.program_identity(plan_digest, admission.model_program_recipe)
        weight_plan_digest = definition_source.program_identity(
            weight_plan_digest, admission.model_program_recipe)
    return SimpleNamespace(
        inventory=admission.checkpoint.inventory,
        parallel_plan=SimpleNamespace(
            semantic_plan_digest=plan_digest,
            weight_plan_digest=weight_plan_digest,
        ),
        sealed=SimpleNamespace(
            topology_digest=stable_digest(topology),
            partition_digest=stable_digest(ownership),
        ),
        max_sequence_tokens=admission.max_sequence_tokens,
        use_prefix_cache=admission.use_prefix_cache,
    )


def eager_snapshots(stage: Any, capacities: tuple[int, ...]) -> dict[int, Any]:
    """Retain same-body witnesses before capture, never from an audit callback."""
    stage.bucket_logical_ms = _logical_m_values(stage.captured_m, capacities)
    snapshots = {}
    for logical_m in _bounded_audit_logical_ms(stage.bucket_logical_ms):
        if _has_typed_boundary(stage):
            from cubic_serve.runtime.text_qualification import prepare_rows

            prepare_rows(stage, logical_m)
        else:
            _prepare_audit(stage, stage.resource, poison=0, logical_m=logical_m)
        dist.barrier()
        stage.launch()
        torch.cuda.synchronize(stage.hidden.device)
        dist.barrier()
        if _has_typed_boundary(stage):
            from cubic_serve.runtime.text_qualification import boundary_snapshot

            snapshots[logical_m] = boundary_snapshot(stage, logical_m)
        else:
            snapshots[logical_m] = _active_snapshot(stage, logical_m)
    return snapshots


def audit_capture(stage: Any, identity: Any, snapshots: dict[int, Any]) -> str:
    if _has_typed_boundary(stage):
        from cubic_serve.runtime.text_qualification import (
            boundary_snapshot,
            compare_snapshot,
            prepare_rows,
        )

        typed_witnesses = []
        for logical_m, expected in snapshots.items():
            for poison in (False, True):
                prepare_rows(stage, logical_m, poison=poison)
                stage.stream.wait_stream(torch.cuda.current_stream(stage.arena.device))
                stage.replay()
                stage.synchronize()
                dist.barrier()
                witness = compare_snapshot(expected, boundary_snapshot(stage, logical_m))
                typed_witnesses.append((logical_m, poison, witness))
                if getattr(stage, "_state_graph", None) is not None:
                    prepare_rows(stage, logical_m, poison=poison)
                    stage.logits.fill_(float("nan"))
                    untouched_logits = stage.logits.clone()
                    stage.stream.wait_stream(torch.cuda.current_stream(stage.arena.device))
                    stage.replay_state()
                    stage.synchronize()
                    dist.barrier()
                    if not torch.equal(stage.logits.view(torch.uint8),
                                       untouched_logits.view(torch.uint8)):
                        raise ValueError("draft feedback replay changed protected logits")
                    actual = boundary_snapshot(stage, logical_m)
                    actual.pop("logits")
                    state_expected = {k: v for k, v in expected.items() if k != "logits"}
                    witness = compare_snapshot(state_expected, actual)
                    typed_witnesses.append((logical_m, poison, "feedback", witness))
        return stable_digest({
            "schema": "cubic.typed-state.boundary-capture-identity.v1",
            "rank": stage.rank,
            "capacity": stage.captured_m,
            "topology": identity.sealed.topology_digest,
            "partition": identity.sealed.partition_digest,
            "witnesses": typed_witnesses,
        })
    witnesses = []
    variants = (False, True) if stage._overlap_resources is not None else (False,)
    for overlap in variants:
        stage.select_decode_overlap(overlap)
        witnesses.append(
            _cold_start_graph_variant_audit(
                stage,
                stage.resource,
                stage,
                identity,
                snapshots,
            )
        )
    stage.select_decode_overlap(False)
    return stable_digest(witnesses)


def audit_state(executors: dict[int, Any], identity: Any) -> str:
    """Exercise production migration, continuation and immutable prefix forks."""
    if _has_typed_boundary(executors[1].body):
        from cubic_serve.runtime.text_qualification import audit_owner_migration, audit_prefix_forks

        migration = audit_owner_migration(
            executors, max_sequence_tokens=identity.max_sequence_tokens
        )
        if not identity.use_prefix_cache:
            return migration
        prefix = audit_prefix_forks(executors, max_sequence_tokens=identity.max_sequence_tokens)
        return stable_digest({"migration": migration, "prefix": prefix})
    migration = _run_bucket_migration_audit(executors, identity)
    if not identity.use_prefix_cache:
        return stable_digest({"migration": migration, "prefix": "disabled"})
    stage = executors[1].body
    # These are cold witness rows, released before readiness. MTP0 serving
    # does not retain or maintain a target-hidden cache.
    hidden = torch.zeros(
        (4, stage.hidden.shape[-1]), device=stage.hidden.device, dtype=stage.hidden.dtype
    )
    with _startup_check_phase(stage.rank, "prefix cache fork"):
        prefix = _cold_start_prefix_cache_audit(executors, {}, hidden, identity)
    del hidden
    return stable_digest({"migration": migration, "prefix": prefix})


def _draft_fixture_state_map(round_, n, pages_per_owner):
    """Cold diagnostic ownership: low pages and disjoint high state bundles.

    The complete bucket reserves its owner prefix even for a padded case, so
    every smaller case is covered by the same immutable original snapshot.
    This is not a live allocator and must never assign serving ownership.
    """
    storage = getattr(round_.target.body.arena, "shared_storage", None)
    if storage is None:
        return tuple(range(n)), None
    other = round_.align.body.arena.shared_storage
    if (storage is not other or n > round_.C
            or storage.allocator.cold_owner_capacity(
                state_slots=round_.depth + 1, pages=pages_per_owner) < round_.C):
        raise ValueError("draft fixture exceeds disjoint shared cache resources")
    owners = tuple(storage.state_capacity - 1 - i for i in range(n))
    checkpoints = tuple(storage.state_capacity - 1 - round_.C - i
                        for i in range(n * round_.depth))
    return owners, checkpoints


def _draft_fixture_protected_pages(arena, active_pages, state_rows, *, device):
    """Potential page views overlapping owned state have no page identity."""
    storage = getattr(arena, "shared_storage", None)
    excluded = set()
    if storage is not None:
        pool = storage.allocator
        for row in state_rows.tolist():
            start = pool.state_offset_bytes + row * pool.state_bytes
            end = start + pool.state_bytes
            excluded.update(range(start // pool.page_bytes,
                                  (end + pool.page_bytes - 1) // pool.page_bytes))
    return torch.tensor([p for p in range(active_pages, arena.spec.page_capacity)
                         if p not in excluded], dtype=torch.int64, device=device)


def _draft_window_audit_rows(round_, pending, owner_count, pages_per_owner, *,
                             owner_order=None, sampling=None):
    """Retain each review prefix before committing its final durable owner."""
    from dataclasses import replace

    from cubic_serve.generation.sampler import GREEDY_SAMPLING
    from cubic_serve.runtime.continuous_batch import BatchRow

    if owner_order is None:
        owner_order = tuple(range(owner_count))
    if (type(owner_order) is not tuple or len(owner_order) != owner_count
            or set(owner_order) != set(range(owner_count))):
        raise ValueError("draft fixture row map must permute its declared owners")
    owner_ids, checkpoints = _draft_fixture_state_map(round_, owner_count, pages_per_owner)
    values = []
    for physical_owner, owner in enumerate(owner_order):
        state_owner = owner_ids[owner]
        carried = pending[state_owner]
        position, root, ordinal = ((0, (101 + owner) % round_.target.vocab_size, 0) if carried is None else
                                  (carried[1], carried[2], carried[4]))
        table = tuple(range(owner * pages_per_owner, (owner + 1) * pages_per_owner))
        for j in range(round_.stride):
            p = position + j
            physical = physical_owner * round_.stride + j
            values.append(BatchRow(
                request_id=f"audit:draft-window:{owner}",
                token_id=(root if j == 0 else carried[6][j - 1] if carried is not None else 0),
                position=p, state_source_slot=state_owner,
                state_destination_slot=(state_owner if j == round_.depth else
                    round_.target._scratch_states[physical] if checkpoints is None else
                    checkpoints[physical_owner * round_.depth + j]),
                page_source=table[p // round_.target._block_size],
                page_destination=table[p // round_.target._block_size], page_table=table,
                segment_length=round_.stride if j == 0 else 0, sample_ordinal=ordinal + j,
                mtp_position=p, mtp_page_destination=table[p // round_.target._block_size],
                mtp_page_table=table, is_decode=True,
                sampling=(GREEDY_SAMPLING if sampling is None else
                          replace(sampling, seed=sampling.seed + owner)),
            ))
    return tuple(values)


def _initialize_draft_window_fixture(fields, target, provider, target_states, owners, pool, *,
                                     provider_states=None):
    """Initialize owned state through its declared schema, including rollback."""
    for _name, tensor, indices in fields:
        tensor.index_fill_(0, indices, 0)
    target.reset_state_rows(target_states)
    provider.reset_state_rows(owners if provider_states is None else provider_states)
    for name, tensor in pool.backup.items():
        tensor.index_copy_(0, owners, provider.states[name].index_select(0, owners))


def _prepare_cold_draft_fixture(round_, hidden, admission):
    """Declare unowned offline fixture inputs before preserving their image.

    Cases already initialize these exact mathematical inputs. Old warmup bytes
    have no request identity here; they need not become a disk-backed original.
    All existing seed, state, ownership and protected-write comparisons remain.
    """
    error = None
    try:
        if not (getattr(admission, "startup_qualification", False)
                or getattr(admission, "startup_numerical_audit", False)):
            raise ValueError("cold fixture requires explicit offline qualification")
        if any(value is not None for value in round_.pool.metadata):
            raise ValueError("cold fixture has pending proposal owners")
        for name in ("_resident_window", "_candidate_window",
                     "_budget_resident_window", "_budget_candidate_window"):
            if getattr(getattr(round_, name, None), "inflight", False):
                raise ValueError("cold fixture has an inflight resident window")
        arenas = {id(body.arena): body.arena
                  for body in (round_.target.body, round_.align.body)}
        for arena in arenas.values():
            arena._require_healthy()
            if (arena._scratch_active or any(arena._page_pool._holds)
                    or any(any(pool._holds) for pool in arena._state_pools.values())):
                raise ValueError("cold fixture has live cache or scratch owners")
        pages = (4 * round_.stride + round_.target._block_size - 1) // round_.target._block_size
        if any(round_.C * pages > arena.spec.page_capacity for arena in arenas.values()):
            raise ValueError("cold fixture exceeds admitted pages")
    except Exception as exc:
        error = type(exc).__name__ + ": " + str(exc)
    errors = [error]
    if dist.is_initialized():
        errors = [None] * dist.get_world_size()
        dist.all_gather_object(errors, error)
    if any(errors):
        raise RuntimeError("cold fixture ownership prerequisite failed: " + repr(errors))
    device = round_.target.body.hidden.device
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    fields, owners, target_states, provider_states = _draft_window_fixture_fields(
        round_, hidden, round_.C, pages)
    _initialize_draft_window_fixture(
        fields, round_.target.body.arena, round_.align.body.arena,
        target_states, owners, round_.pool, provider_states=provider_states)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _draft_window_audit_owner_maps(capacity, owner_domain=None):
    """Bounded full/padded domains and a physical permutation per domain."""
    if type(capacity) is not int or capacity <= 0:
        raise ValueError("draft fixture requires a positive owner capacity")
    domain = (tuple(range(capacity // 2 + 1, capacity + 1))
              if owner_domain is None else owner_domain)
    if not domain or domain[-1] != capacity or any(not 0 < n <= capacity for n in domain):
        raise ValueError("draft fixture requires its complete served owner domain")
    counts = tuple(dict.fromkeys((1, min(2, capacity), *_bounded_audit_logical_ms(domain))))
    return tuple(dict.fromkeys(order for count in counts
                 for order in (tuple(range(count)), tuple(reversed(range(count))))))



def _draft_fixture_row_ranges(indices, capacity):
    """Coalesce ascending rows, preserving duplicate and exceptional indices."""
    rows = indices.tolist()
    first = 0
    while first < len(rows):
        index = rows[first]
        stop = first + 1
        if type(index) is int and 0 <= index < capacity:
            while (stop < len(rows) and rows[stop] == index + stop - first
                   and rows[stop] < capacity):
                stop += 1
        yield first, stop, index
        first = stop


def _draft_fixture_field_ranges(fields, destination_buffers=()):
    # Copy helpers never change their index metadata. Alias destinations use
    # the original per-field reads; all other spans are local to this call.
    fields = tuple(fields)
    written_storage = {(tensor.device, tensor.untyped_storage().data_ptr())
                       for _, tensor, _ in fields}
    written_storage.update((tensor.device, tensor.untyped_storage().data_ptr())
                           for tensor in destination_buffers)
    cache = {}
    for name, tensor, indices in fields:
        key = id(indices), tensor.shape[0]
        storage = indices.device, indices.untyped_storage().data_ptr()
        if storage in written_storage:
            spans = tuple(_draft_fixture_row_ranges(indices, tensor.shape[0]))
        else:
            if key not in cache:
                cache[key] = (indices, tuple(_draft_fixture_row_ranges(indices, tensor.shape[0])))
            spans = cache[key][1]
        yield name, tensor, indices, spans



def _snapshot_draft_fixture_fields(
    fields, *, buffers=None, owner_count=1, owner_capacity=1, field_ready=None,
):
    """Retain completed witnesses on CPU, copying contiguous rows together."""
    if field_ready is not None and buffers is not None:
        stores = [value.untyped_storage().data_ptr() for value in buffers.values() if value.numel()]
        if len(stores) != len(set(stores)):
            raise ValueError("comparison snapshots share writable field storage")
    values = {}
    for name, tensor, indices, spans in _draft_fixture_field_ranges(
            fields, () if buffers is None else tuple(buffers.values())):
        shape = (indices.numel(), *tensor.shape[1:])
        if buffers is None:
            snapshot = torch.empty(shape, dtype=tensor.dtype)
        else:
            if (owner_count <= 0 or owner_capacity < owner_count
                    or indices.numel() % owner_count):
                raise ValueError("draft snapshot rows differ from their owner capacity")
            capacity_shape = (
                indices.numel() // owner_count * owner_capacity, *tensor.shape[1:])
            buffer = buffers.get(name)
            if buffer is None:
                # Four exclusive case roles retain complete immutable witnesses.
                # Their CPU storage is reused only after the case has restored
                # its original state. Every exposed byte is copied below.
                size = torch.Size(capacity_shape).numel() * tensor.element_size()
                if tensor.is_cuda:
                    storage = torch.empty(0, dtype=torch.uint8, pin_memory=True).untyped_storage()
                    storage.resize_(size)
                else:
                    storage = torch.UntypedStorage(size, device="cpu")
                buffer = torch.empty(0, dtype=tensor.dtype).set_(storage, 0, capacity_shape)
                buffers[name] = buffer
            elif (buffer.shape != capacity_shape or buffer.dtype != tensor.dtype
                    or buffer.device.type != "cpu" or not buffer.is_contiguous()):
                raise ValueError("draft snapshot buffer schema changed within its bucket")
            snapshot = buffer[:indices.numel()]
        asynchronous = tensor.is_cuda and snapshot.is_pinned()
        for first, stop, index in spans:
            if stop == first + 1:
                snapshot[first].copy_(tensor[index], non_blocking=asynchronous)
            else:
                snapshot[first:stop].copy_(tensor[index:index + stop - first],
                                          non_blocking=asynchronous)
        if asynchronous:
            # One field is immutable and complete before callbacks or later
            # fields can read it, including index metadata aliasing its buffer.
            torch.cuda.current_stream(tensor.device).synchronize()
        values[name] = snapshot
        if field_ready is not None:
            # Blocking field copies have completed; no model callback runs here.
            field_ready(name, snapshot)
    return values



def _restore_draft_fixture_fields(fields, values):
    for name, tensor, _indices, spans in _draft_fixture_field_ranges(fields):
        asynchronous = tensor.is_cuda and values[name].is_pinned()
        for first, stop, index in spans:
            if stop == first + 1:
                tensor[index].copy_(values[name][first], non_blocking=asynchronous)
            else:
                tensor[index:index + stop - first].copy_(values[name][first:stop],
                                                      non_blocking=asynchronous)
        if asynchronous:
            # Preserve the original cross-field index dependencies and make
            # restored state visible to executors using another CUDA stream.
            torch.cuda.current_stream(tensor.device).synchronize()



def _equal_draft_fixture_bytes(actual, expected):
    """Compare immutable storage bits with native CPU bytes or exact words."""
    if (actual.dtype != torch.uint8 or expected.dtype != torch.uint8
            or actual.shape != expected.shape):
        return False
    if (_CPU_MEMCMP is not None and actual.device.type == expected.device.type == "cpu"
            and actual.is_contiguous() and expected.is_contiguous()):
        # Immutable CPU witnesses: compare all storage bits without floating math.
        return actual.numel() == 0 or _CPU_MEMCMP(
            actual.data_ptr(), expected.data_ptr(), actual.numel()) == 0
    if (not actual.is_contiguous() or not expected.is_contiguous()
            or actual.storage_offset() % 8 or expected.storage_offset() % 8):
        return torch.equal(actual, expected)
    left, right = actual.reshape(-1), expected.reshape(-1)
    stop = left.numel() // 8 * 8
    if stop and not torch.equal(left[:stop].view(torch.int64), right[:stop].view(torch.int64)):
        return False
    return stop == left.numel() or torch.equal(left[stop:], right[stop:])


def _compare_draft_fixture_fields(actual, expected, *, pool=None, receipts=None):
    """Finish all immutable CPU reads before reporting the first field failure."""
    def unchanged(name):
        return _equal_draft_fixture_bytes(
            actual[name].contiguous().view(torch.uint8),
            expected[name].contiguous().view(torch.uint8))

    checks = [] if receipts is None else receipts
    failure = None
    try:
        if receipts is None:
            for name in expected:
                checks.append((name, unchanged(name) if pool is None
                               else pool.submit(unchanged, name)))
        for name, result in checks:
            same = result if pool is None else result.result()
            if not same and failure is None:
                failure = name
    finally:
        if pool is not None:
            wait([result for _, result in checks])
    if failure is not None:
        raise ValueError(f"draft composition changed rounded boundary {failure}")


def _compare_draft_candidate_scores(actual, expected, token_ids):
    # Candidate padding is the semantic pair (NaN, -1); IEEE NaN payloads
    # carry no probability or owner identity. Every finite score bit remains
    # exact. This allowance never applies to model outputs or durable state.
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError("draft candidate witness ABI differs")
    sentinel = token_ids == -1
    if (not torch.equal(torch.isnan(actual), sentinel)
            or not torch.equal(torch.isnan(expected), sentinel)
            or not torch.equal(actual[~sentinel].contiguous().view(torch.uint8),
                               expected[~sentinel].contiguous().view(torch.uint8))):
        raise ValueError("draft composition changed finite candidates or sentinel validity")


def _draft_window_fixture_fields(round_, hidden, n, pages_per_owner):
    target, provider = round_.target.body.arena, round_.align.body.arena
    device, stride = round_.target.body.hidden.device, round_.stride
    owner_ids, checkpoints = _draft_fixture_state_map(round_, n, pages_per_owner)
    owners = torch.tensor(owner_ids, device=device, dtype=torch.int64)
    if checkpoints is None:
        target_scratch = round_.target._scratch_states[:n * stride]
        provider_scratch = round_.align._draft_binding.executor._scratch_states[:n * stride]
    else:
        target_scratch = provider_scratch = checkpoints
    target_states = torch.cat((owners, torch.tensor(
        target_scratch, device=device, dtype=torch.int64)))
    provider_states = torch.cat((owners, torch.tensor(
        provider_scratch, device=device, dtype=torch.int64)))
    page_ids = torch.arange(n * pages_per_owner, device=device)
    fields = []
    for name, arena in (("target", target), ("provider", provider)):
        for key, tensor in {"key": arena.key, "value": arena.value,
                            "written": arena.written, **arena.page_fields}.items():
            fields.append((name + "." + key, tensor, page_ids))
        for key, tensor in arena.states.items():
            fields.append((name + "." + key, tensor,
                           target_states if name == "target" else provider_states))
    # A segment's final target row writes its durable owner directly. Only
    # earlier rows can be selected by rejection repair (owner_copy MODE=2,
    # accepted < depth). The unused final scratch slot has no request identity;
    # a different physical bucket may overwrite it as private padding. Retain
    # exact feedback witnesses for every durable owner and selectable checkpoint.
    feedback_states = torch.cat((
        owners, (target_states[n:] if checkpoints is not None else
                 target_states[n:].reshape(n, stride)[:, :round_.depth].reshape(-1)),
    ))
    fields.append(("feedback", hidden, feedback_states))
    for key, tensor in round_.pool.backup.items():
        fields.append(("backup." + key, tensor, owners))

    return fields, owners, target_states, provider_states


class _DraftOriginalSnapshot:
    """Reuse role-zero storage, refreshing every row not restored by the last case."""

    def __init__(self, fields, *, buffers, owner_count, owner_capacity):
        self.fields = tuple(fields)
        self.values = _snapshot_draft_fixture_fields(
            self.fields, buffers=buffers, owner_count=owner_count,
            owner_capacity=owner_capacity)
        lookup, valid = {}, {}
        self.bindings, self.valid, self.prepared = {}, {}, {}
        for name, tensor, indices in self.fields:
            key = id(indices), tensor.shape[0]
            if key not in lookup:
                rows = tuple(row if row >= 0 else tensor.shape[0] + row
                             for row in indices.tolist())
                if any(type(row) is not int or not 0 <= row < tensor.shape[0] for row in rows):
                    raise ValueError("original fixture row is outside its physical storage")
                lookup[key] = {row: offset for offset, row in enumerate(rows)}
                valid[key] = frozenset(rows)
            self.bindings[name] = tensor, lookup[key]
            self.valid[name] = valid[key]

    @staticmethod
    def _spans(pairs):
        spans = []
        for source, destination in pairs:
            if (spans and source == spans[-1][0] + spans[-1][2]
                    and destination == spans[-1][1] + spans[-1][2]):
                begin, target, count = spans[-1]
                spans[-1] = begin, target, count + 1
            else:
                spans.append((source, destination, 1))
        return tuple(spans)

    def prepare(self, fields):
        if self.prepared:
            raise ValueError("original fixture receipt is still owned by a case")
        fields = tuple(fields)
        writable = {(tensor.device, tensor.untyped_storage().data_ptr())
                    for _, tensor, _ in fields}
        plans = {}
        for name, tensor, indices in fields:
            original_tensor, lookup = self.bindings[name]
            if tensor is not original_tensor:
                raise ValueError("original fixture storage binding changed")
            if (indices.device, indices.untyped_storage().data_ptr()) in writable:
                raise ValueError("original fixture indices alias writable storage")
            key = id(indices), id(lookup), id(self.valid[name]), tensor.shape[0]
            if key not in plans:
                rows = tuple(row if row >= 0 else tensor.shape[0] + row
                             for row in indices.tolist())
                if any(type(row) is not int or row not in lookup for row in rows):
                    raise ValueError("original fixture receipt does not cover its restore")
                pairs = tuple((lookup[row], row) for row in rows)
                missing = tuple(pair for pair, row in zip(pairs, rows, strict=True)
                                if row not in self.valid[name])
                plans[key] = (indices, self._spans(pairs), self._spans(missing), frozenset(rows))
            _, spans, missing, valid_rows = plans[key]
            for source, destination, count in missing:
                self.values[name][source:source + count].copy_(
                    tensor[destination:destination + count])
            self.prepared[name] = tensor, indices, spans, valid_rows

    def restore(self, fields):
        for name, tensor, indices in fields:
            original_tensor, original_indices, spans, valid_rows = self.prepared[name]
            if tensor is not original_tensor or indices is not original_indices:
                raise ValueError("original fixture restore changed its frozen row binding")
            for source, destination, count in spans:
                tensor[destination:destination + count].copy_(
                    self.values[name][source:source + count])
            # Unselected private padding may legally change during a case. Only
            # rows just restored are reusable; all others must be read again.
            self.valid[name] = valid_rows
        self.prepared.clear()



def _draft_witness_budget(local_ranks):
    """Bound offline witness memory by current host and cgroup headroom."""
    available = next(int(line.split()[1]) * 1024 for line in
                     Path("/proc/meminfo").read_text().splitlines()
                     if line.startswith("MemAvailable:"))
    # The cgroup mount may expose either the namespace root or the host path.
    roots = {Path("/sys/fs/cgroup")}
    for line in Path("/proc/self/cgroup").read_text().splitlines():
        if line.startswith("0::"):
            root = Path("/sys/fs/cgroup") / line[3:].lstrip("/")
            roots.update((root, *root.parents))
    for root in roots:
        if not root.is_relative_to("/sys/fs/cgroup"):
            continue
        try:
            limit = (root / "memory.max").read_text().strip()
            if limit != "max":
                used = int((root / "memory.current").read_text())
                available = min(available, max(0, int(limit) - used))
        except FileNotFoundError:
            continue
    budget = min(64 << 20, available // (8 * max(1, local_ranks)))
    if budget < 4096:
        raise MemoryError("insufficient host headroom for bounded audit witnesses")
    return budget


def _draft_tensor_chunks(tensor, byte_limit):
    """Canonical contiguous-order views, including a row larger than the budget."""
    if byte_limit < tensor.element_size():
        raise ValueError("audit chunk cannot hold one storage element")
    if not tensor.numel():
        return
    if tensor.numel() * tensor.element_size() <= byte_limit:
        yield tensor
        return
    axis = next(axis for axis, length in enumerate(tensor.shape) if length > 1)
    row_bytes = tensor.numel() // tensor.shape[axis] * tensor.element_size()
    rows = max(1, byte_limit // row_bytes)
    for start in range(0, tensor.shape[axis], rows):
        yield from _draft_tensor_chunks(
            tensor.narrow(axis, start, min(rows, tensor.shape[axis] - start)), byte_limit)


def _draft_field_chunks(tensor, spans, byte_limit):
    for first, stop, index in spans:
        selected = (tensor[index:index + stop - first] if stop > first + 1
                    else tensor[index].unsqueeze(0))
        yield from _draft_tensor_chunks(selected, byte_limit)


def _storage_is_zero(tensor, byte_limit=8 << 20):
    """Prove every storage bit zero with bounded strided materialization."""
    unchanged = torch.ones((), dtype=torch.bool, device=tensor.device)
    for chunk in _draft_tensor_chunks(tensor, byte_limit):
        flat = chunk.contiguous().reshape(-1)
        raw = flat.as_strided((flat.numel(),), (1,)).view(torch.uint8)
        unchanged.logical_and_(raw.any() == 0)
    return bool(unchanged)


def _draft_chunk_bytes(chunk):
    # At most two chunk-sized CPU allocations, never a full field or role.
    host = chunk.detach().to(device="cpu").contiguous()
    # Singleton dimensions may be contiguous while retaining a non-unit stride.
    flat = host.reshape(-1).as_strided((host.numel(),), (1,))
    return memoryview(flat.view(torch.uint8).numpy())


def _draft_fixture_identities(fields, byte_budget):
    """Exactly tensor_digest's metadata/bytes, with bounded field-level overlap."""
    fields = tuple(fields)
    if not fields:
        return {}
    if len({name for name, _, _ in fields}) != len(fields):
        raise ValueError("duplicate draft witness field identity")
    from cubic_serve.runtime.performance import _grammar_mask_worker_budget

    workers = min(len(fields), _grammar_mask_worker_budget())
    byte_limit = byte_budget // (8 * workers)
    # Keep comparable large fields in flight together. Interleaved tiny fields
    # otherwise leave hash workers idle behind one large pending update.
    pending = iter(sorted(
        _draft_fixture_field_ranges(fields),
        key=lambda item: (item[2].numel() * item[1].shape[1:].numel()
                          * item[1].element_size()),
        reverse=True,
    ))
    active, identities = [], {}
    pool = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None

    def next_field():
        try:
            name, tensor, indices, spans = next(pending)
        except StopIteration:
            return
        metadata = {"shape": (indices.numel(), *tensor.shape[1:]),
                    "dtype": str(tensor.dtype), "layout": str(tensor.layout)}
        digest = hashlib.blake2s()
        digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode())
        digest.update(b"\0")
        active.append([name, digest, iter(_draft_field_chunks(tensor, spans, byte_limit)), None])

    try:
        for _ in range(workers):
            next_field()
        while active:
            for record in tuple(active):
                name, digest, chunks, future = record
                # One update per field in flight; exact canonical byte order.
                if future is not None:
                    future.result()
                try:
                    chunk = next(chunks)
                except StopIteration:
                    identities[name] = "b2s256:" + digest.hexdigest()
                    active.remove(record)
                    next_field()
                    continue
                # Only this main thread touches CUDA. Workers read immutable
                # completed CPU bytes; at most workers chunks remain in flight.
                payload = _draft_chunk_bytes(chunk)
                if pool is None:
                    digest.update(payload)
                else:
                    record[3] = pool.submit(digest.update, payload)
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
    return {name: identities[name] for name, _, _ in fields}


def _compare_draft_fixture_identities(actual, expected):
    if actual.keys() != expected.keys():
        raise ValueError("draft composition changed state field schema")
    for name in expected:
        if actual[name] != expected[name]:
            raise ValueError(f"draft composition changed rounded boundary {name}")


def _draft_reference_headroom(device):
    free, total = torch.cuda.mem_get_info(device)
    limit = int(total * torch.cuda.get_per_process_memory_fraction(device))
    return max(0, min(free, limit - torch.cuda.memory_reserved(device)))


def _draft_reference_device_budget(fields, byte_budget):
    """Admit two reference roles after reserving bounded comparison workspace.

    This is optional offline witness storage, never serving admission. Uncached
    fields keep the existing bounded canonical byte/hash proof. Eight complete
    host witness budgets remain free; comparisons need at most two one-eighth
    budget contiguous chunks, plus the already bound graph/guard resources.
    """
    devices = {tensor.device for _, tensor, _ in fields}
    if len(devices) != 1 or next(iter(devices)).type != "cuda":
        return 0
    free = _draft_reference_headroom(next(iter(devices)))
    return max(0, free - 8 * byte_budget) // 2


def _retire_draft_fixture_references(reference_cache):
    """Release completed private witnesses before pricing the next reference.

    A CPU-only fallback can follow earlier device images whose unused allocator
    blocks are still reserved. Retire every nonempty reference cache, including
    that fallback, so resource admission cannot remain stuck at zero headroom.
    This frees no live model, graph, KV, state or witness storage.
    """
    occupied = bool(reference_cache)
    reference_cache.clear()
    if occupied:
        torch.cuda.empty_cache()


class _DraftFixtureReference:
    """Canonical identities plus optional private immutable device byte images.

    Each actual candidate still proves every field and storage bit. Repeated
    reference D2H copies/hashes are replaced by exact comparison against owned
    images; a digest alone never supplies a missing comparison. No image retains
    live model storage, and unmatched fields retain the original bounded path.
    """

    def __init__(self, fields, *, byte_budget, device_byte_budget):
        if type(device_byte_budget) is not int or device_byte_budget < 0:
            raise ValueError("draft reference requires a nonnegative device byte budget")
        fields = tuple(fields)
        if len({name for name, _, _ in fields}) != len(fields):
            raise ValueError("duplicate draft witness field identity")
        self.images = {}
        self.bytes = 0
        self.byte_budget = byte_budget
        self.bindings = self._bindings(fields)
        ranges = list(_draft_fixture_field_ranges(fields))
        # Whole fields avoid a per-byte cache index. Largest-first retains the
        # actual dominant state payload within the measured resource envelope.
        for name, tensor, indices, spans in sorted(
                ranges, key=lambda item: item[2].numel() * item[1].shape[1:].numel()
                * item[1].element_size(), reverse=True):
            size = indices.numel() * tensor.shape[1:].numel() * tensor.element_size()
            if not tensor.is_cuda or self.bytes + size > device_byte_budget:
                continue
            if _draft_reference_headroom(tensor.device) < size + 8 * byte_budget:
                continue
            # Inference-mode tensors have no mutation counter. Own normal
            # no-grad images so corruption is detectable under either caller.
            with torch.inference_mode(False), torch.no_grad():
                # One allocation per field avoids allocator segment waste from
                # thousands of separately allocated comparison-sized chunks.
                storage = torch.empty(size // tensor.element_size(), dtype=tensor.dtype,
                                      device=tensor.device)
                images, offset = [], 0
                for chunk in _draft_field_chunks(tensor, spans, byte_budget // 8):
                    image = storage[offset:offset + chunk.numel()].reshape(chunk.shape)
                    image.copy_(chunk)
                    images.append(image)
                    offset += chunk.numel()
                if offset != storage.numel():
                    raise ValueError("draft reference storage extent changed")
            self.images[name] = tuple((image, image._version) for image in images)
            self.bytes += size
        self._identities = _draft_fixture_identities(
            [field for field in fields if field[0] not in self.images], byte_budget)

    @staticmethod
    def _bindings(fields):
        rows, bindings = {}, {}
        for name, tensor, indices in fields:
            if id(indices) not in rows:
                rows[id(indices)] = tuple(indices.detach().cpu().tolist())
            bindings[name] = (tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()),
                              tensor.dtype, tensor.device, rows[id(indices)])
        return bindings

    def __getitem__(self, name):
        if name not in self._identities:
            _, shape, _, dtype, _, rows = self.bindings[name]
            digest = hashlib.blake2s()
            metadata = {"shape": (len(rows), *shape[1:]), "dtype": str(dtype),
                        "layout": str(torch.strided)}
            digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode())
            digest.update(b"\0")
            for image, version in self.images[name]:
                if image._version != version:
                    raise ValueError("immutable draft reference image was modified")
                digest.update(_draft_chunk_bytes(image))
            self._identities[name] = "b2s256:" + digest.hexdigest()
        return self._identities[name]

    @property
    def identities(self):
        # Canonical hashes are available to evidence consumers, but comparison
        # of owned images needs no reference D2H copy or hash at all.
        return {name: self[name] for name in self.bindings}

    @staticmethod
    def _bytes(value):
        flat = value.contiguous().reshape(-1)
        return flat.as_strided((flat.numel(),), (1,)).view(torch.uint8)

    def compare(self, fields):
        fields = tuple(fields)
        if (len(fields) != len(self.bindings)
                or {name for name, _, _ in fields} != self.bindings.keys()):
            raise ValueError("draft composition changed state field schema")
        bindings = self._bindings(fields)
        uncached = []
        for name, tensor, indices, spans in _draft_fixture_field_ranges(fields):
            if bindings[name] != self.bindings[name]:
                raise ValueError("draft reference field binding changed")
            images = self.images.get(name)
            if images is None:
                uncached.append((name, tensor, indices))
                continue
            chunks = _draft_field_chunks(tensor, spans, self.byte_budget // 8)
            for chunk, (image, version) in zip(chunks, images, strict=True):
                if image._version != version:
                    raise ValueError("immutable draft reference image was modified")
                if not _equal_draft_fixture_bytes(
                        self._bytes(chunk), self._bytes(image)):
                    raise ValueError(f"draft composition changed rounded boundary {name}")
        actual = _draft_fixture_identities(uncached, self.byte_budget)
        _compare_draft_fixture_identities(
            actual, {name: self._identities[name] for name, _, _ in uncached})


class _DraftBucketSnapshot:
    """One exact image; proved zero runs need neither host copies nor raw hashes."""

    def __init__(self, fields, *, byte_budget, local_ranks=1, directory=None):
        # Freeze row identity before any case can mutate its inputs. Tensor
        # payloads remain live; only small, independent index metadata is kept.
        indices_by_identity, frozen = {}, []
        for name, tensor, indices in fields:
            if id(indices) not in indices_by_identity:
                indices_by_identity[id(indices)] = indices.detach().to(device="cpu").clone()
            frozen.append((name, tensor, indices_by_identity[id(indices)]))
        self.fields = tuple(frozen)
        self.bindings = tuple((tuple(t.shape), tuple(t.stride()), t.data_ptr(), t.dtype, t.device)
                              for _, t, _ in self.fields)
        self.byte_budget = byte_budget
        self.chunk_bytes = byte_budget // 8
        self.total_bytes = sum(indices.numel() * tensor[0].numel() * tensor.element_size()
                               for _, tensor, indices in self.fields if indices.numel())
        # Keep the representation metadata inside the existing byte budget.
        # Adjacent zero chunks coalesce; excess runs use the ordinary byte image.
        self.zero_runs = []
        self.zero_bytes = 0
        zero_run_limit = max(1, byte_budget // 2048)
        self.file, self.memory, self.closed = None, None, False
        if self.total_bytes <= byte_budget // 2:
            self.memory = torch.empty(self.total_bytes, dtype=torch.uint8)
        else:
            if directory is None:
                from cubic_serve import cache_root
                directory = cache_root() / "qualification-tmp"
            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=True)
            # A tmpfs spill would merely move the same memory into page cache.
            mounts = []
            for line in Path("/proc/self/mountinfo").read_text().splitlines():
                before, after = line.split(" - ", 1)
                mount = Path(before.split()[4].replace("\\040", " "))
                if directory.resolve().is_relative_to(mount):
                    mounts.append((len(mount.parts), after.split()[0]))
            if not mounts or max(mounts)[1] in ("tmpfs", "ramfs"):
                raise MemoryError("audit snapshot requires disk-backed temporary storage")
            space = os.statvfs(directory)
            if self.total_bytes * max(1, local_ranks) > space.f_bavail * space.f_frsize // 2:
                raise OSError("insufficient disk headroom for original audit state")
            self.file = tempfile.TemporaryFile(dir=directory, buffering=0)  # noqa: SIM115 - bucket owns close
        try:
            if self.file is not None:
                self.file.truncate(self.total_bytes)
            digest = hashlib.blake2s()
            offset = 0
            for _, tensor, _, spans in _draft_fixture_field_ranges(self.fields):
                for chunk in _draft_field_chunks(tensor, spans, self.chunk_bytes):
                    size = chunk.numel() * chunk.element_size()
                    adjacent = bool(self.zero_runs and self.zero_runs[-1][1] == offset)
                    zero = ((adjacent or len(self.zero_runs) < zero_run_limit)
                            and not bool(chunk.contiguous().view(torch.uint8).any()))
                    # The image digest binds the lossless representation, not
                    # tensor_digest's externally used canonical field identity.
                    digest.update(struct.pack("<?QQ", zero, offset, size))
                    if zero:
                        if adjacent:
                            self.zero_runs[-1] = (self.zero_runs[-1][0], offset + size)
                        else:
                            self.zero_runs.append((offset, offset + size))
                        self.zero_bytes += size
                        offset += size
                        continue
                    payload = _draft_chunk_bytes(chunk)
                    digest.update(payload)
                    if self.memory is not None:
                        self.memory[offset:offset + size].copy_(
                            torch.frombuffer(payload, dtype=torch.uint8))
                    elif payload.obj.any():
                        self.file.seek(offset)
                        while payload:
                            written = self.file.write(payload)
                            if not written:
                                raise OSError("audit snapshot short write")
                            payload = payload[written:]
                        os.fdatasync(self.file.fileno())
                        os.posix_fadvise(self.file.fileno(), offset, size, os.POSIX_FADV_DONTNEED)
                    offset += size
            if offset != self.total_bytes:
                raise ValueError("audit snapshot byte schema changed")
            self.identity = digest.digest()
        except BaseException:
            self.close()
            raise

    def restore(self):
        if self.closed:
            raise ValueError("audit snapshot is already released")
        bindings = tuple((tuple(t.shape), tuple(t.stride()), t.data_ptr(), t.dtype, t.device)
                         for _, t, _ in self.fields)
        if bindings != self.bindings:
            raise ValueError("audit original state storage binding changed")
        digest = hashlib.blake2s()
        offset = 0
        run = 0
        previous = -1
        for start, stop in self.zero_runs:
            if not previous < start < stop <= self.total_bytes:
                raise ValueError("audit original zero-run schema changed")
            previous = stop
        zero_streams = {}
        try:
            for _, tensor, _, spans in _draft_fixture_field_ranges(self.fields):
                for chunk in _draft_field_chunks(tensor, spans, self.chunk_bytes):
                    size = chunk.numel() * chunk.element_size()
                    while run < len(self.zero_runs) and self.zero_runs[run][1] <= offset:
                        run += 1
                    zero = (run < len(self.zero_runs)
                            and self.zero_runs[run][0] <= offset
                            and offset + size <= self.zero_runs[run][1])
                    digest.update(struct.pack("<?QQ", zero, offset, size))
                    if zero:
                        if chunk.is_cuda:
                            stream = torch.cuda.current_stream(chunk.device)
                            zero_streams[(stream.device, stream.cuda_stream)] = stream
                        chunk.zero_()
                        offset += size
                        continue
                    if self.memory is not None:
                        raw = self.memory[offset:offset + size]
                    else:
                        payload = bytearray(size)
                        view = memoryview(payload)
                        self.file.seek(offset)
                        while view:
                            read = self.file.readinto(view)
                            if not read:
                                raise OSError("audit snapshot short read")
                            view = view[read:]
                        raw = torch.frombuffer(payload, dtype=torch.uint8)
                    digest.update(memoryview(raw.numpy()))
                    # Adjacent heterogeneous fields need not begin at an
                    # element-aligned byte offset in the shared host image.
                    # Clone only that bounded chunk before the typed view.
                    if raw.storage_offset() % chunk.element_size():
                        raw = raw.clone()
                    chunk.copy_(raw.view(chunk.dtype).reshape(chunk.shape))
                    if self.file is not None:
                        os.posix_fadvise(self.file.fileno(), offset, size, os.POSIX_FADV_DONTNEED)
                    offset += size
        finally:
            # Keep restore blocking even when zero fills bypass host transport.
            for stream in zero_streams.values():
                stream.synchronize()
        if offset != self.total_bytes:
            raise ValueError("audit restore byte schema changed")
        if digest.digest() != self.identity:
            raise ValueError("audit original state storage identity changed")

    def close(self):
        if self.file is not None:
            self.file.close()
        self.memory = None
        self.closed = True


def _qualify_draft_window_case(capacity, round_, window, providers, targets, hidden,
                               admission, owner_order, *, rank, sampler_rank, sampling=None,
                               original_cache=None, split_review=False,
                               target_constraints=False, reference_cache=None,
                               witness_byte_budget=None, reference_device_budget=None,
                               input_observer=None, initialize_cold_fixture=False):
    """Prove the complete transaction, optionally observing completed forwards.

    Observation never replaces the existing state or output checks. Observed
    executions cannot supply serving cost priors because snapshot work changes
    their timing. The caller owns observer teardown on success and failure.
    """
    from cubic_serve.runtime.continuous_batch import (
        BatchCommand, DraftWindowBinding, advance_carried_draft_rows,
    )
    from cubic_serve.runtime.performance import (
        _bucket_for_logical_m, _execute_pipelined_draft_batch, _pipelined_results,
        _ragged_speculative_plan,
    )

    separate = split_review or target_constraints
    target, provider = round_.target.body.arena, round_.align.body.arena
    prior_constraint_binding = round_.target._sampling.constraint_bound
    prior_head_constraint_binding = round_.head._sampling.constraint_bound
    device = round_.target.body.hidden.device
    depth, stride = round_.depth, round_.stride
    pages_per_owner = (4 * stride + round_.target._block_size - 1) // round_.target._block_size
    n = len(owner_order)
    if pages_per_owner * n > min(target.spec.page_capacity, provider.spec.page_capacity):
        raise ValueError("draft startup fixture exceeds admitted cache")
    if type(initialize_cold_fixture) is not bool:
        raise ValueError("cold fixture policy must be explicit")
    if initialize_cold_fixture and (original_cache is None or not original_cache):
        # Establish the bucket's declared cold inputs BEFORE taking protected
        # row baselines. A smaller first case does not own the other rows.
        _prepare_cold_draft_fixture(round_, hidden, admission)
    fields, owners, target_states, provider_states = _draft_window_fixture_fields(
        round_, hidden, n, pages_per_owner)

    byte_budget = (_draft_witness_budget(dist.get_world_size() if dist.is_initialized() else 1)
                   if witness_byte_budget is None
                   else witness_byte_budget)

    def rows():
        return _draft_window_audit_rows(round_, round_.pool.metadata, n, pages_per_owner,
                                        owner_order=owner_order, sampling=sampling)

    protected_written = tuple(
        (name, arena.written, _draft_fixture_protected_pages(
            arena, n * pages_per_owner, state_rows, device=device))
        for name, arena, state_rows in (("target", target, target_states),
                                       ("provider", provider, provider_states)))
    protected_baseline = tuple(
        tensor.index_select(0, indices).clone() for _, tensor, indices in protected_written)

    def check_protected_written(phase):
        differences = tuple(name for (name, tensor, indices), expected in zip(
            protected_written, protected_baseline, strict=True)
            if not torch.equal(tensor.index_select(0, indices), expected))
        errors = [None] * dist.get_world_size()
        dist.all_gather_object(errors, differences)
        if any(errors):
            raise ValueError(f"draft composition changed protected page activity {phase}: {errors}")

    owns_original = original_cache is None
    if owns_original:
        original = _DraftBucketSnapshot(fields, byte_budget=byte_budget)
    else:
        if not original_cache:
            full_fields, *_ = _draft_window_fixture_fields(
                round_, hidden, round_.C, pages_per_owner)
            original_cache["receipt"] = _DraftBucketSnapshot(
                full_fields, byte_budget=byte_budget,
                local_ranks=dist.get_world_size() if dist.is_initialized() else 1)
        original = original_cache["receipt"]
    owner_ids = tuple(int(x) for x in owners.tolist())
    def metadata():
        return [round_.pool.metadata[i] for i in owner_ids]
    def set_metadata(values):
        for i, value in zip(owner_ids, values, strict=True):
            round_.pool.metadata[i] = value
    def take(tensor, dim):
        return tensor.index_select(dim, owners)
    def put(tensor, dim, values):
        tensor.index_copy_(dim, owners, values)
    old_metadata = list(metadata())
    old_tokens = take(round_.pool.tokens, 1).clone()
    old_scores = take(round_.pool.scores, 2).clone()
    old_ids = take(round_.pool.ids, 2).clone()
    reference_key = (id(round_), capacity, owner_order, sampling,
                     prior_constraint_binding, prior_head_constraint_binding)
    reference = None if reference_cache is None else reference_cache.get(reference_key)
    if reference_cache is not None:
        hits = [None] * dist.get_world_size()
        dist.all_gather_object(hits, reference is not None)
        if not all(hits):
            reference = None
        if reference is None:
            _retire_draft_fixture_references(reference_cache)

    try:
        _initialize_draft_window_fixture(
            fields, target, provider, target_states, owners, round_.pool,
            provider_states=provider_states)
        set_metadata([None] * n)
        def ordinary(batch_id, *, calibrate=False):
            if input_observer is not None:
                input_observer.before_forward("ordinary")
            command = BatchCommand(batch_id, capacity, rows())
            if calibrate and device.type == "cuda":
                # Prior reference images and host fixture preparation may finish
                # at different times on each rank. Exclude that offline skew
                # from the complete-round prior, before its wall timer starts.
                # Serving still measures its actual feeding/collective costs.
                torch.cuda.synchronize(device)
                dist.barrier(device_ids=[device.index])
            result = _execute_pipelined_draft_batch(
                command, targets[capacity], providers,
                hidden, admission, tuple(sorted(targets)), rank=rank, sampler_rank=sampler_rank,
            )
            if result is None:
                raise ValueError("draft startup fixture did not use the carried round")
            if input_observer is not None:
                input_observer.after_forward("ordinary")
            return result

        ordinary(0)
        steady_costs = []
        if reference is None:
            device_budget = (_draft_reference_device_budget(fields, byte_budget)
                             if reference_device_budget is None else reference_device_budget)
            seed = _DraftFixtureReference(
                fields, byte_budget=byte_budget, device_byte_budget=device_budget)
            pending = list(metadata())
            seed_tokens = take(round_.pool.tokens, 1).clone()
            seed_scores = take(round_.pool.scores, 2).clone()
            seed_ids = take(round_.pool.ids, 2).clone()
            expected_outputs = []
            for step in range(2):
                previous = list(metadata())
                result = ordinary(step + 1, calibrate=input_observer is None)
                if not result.draft_ingress:
                    steady_costs.append(result.step_ms)
                expected_outputs.append(tuple(
                    (*previous[owner][6][:int(result[0])], int(result[1]))
                    for owner, result in zip(owner_order, round_.host_result[:n].tolist(), strict=True)
                ))
            expected = _DraftFixtureReference(
                fields, byte_budget=byte_budget, device_byte_budget=device_budget)
            check_protected_written("sealed rounds")
            expected_pending = list(metadata())
            expected_tokens = take(round_.pool.tokens, 1).clone()
            expected_scores = take(round_.pool.scores, 2).clone()
            expected_ids = take(round_.pool.ids, 2).clone()
            if reference_cache is not None:
                # Private immutable references and small pending/RNG metadata.
                # Both roles share a resource-derived device bound; uncached
                # fields retain the bounded canonical identity proof.
                reference_cache.clear()
                reference_cache[reference_key] = (
                    seed, pending, seed_tokens, seed_scores, seed_ids,
                    expected_outputs, expected, expected_pending,
                    expected_tokens, expected_scores, expected_ids,
                )
        else:
            (seed, pending, seed_tokens, seed_scores, seed_ids,
             expected_outputs, expected, expected_pending,
             expected_tokens, expected_scores, expected_ids) = reference
            check_protected_written("reference ingress")
        if reference is None:
            # Regenerate exactly the original ingress, including the old
            # pending storage from before its first call. No altered seed is
            # allowed to hide behind a cached expected result.
            _initialize_draft_window_fixture(
                fields, target, provider, target_states, owners, round_.pool,
                provider_states=provider_states)
            set_metadata([None] * n)
            put(round_.pool.tokens, 1, old_tokens)
            put(round_.pool.scores, 2, old_scores)
            put(round_.pool.ids, 2, old_ids)
            ordinary(0)
        seed.compare(fields)
        if metadata() != pending:
            raise ValueError("draft seed regeneration changed pending owner metadata")
        for actual_tensor, seed_tensor in (
                (take(round_.pool.tokens, 1), seed_tokens),
                (take(round_.pool.ids, 2), seed_ids)):
            if not torch.equal(actual_tensor.contiguous().view(torch.uint8),
                               seed_tensor.contiguous().view(torch.uint8)):
                raise ValueError("draft seed regeneration changed pending proposal boundary")
        _compare_draft_candidate_scores(take(round_.pool.scores, 2), seed_scores, seed_ids)
        check_protected_written("regenerated seed")
        torch.cuda.synchronize(device)
        if separate:
            outputs = []
            # The first round uses the ordinary canonical binder. Subsequent
            # rounds exercise the actual continuation cursor/row advancement.
            batch_rows = rows()
            target_constraint_costs = []
            for step in range(2):
                if step:
                    batch_rows = advance_carried_draft_rows(
                        batch_rows, round_.pool.metadata, depth=depth,
                        block_size=round_.target._block_size)
                    if batch_rows != rows():
                        raise ValueError("masked continuation rows differ from canonical binding")
                segments, _ = _ragged_speculative_plan(batch_rows)
                round_.target._sampling.bind_constraints(
                    [None] * len(batch_rows), [False] * len(batch_rows))
                if target_constraints:
                    native_started = time.monotonic_ns()
                    forward_ahead = (step > 0 and getattr(
                        round_, "constraint_continuation_forward_graph", None) is not None)
                    if forward_ahead:
                        limit = pages_per_owner * round_.target._block_size
                        round_.bind_constraint_continuation(DraftWindowBinding(
                            depth, tuple(owner_ids[i] for i in owner_order),
                            tuple(batch_rows[s.start].position for s in segments),
                            (limit,) * n, (limit,) * n, (2 * stride,) * n,
                            tuple(batch_rows[s.start].page_table for s in segments),
                            tuple(batch_rows[s.start].mtp_page_table for s in segments),
                        ))
                        round_.start_constraint_forward()
                    try:
                        completed = round_.run(batch_rows, segments, target_constraints=True)
                    finally:
                        if forward_ahead:
                            round_.close_constraint_continuation()
                    target_constraint_costs.append((time.monotonic_ns() - native_started) / 1e6)
                else:
                    round_.run(batch_rows, segments, review_only=True)
                    proposals = []
                    for depth_index in range(depth):
                        round_.head._sampling.bind_constraints([None] * n, [False] * n)
                        proposals.append(round_.run_grammar_proposal(
                            depth_index, n, stochastic=sampling is not None))
                    round_.head.materialize_proposals((n,) * depth)
                    completed = tuple(
                        (int(decision[0]), int(decision[1]),
                         *(tokens[r] for tokens in proposals), 1)
                        for r, decision in enumerate(round_.host_decisions[:n].tolist())
                    )
                parts = _pipelined_results(batch_rows, segments, completed, pool=round_.pool)
                outputs.append(tuple(part.output_token_ids for part in parts))
            records = (None, None)
        elif getattr(window, "requires_budget", False):
            round_.budget.bind([None] * n)
        if not separate:
            if input_observer is not None:
                input_observer.before_forward("resident")
            window_started = time.monotonic_ns()
            window.bind(
                owners=tuple(owner_ids[i] for i in owner_order), target_positions=tuple(pending[owner][1] for owner in owner_order),
                target_limits=(pages_per_owner * round_.target._block_size,) * n,
                provider_limits=(pages_per_owner * round_.target._block_size,) * n,
                remaining=(3 * stride,) * n, eos=((),) * n,
                # Reuse the same two mathematical rounds, but stop through the
                # actual observation budget before the granted maximum of three.
                decision_tokens=(0,) * n,
                decision_rounds=(1,) * (n - 1) + (2,),
                max_steps=3, max_duration_seconds=3600.,
            )
            window.replay()
            records = window.consume()
            window_ms = (time.monotonic_ns() - window_started) / 1e6
            if input_observer is not None:
                input_observer.after_forward("resident")
        if len(records) != 2:
            raise ValueError("draft composition ended before its granted rounds")
        schedule = [None] * dist.get_world_size()
        dist.all_gather_object(schedule, (capacity, owner_order, sampling, len(records)))
        if any(item != schedule[0] for item in schedule):
            raise ValueError("draft composition rank event schedules differ")
        for step, output in enumerate(expected_outputs):
            if separate:
                actual_outputs = outputs[step]
            else:
                receipt = window.receipt(step, batch_id=0,
                                         request_ids=tuple(pending[owner][0] for owner in owner_order))
                actual_outputs = tuple(item.output_token_ids for item in receipt.segment_results)
            if actual_outputs != output:
                raise ValueError(
                    f"draft composition token prefixes differ capacity={capacity} "
                    f"owners={owner_order} sampling={sampling} step={step} "
                    f"split={split_review} actual={actual_outputs} expected={output}")
        check_protected_written("resident window")
        expected.compare(fields)
        for actual_tensor, expected_tensor in (
            (take(round_.pool.tokens, 1), expected_tokens),
            (take(round_.pool.ids, 2), expected_ids),
        ):
            if not torch.equal(actual_tensor.contiguous().view(torch.uint8),
                               expected_tensor.contiguous().view(torch.uint8)):
                raise ValueError("draft composition changed pending proposal boundary")
        _compare_draft_candidate_scores(take(round_.pool.scores, 2), expected_scores,
                                        expected_ids)
        if metadata() != expected_pending:
            raise ValueError("draft composition changed pending owner metadata")
        logical_rows = n * stride
        if (input_observer is None
                and _bucket_for_logical_m(logical_rows, tuple(sorted(targets))) == capacity
                and _bucket_for_logical_m(n, tuple(sorted(providers))) == round_.C):
            # Only after the complete output/state/write-set witness passed.
            # Reuse existing steady ordinary calls: no extra forward, reference
            # callback, synchronization or calibration pass is introduced.
            if steady_costs:
                prices = getattr(round_, "_startup_round_costs", None)
                if prices is None:
                    prices = round_._startup_round_costs = {}
                prices.setdefault((logical_rows, sampling is None), []).extend(steady_costs)
            # Cost evidence belongs to the execution that actually ran. A
            # staged K1 observation cannot price an unseen captured K3 round.
            # These reuse the existing qualified forwards, omitting reference
            # snapshots/comparisons and controller feeding. Real request costs
            # supersede the priors, including schema-specific matcher work.
            # CPU grammar work depends on the actual request language. An
            # all-ones phase fixture is not its complete-round price. The
            # integer budget window, in contrast, executes the same native
            # ledger/sampling composition here and in serving.
            if target_constraints:
                constrained = getattr(round_, "_startup_constraint_round_costs", None)
                if constrained is None:
                    constrained = round_._startup_constraint_round_costs = {}
                # Native priors omit the request-specific CPU matcher. Live
                # complete-round observations supersede these and model their
                # excess within the same processing domain/physical bucket.
                constrained.setdefault((logical_rows, sampling is None, 1), []).extend(
                    target_constraint_costs)
            if (not separate and window.requires_budget
                    and window.requires_greedy == (sampling is None)):
                constrained = getattr(round_, "_startup_constraint_round_costs", None)
                if constrained is None:
                    constrained = round_._startup_constraint_round_costs = {}
                constrained.setdefault((logical_rows, sampling is None, 2), []).append(
                    window_ms / len(records))
        return (capacity, owner_order, depth,
                None if sampling is None else (sampling.temperature, sampling.top_k,
                                                sampling.top_p, sampling.seed),
                "sealed-round-composition-identity")
    finally:
        round_.target._sampling.constraint_bound = prior_constraint_binding
        round_.head._sampling.constraint_bound = prior_head_constraint_binding
        if owns_original:
            try:
                original.restore()
            finally:
                original.close()
        set_metadata(old_metadata)
        put(round_.pool.tokens, 1, old_tokens)
        put(round_.pool.scores, 2, old_scores)
        put(round_.pool.ids, 2, old_ids)
        window.host_flags[:] = 0
        round_.head_greedy.invalidate_metadata()
        round_.target_greedy.invalidate_metadata()
        round_._last_transfer = True
        torch.cuda.synchronize(device)


def _draft_window_case_plan(capacity, round_, provider_capacities, target_capacities, reuse):
    """Keep every existing witness; cluster identical immutable references."""
    from cubic_serve.generation.sampler import SamplingParameters

    target_domain = set(_logical_m_values(capacity, target_capacities))
    owner_domain = tuple(n for n in _logical_m_values(round_.C, provider_capacities)
                         if n * round_.stride in target_domain)
    cases = []
    stochastic = SamplingParameters(1., .95, 20, 0x123456789)
    for attribute in ("_resident_window", "_candidate_window",
                      "_budget_resident_window", "_budget_candidate_window"):
        window = getattr(round_, attribute, None)
        if window is None:
            continue
        maps = _draft_window_audit_owner_maps(round_.C, owner_domain)
        if reuse:
            padded = max(1, min(round_.C - 1, owner_domain[0]))
            maps = tuple(dict.fromkeys((tuple(range(round_.C)), tuple(reversed(range(padded))))))
        for sampling in ((None,) if window.requires_greedy else (None, stochastic)):
            cases.extend((attribute, window, sampling, order, False, False) for order in maps)
    if getattr(round_, "grammar_review_graph", None) is not None:
        maps = tuple(dict.fromkeys((tuple(range(round_.C)),
                                  tuple(reversed(range(max(1, round_.C - 1)))))))
        for label, split, constrained in (("masked-review", True, False),
                                          ("target-constraint", False, True)):
            for sampling in (None, stochastic):
                cases.extend((label, round_._resident_window, sampling, order, split, constrained)
                             for order in maps)
    return sorted(cases, key=lambda case: (len(case[3]), case[3], case[2] is not None))


def qualify_draft_windows(rounds, providers, targets, hidden, admission, *, rank, sampler_rank,
                         progress=None, reuse_receipts=True, initialize_cold_fixture=False):
    """Witness unchanged-arithmetic full/padded/permuted multi-round composition.

    This is capture/state identity. Independent speculative numerical audits
    remain a separate acceptance gate.
    """
    from itertools import groupby
    from cubic_serve.runtime.performance import (
        _read_execution_audit_receipt, _write_execution_audit_receipt,
    )
    started = time.perf_counter()

    if not rounds:
        return stable_digest(())
    collective = bool(targets and providers and dist.is_initialized())
    key = (_draft_composition_receipt_key(rounds, providers, targets, admission)
           if collective else None)
    cached = (None if key is None or not reuse_receipts else
              _read_execution_audit_receipt(key, rank, check="mtp-composition"))
    reuse = False
    if collective:
        device = next(iter(targets.values())).body.hidden.device
        consensus = torch.tensor([cached is not None], dtype=torch.int32, device=device)
        dist.all_reduce(consensus, op=dist.ReduceOp.MIN)
        reuse = bool(consensus.item())
    diagnostic_event(_LOGGER, 'quantized_qualification.mtp_composition_qualification_rank_d_full_receipt',
        rank,
        "hit" if reuse else "miss",
    )

    def schema(item):
        round_ = item[1]
        return round_.depth, round_.stride, round_.target._block_size

    cases = {key: _draft_window_case_plan(key[0], round_, tuple(sorted(providers)),
                                        tuple(sorted(targets)), reuse)
             for key, round_ in rounds.items()}
    total = sum(map(len, cases.values()))
    completed, notified = 0, time.monotonic()
    if progress is not None:
        progress(0, total)
    witnesses = []
    for _, members in groupby(sorted(rounds.items(), key=lambda item: (schema(item), item[0])), key=schema):
        group = tuple(members)
        world = dist.get_world_size() if dist.is_initialized() else 1
        byte_budget = _draft_witness_budget(world)
        original_cache = {}
        reference_cache = {}
        try:
            for (capacity, _depth), round_ in group:
                if original_cache:
                    raise ValueError("previous draft bucket still owns its original state")
                _retire_draft_fixture_references(reference_cache)
                for attribute, window, sampling, owner_order, split, constrained in cases[(capacity, _depth)]:
                    witnesses.append((attribute, _qualify_draft_window_case(
                        capacity, round_, window, providers, targets, hidden, admission,
                        owner_order, rank=rank, sampler_rank=sampler_rank,
                        sampling=sampling, witness_byte_budget=byte_budget,
                        original_cache=original_cache, reference_cache=reference_cache,
                        split_review=split, target_constraints=constrained,
                        initialize_cold_fixture=initialize_cold_fixture)))
                    completed += 1
                    now = time.monotonic()
                    if progress is not None and (completed == total or now - notified >= 1.):
                        progress(completed, total)
                        notified = now
                original = original_cache.pop("receipt", None)
                if original is not None:
                    try:
                        original.restore()
                    finally:
                        original.close()
                for attribute in ("_resident_window", "_candidate_window",
                                  "_budget_resident_window", "_budget_candidate_window"):
                    window = getattr(round_, attribute, None)
                    if window is not None:
                        window.qualified = True
                if getattr(round_, "grammar_review_graph", None) is not None:
                    round_.target_constraints_qualified = True
                    round_.constraint_continuation_qualified = bool(
                        getattr(round_, "constraint_continuation_forward_graph", None) is not None)
                    round_.grammar_qualified = True
                    round_.grammar_continuation_qualified = True
                round_.budget_qualified = bool(
                    getattr(round_, "budget", None) is not None and round_.budget.qualified
                    and all(getattr(getattr(round_, attribute, None), "qualified", False)
                            for attribute in ("_budget_resident_window", "_budget_candidate_window"))
                )
        finally:
            original = original_cache.pop("receipt", None)
            if original is not None:
                try:
                    original.restore()
                finally:
                    original.close()
            _retire_draft_fixture_references(reference_cache)
    live = stable_digest(witnesses)
    # A write requires the complete matrix to have passed on every rank.
    # A partial miss never composes independently cached rank proofs.
    if collective:
        dist.barrier()
    if key is not None and not reuse:
        _write_execution_audit_receipt(key, rank, live, check="mtp-composition")
    diagnostic_event(_LOGGER, 'quantized_qualification.mtp_composition_proof_rank_d_cases_d',
        rank,
        len(witnesses),
        "reused" if reuse else "full",
        time.perf_counter() - started,
    )
    return stable_digest({"full": cached if reuse else live, "live": live})


def portfolio_resources(
    executors: dict[int, Any], draft_executors: dict[int, Any] | None = None,
) -> RankReplayResources:
    """Name the complete instance-owned graph portfolio, including child lanes.

    These are lifetime identities, deliberately distinct from reusable semantic
    plan digests. Every bucket aliases the same cache and is leased together.
    """
    stages = []
    graphs = []
    streams: list[int] = []
    executor_owners = []
    portfolio = (*executors.values(), *(draft_executors or {}).values())
    for executor in portfolio:
        executor_owners.append(id(executor))
        transfers = {id(p): p for p in getattr(executor, "_state_transfer_bindings", {}).values()
                     if p is not None}
        for transfer in transfers.values():
            graphs.extend((id(transfer), cap, id(graph))
                          for cap, graph in transfer.graphs.items())
            streams.append(transfer.stream.cuda_stream)
            executor_owners.extend((id(transfer), id(transfer.ctx), id(transfer.decisions),
                                    id(transfer.done), *(id(h) for h in transfer.hosts),
                                    *(id(e) for e in transfer.events)))
        windows = (getattr(executor, "_resident_window", None),
                   getattr(executor, "_resident_stochastic_window", None))
        for window in windows:
            if window is None:
                continue
            graphs.append((id(window), int(window.graph), int(window.exec)))
            graphs.extend((id(window), id(child)) for child in window.children)
            executor_owners.extend((id(window), id(window.host), id(window.meta)))
            maintenance = getattr(window, 'draft_maintenance', None)
            if maintenance is not None:
                executor_owners.extend((id(maintenance),id(maintenance.updates),id(maintenance.reset_activity),
                    id(maintenance.owner_slots),
                    id(maintenance.private_states),id(maintenance.private_pages),
                    id(maintenance.table_host)))
        compaction = getattr(executor, "_resident_compaction", None)
        if compaction is not None:
            graphs.append((id(compaction), id(compaction.graph)))
            executor_owners.extend((id(compaction), id(compaction.indices),
                                    id(compaction.scratch), id(compaction.count)))
        draft_rounds = getattr(executor, "_pipelined_draft_rounds", {})
        if not draft_rounds:
            current = getattr(executor, "_pipelined_draft_round", None)
            draft_rounds = {} if current is None else {current.depth: current}
        for draft_round in draft_rounds.values():
            # The round aliases model/cache storage and owns additional fixed
            # ingress, verification and pending buffers. One lease covers all.
            graphs.extend((id(draft_round), id(graph)) for graph in
                          (draft_round.graph, draft_round.candidate_graph,
                           getattr(draft_round, "budget_graph", None),
                           getattr(draft_round, "budget_candidate_graph", None),
                           getattr(draft_round, "target_constraint_forward_graph", None),
                           getattr(draft_round, "constraint_continuation_forward_graph", None),
                           getattr(draft_round, "grammar_review_graph", None)) if graph is not None)
            graphs.extend((id(draft_round), id(graph)) for graph in
                          getattr(draft_round, "target_constraint_graphs", {}).values())
            graphs.extend((id(draft_round), id(graph)) for graph in
                          getattr(draft_round, "_forward_graphs", {}).values())
            for phase_graphs in getattr(draft_round, "grammar_proposal_graphs", {}).values():
                graphs.extend((id(draft_round), id(graph)) for graph in phase_graphs)
            for attribute in ("host_decisions", "grammar_host_tokens", "constraint_forward_valid"):
                if hasattr(draft_round, attribute):
                    executor_owners.append(id(getattr(draft_round, attribute)))
            executor_owners.extend((id(draft_round), id(draft_round.pool)))
            streams.append(draft_round.target.slot.stream.cuda_stream)
            for attribute in ("_resident_window", "_candidate_window",
                              "_budget_resident_window", "_budget_candidate_window"):
                draft_window = getattr(draft_round, attribute, None)
                if draft_window is not None:
                    graphs.append((id(draft_window), int(draft_window.graph), int(draft_window.exec)))
                    graphs.extend((id(draft_window), id(child)) for child in draft_window.children)
                    executor_owners.extend((id(draft_window), id(draft_window.host),
                                            id(draft_window.meta), id(draft_window.staging)))
            budget = getattr(draft_round, "budget", None)
            if budget is not None:
                executor_owners.extend(id(value) for value in (
                    budget, budget.state, budget.host, budget.scratch, budget.markers,
                    budget.offsets, budget.mask, budget.activity,
                ))
        binding = getattr(executor, "_draft_binding", None)
        if binding is not None:
            child = binding.executor
            executor_owners.append(id(child))
            update = getattr(child, "_input_update_graph", None)
            if update is not None:
                graphs.append((id(child), id(update)))
        stage = executor.slot
        stages.append(stage if hasattr(stage, "buffers") else executor.body)
        streams.append(stage.stream.cuda_stream)
        captured = getattr(stage, "_graph", None)
        graphs.append((id(stage), id(captured if captured is not None else stage.graph)))
        state_graph = getattr(stage, "_state_graph", None)
        if state_graph is not None:
            graphs.append((id(stage), id(state_graph)))
        update_graph = getattr(executor, "_input_update_graph", None)
        if update_graph is not None:
            graphs.append((id(executor), id(update_graph)))
        if hasattr(executor._sampling, "_constraint_stream"):
            streams.append(executor._sampling._constraint_stream.cuda_stream)
            executor_owners.extend(
                (
                    id(executor._sampling._constraint_mask),
                    id(executor._sampling._constraint_activity),
                    *(id(host) for host in executor._sampling._constraint_hosts),
                )
            )
        for sampler in (
            *executor._sampling.greedy_plans,
            *getattr(executor._sampling, "masked_plans", ()),
        ):
            graphs.extend((id(sampler), id(graph)) for graph in sampler.graphs)
            graphs.extend(
                (id(sampler), id(graph))
                for graph in (
                    sampler.grammar_candidate_graph,
                    sampler.grammar_random_graph,
                )
                if graph is not None
            )
            streams.append(sampler.stream.cuda_stream)
            executor_owners.extend((id(sampler), id(sampler.token_mask)))
        overlap = getattr(stage, "_overlap_resources", None)
        if overlap is not None:
            stages.extend(overlap[1])
            graph, _, schedule, *_ = overlap
            graphs.append((id(stage), id(graph)))
            streams.extend((schedule.compute.cuda_stream, schedule.communication.cuda_stream))
        legacy_overlap = getattr(stage, "_overlap", None)
        if legacy_overlap is not None:
            graph, schedule, children = legacy_overlap
            stages.extend(children)
            graphs.append((id(stage), id(graph)))
            streams.extend((schedule.compute.cuda_stream, schedule.communication.cuda_stream))
    rank = stages[0].rank
    # Rank executors own pre/post-replay staging and sampling workspaces in
    # addition to the tensors in each model body.
    return RankReplayResources(
        rank=rank,
        graph_instance_id=stable_digest(graphs),
        stream_lane_id=stable_digest(
            streams
        ),
        buffer_set_id=stable_digest([id(stage.buffers) for stage in stages] + executor_owners),
        workspace_arena_id=stable_digest([
            id(stage.program) if isinstance(stage, FlashNextDecodeStage)
            else id(getattr(stage, "body", stage))
            for stage in stages
        ] + executor_owners),
        scratch_arena_id=stable_digest([id(stage.arena) for stage in stages]),
    )
