"""Normal startup resources and per-operator cold/hot plan reuse.

Engineering qualification is deliberately not a prerequisite or a cache value.
Each process constructs and binds its own graphs; portable caches contain only
selected plans and measured costs, indexed by kernels, GPU model and operands.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from cubic_serve.ops.cuda.collective import all_reduce
from cubic_serve.executor.readiness import execution_ready, prepare_execution
from cubic_serve.guardrail.compare import stable_digest
from cubic_serve.ops.cuda.cache_identity import module_kernels, plan_path

_LOGGER = logging.getLogger(__name__)


class StartupPreparation:
    engineering = False
    recording = False

    def __init__(self, rank, device, *, consensus=None, topology=()):
        self.rank, self.device = rank, device
        self.kernels = ()
        self.topology = topology
        self.hits = self.misses = 0
        self.failed = False
        self._consensus = consensus

    def consensus(self, valid):
        if self._consensus is not None:
            return self._consensus(valid)
        flag = torch.tensor([int(valid)], dtype=torch.int32, device=self.device)
        all_reduce(flag, op=dist.ReduceOp.MIN)
        return bool(flag.item())

    def bind_portfolio(self, executors):
        from cubic_serve.runtime.quantized_qualification import _bound_composition_plan
        self.kernels = tuple(sorted(set(
            _bound_composition_plan(e.body, kernels_only=True) for e in executors.values())))

    def run(self, label, identity, action, *, kernels=None, validate=None):
        """Restore a measured plan, or produce precisely this missing entry."""
        from cubic_serve.ops.cuda.launch_cost import _read_startup_calibration, _write_startup_calibration
        from cubic_serve.runtime.quantized_qualification import _receipt_pack, _receipt_unpack

        kernels = self.kernels if kernels is None else kernels
        path = plan_path("startup", kernels, self.device, (label, identity, self.rank))
        body = _read_startup_calibration(path, "cubic.startup-cost.v1")
        value = None
        valid = body is not None and "value" in body
        if valid:
            try:
                value = _receipt_unpack(body["value"])
                valid = validate is None or bool(validate(value))
            except (TypeError, ValueError, KeyError):
                valid = False
        if self.consensus(valid):
            self.hits += 1
            return value
        self.misses += 1
        try:
            value = action()
            if validate is not None and not validate(value):
                raise ValueError("startup preparation produced an invalid plan entry")
            packed = _receipt_pack(value)
        except BaseException:
            self.failed = True
            raise
        # The action may be collective. Publish only after all ranks completed.
        if not self.consensus(True):
            raise RuntimeError("startup preparation did not complete on every rank")
        _write_startup_calibration(path, "cubic.startup-cost.v1", {"value": packed})
        return value

    def publish(self, label, identity, value, *, kernels):
        """Publish actual offline measurements, without any qualification flags."""
        from cubic_serve.ops.cuda.launch_cost import _write_startup_calibration
        from cubic_serve.runtime.quantized_qualification import _receipt_pack
        path = plan_path("startup", kernels, self.device, (label, identity, self.rank))
        _write_startup_calibration(path, "cubic.startup-cost.v1", {"value": _receipt_pack(value)})

    def memory_plan(self, identity, measure, *, implementation, dependencies=None):
        """Reuse a measured portfolio size, never its live resources or proofs.

        Geometry/runtime/source inputs form the key. The selected operator-plan
        files actually consulted by the measurement form its dependency manifest.
        All ranks agree on reuse before any rank enters collective construction.
        Final allocation and post-capture reserve checks remain process-local.
        """
        from cubic_serve.ops.cuda.launch_cost import (
            _plan_file_digest, _read_startup_calibration, _write_startup_calibration,
            startup_plan_dependencies,
        )
        from cubic_serve.runtime.quantized_qualification import _receipt_pack, _receipt_unpack

        self.memory_plan_reused = False
        path = plan_path("startup-memory", implementation, self.device, (identity, self.rank))
        body = _read_startup_calibration(path, "cubic.startup-memory.v2")
        value = None
        valid = isinstance(body, dict) and isinstance(body.get("dependencies"), dict)
        if valid:
            try:
                value = _receipt_unpack(body["value"])
                valid = (valid_memory_plan(value)
                         and all(isinstance(name, str)
                                 and (digest is None or isinstance(digest, str))
                                 and _plan_file_digest(Path(name)) == digest
                                 for name, digest in body["dependencies"].items()))
            except (KeyError, TypeError, ValueError):
                valid = False
        if self.consensus(valid):
            self.hits += 1
            self.memory_plan_reused = True
            _LOGGER.info("Graph memory plan restored rank=%d resident_bytes=%d transient_bytes=%d preparation_bytes=%d",
                         self.rank, value["graph_bytes"], value["transient_bytes"], value["preparation_bytes"])
            return value
        self.misses += 1
        _LOGGER.info("Graph memory plan missing or changed rank=%d; calibrating resources once", self.rank)
        try:
            with startup_plan_dependencies() as measured_dependencies:
                value = measure()
        except BaseException:
            self.failed = True
            raise
        if not valid_memory_plan(value):
            self.failed = True
            raise ValueError("invalid graph memory calibration")
        if not self.consensus(True):
            raise RuntimeError("graph memory calibration incomplete on a peer")
        _write_startup_calibration(path, "cubic.startup-memory.v2",
                                  {"value": _receipt_pack(value),
                                   "dependencies": dict(dependencies or {}, **measured_dependencies)})
        return value

    def cost_identity(self, label, capacity, executor, entry):
        from cubic_serve.runtime.quantized_qualification import _bound_composition_plan
        resource = executor.body
        if label == "sampling-costs":
            resource = SimpleNamespace(program=(executor._sampling.greedy_plans,
                                                executor._sampling.masked_plans))
        images = (_bound_composition_plan(resource, kernels_only=True),)
        workload = _bound_composition_plan(resource, kernels_only=True, workload=True)
        from cubic_serve.runtime.performance import _cold_start_calibration_signature
        corpus = _cold_start_calibration_signature(label, capacity, executor.body, entry)
        identity = (capacity, entry, self.topology, workload)
        return (identity if corpus is None else (*identity, corpus)), images

    def costs(self, label, executors, action, *, entry=()):
        values = []
        for capacity, executor in executors.items():
            identity, images = self.cost_identity(label, capacity, executor, entry)
            values.extend(self.run(label, identity,
                lambda capacity=capacity, executor=executor: action({capacity: executor}),
                kernels=images))
        return tuple(values)

    def publish_costs(self, label, executors, values, *, entry=()):
        for capacity, executor in executors.items():
            identity, images = self.cost_identity(label, capacity, executor, entry)
            self.publish(label, identity, tuple(v for v in values if v[0] == capacity), kernels=images)

    def draft_cost_identity(self, key, round_):
        from cubic_serve.runtime.quantized_qualification import _bound_composition_plan
        from cubic_serve.executor import draft_graph, state_transfer
        resources = (*(stage.body for stage in (round_.target, round_.align, round_.head)),
            SimpleNamespace(program=(round_.target_greedy, round_.head_greedy,
                                     round_.target_masked, round_.head_masked, round_.budget)))
        images = (tuple(_bound_composition_plan(stage, kernels_only=True) for stage in resources),
                  module_kernels(draft_graph), module_kernels(state_transfer))
        workload = tuple(_bound_composition_plan(stage, kernels_only=True, workload=True)
                         for stage in resources)
        # Root tokens and request identity are logical ordinals, independent
        # of the physical state IDs supplied by the cache allocator.
        fixture=(tuple((i,(17+i)%round_.target.vocab_size) for i in range(round_.C)),
                 3,1.,.95,min(20,round_.target.vocab_size,round_.target._sampling.parameters.max_top_k),1,
                 (8*round_.stride+round_.target._block_size-1)//round_.target._block_size)
        return (key,self.topology,workload,fixture),images

    def publish_draft_costs(self, rounds):
        for key, round_ in rounds.items():
            identity, images = self.draft_cost_identity(key, round_)
            self.publish("draft-costs", identity,
                (round_._startup_round_costs, round_._startup_constraint_round_costs), kernels=images)

    def capture(self, stage):
        if not getattr(stage, "_captured_once", False):
            raise RuntimeError("execution stage has not captured its CUDA graph")
        prepare_execution(stage, graphs=(stage.graph,), tensors=(stage.hidden,))
        return stable_digest(("live-capture", stage.captured_m))

    def state(self, executors):
        for executor in executors.values():
            body = executor.body
            if not execution_ready(body) or body.arena.spec.state_slot_capacity <= 0:
                raise RuntimeError("execution state is not bound to captured resources")
        return stable_digest(("live-state", tuple(sorted(executors))))

    def component(self, owner, identity, *, method="qualify", args=(), attributes=()):
        graphs = getattr(owner, "graphs", ())
        graphs = tuple(graphs.values()) if isinstance(graphs, dict) else tuple(graphs)
        module = getattr(owner, "module", None)
        prepare_execution(owner, graphs=graphs,
                          modules=() if module is None else (module,),
                          calls=tuple(getattr(owner, "calls", ())),
                          tensors=tuple(getattr(owner, "fields", ())))

    def window(self, window, *, segments=None):
        if window.exec is None or not execution_ready(window.sampler):
            raise RuntimeError("resident window is missing its bound graph or sampler")
        caps = ("qualified", "program_qualified")
        if segments is not None:
            caps += ("segmented_program_qualified",)
        prepare_execution(window, graphs=(window.graph, window.exec), capabilities=caps)
        window.e._bound_active_rows = window.e.captured_m

    def compaction(self, compaction, identity, action):
        prepare_execution(compaction, graphs=(compaction.graph,),
                          tensors=tuple(t for _, t, _, _ in compaction.fields))

    def draft_verifier(self, drafts, targets):
        for capacity, draft in drafts.items():
            if not execution_ready(draft.body) or draft._sampling.workspace is None:
                raise RuntimeError("draft verifier has no bound provider or sampling workspace")
            prepare_execution(draft, graphs=(draft.body.graph,),
                              tensors=(draft._verification_decisions,),
                              capabilities=("_verifier_qualified",))

    def draft_rounds(self, rounds, providers, targets, hidden, admission, **kwargs):
        for key, round_ in rounds.items():
            if not all(execution_ready(s) for s in (round_.target_greedy, round_.head_greedy,
                                                    round_.target_masked, round_.head_masked)):
                raise RuntimeError("draft round has an unprepared sampler")
            for name in ("_resident_window", "_candidate_window", "_budget_resident_window",
                         "_budget_candidate_window"):
                window = getattr(round_, name, None)
                if window is not None:
                    if window.exec is None or window.round is not round_:
                        raise RuntimeError("draft window differs from its captured transaction")
                    prepare_execution(window, graphs=(window.graph, window.exec))
            capabilities = ["grammar_qualified", "grammar_continuation_qualified",
                            "target_constraints_qualified", "constraint_continuation_qualified"]
            graphs = (round_.graph, round_.candidate_graph, round_.grammar_review_graph,
                      round_.target_constraint_forward_graph,
                      round_.constraint_continuation_forward_graph,
                      *round_._forward_graphs.values())
            if round_.budget is not None:
                if not execution_ready(round_.budget):
                    raise RuntimeError("draft budget program has not bound its native kernel")
                graphs += (round_.budget_graph, round_.budget_candidate_graph)
                capabilities.append("budget_qualified")
            prepare_execution(round_, graphs=graphs, capabilities=tuple(capabilities))
            round_._startup_round_costs = {}
            round_._startup_constraint_round_costs = {}
        self.bind_portfolio({**{('target', k): v for k, v in targets.items()},
                             **{('provider', k): v for k, v in providers.items()}})

        for key, round_ in rounds.items():
            def measure(key=key, round_=round_):
                warm_draft_costs({key: round_}, providers, targets, hidden, admission,
                                 rank=self.rank, sampler_rank=kwargs["sampler_rank"])
                return round_._startup_round_costs, round_._startup_constraint_round_costs
            identity, images = self.draft_cost_identity(key, round_)
            ordinary, constrained = self.run("draft-costs", identity, measure, kernels=images,
                validate=lambda values, round_=round_: valid_draft_costs(values, round_))
            rounds[key]._startup_round_costs = ordinary
            rounds[key]._startup_constraint_round_costs = constrained
        return stable_digest(("live-draft-bindings", tuple(rounds)))

    def finish(self):
        if self.failed:
            raise RuntimeError("startup preparation failed")
        if self.rank == 0:
            _LOGGER.info("Startup plans ready reused=%d prepared=%d", self.hits, self.misses)


def valid_memory_plan(value):
    if not isinstance(value, dict):
        return False
    if not (all(type(value.get(name)) is int and value[name] >= 0
                for name in ("graph_bytes", "transient_bytes", "allocator_slack", "preparation_bytes"))
            and isinstance(value.get("extra_state_fields"), tuple)
            and isinstance(value.get("overlap_plans"), dict)
            and isinstance(value.get("round_keys"), tuple)):
        return False
    from cubic_serve.executor.decode_overlap import DecodeOverlapPlan
    try:
        names = []
        for name, dtype, shape in value["extra_state_fields"]:
            if (not isinstance(name, str) or not name
                    or not isinstance(dtype, str) or not isinstance(getattr(torch, dtype, None), torch.dtype)
                    or not isinstance(shape, tuple)
                    or any(type(n) is not int or n <= 0 for n in shape)):
                return False
            names.append(name)
        if len(names) != len(set(names)):
            return False
        for m, plan in value["overlap_plans"].items():
            if type(m) is not int or m <= 0 or not isinstance(plan, tuple) or len(plan) != 3:
                return False
            DecodeOverlapPlan(*plan)
        return all(isinstance(key, tuple) and len(key) == 2
                   and all(type(n) is int and n > 0 for n in key)
                   for key in value["round_keys"])
    except (TypeError, ValueError):
        return False


def valid_draft_costs(values, round_):
    """Retain all measured owner populations and both constraint domains."""
    if not (isinstance(values, tuple) and len(values) == 2
            and all(isinstance(v, dict) for v in values)
            and {(round_.C * round_.stride, False), (round_.C * round_.stride, True)} <= set(values[0])):
        return False
    for domain, prices in enumerate(values):
        for key, samples in prices.items():
            if (not isinstance(key, tuple) or len(key) != 2 + domain
                    or type(key[0]) is not int or not round_.M // 2 < key[0] <= round_.C * round_.stride
                    or key[0] % round_.stride or type(key[1]) is not bool
                    or domain and (type(key[2]) is not int or key[2] not in (1, 2))
                    or not isinstance(samples, (list, tuple)) or not samples
                    or not all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in samples)):
                return False
    return True


def warm_draft_costs(rounds, providers, targets, hidden, admission, *, rank, sampler_rank):
    """Price existing captured transactions on private startup-owned rows.

    No reference comparisons, poison corpus, permutations or numerical verdicts.
    Every measured path uses ordinary bind/guard/commit, before HTTP readiness.
    """
    from cubic_serve.runtime.continuous_batch import BatchCommand, BatchRow
    from cubic_serve.runtime.performance import _execute_pipelined_draft_batch
    from cubic_serve.generation.sampler import SamplingParameters, GREEDY_SAMPLING

    for (capacity, depth), round_ in rounds.items():
        device = round_.target.body.hidden.device
        storage=getattr(round_.target.body.arena,"shared_storage",None)
        owner_ids=tuple(range(round_.C)) if storage is None else tuple(
            storage.state_capacity-1-i for i in range(round_.C))
        owners = torch.tensor(owner_ids,dtype=torch.int64,device=device)
        page_count = (8 * round_.stride + round_.target._block_size - 1) // round_.target._block_size
        pages = torch.arange(round_.C * page_count, dtype=torch.int64, device=device)
        for greedy in (False, True):
            for body in (round_.target.body, round_.align.body):
                body.arena.reset_state_rows(owners)
                body.arena.reset_page_rows(pages)
            hidden.index_fill_(0, owners, 0)
            for tensor in round_.pool.backup.values():
                tensor.index_fill_(0, owners, 0)
            for owner in owner_ids:round_.pool.metadata[owner]=None
            sampling = GREEDY_SAMPLING if greedy else SamplingParameters(
                1., .95, min(20, round_.target.vocab_size,
                             round_.target._sampling.parameters.max_top_k), 1)
            samples = []
            for step in range(3):
                rows = []
                for ordinal_owner,owner in enumerate(owner_ids):
                    pending = round_.pool.metadata[owner]
                    position, root, ordinal = ((0, (17+ordinal_owner)%round_.target.vocab_size, 0) if pending is None
                                              else (pending[1], pending[2], pending[4]))
                    table = tuple(range(ordinal_owner*page_count,(ordinal_owner+1)*page_count))
                    for j in range(round_.stride):
                        physical = ordinal_owner*round_.stride+j
                        p = position + j
                        rows.append(BatchRow(
                            request_id=f"startup:{ordinal_owner}", token_id=root if j == 0 else (
                                0 if pending is None else pending[6][j-1]), position=p,
                            state_source_slot=owner, state_destination_slot=(owner if j == depth
                                else (round_.target._scratch_states[physical] if storage is None else
                                      storage.state_capacity-1-round_.C-ordinal_owner*depth-j)),
                            page_source=table[p // round_.target._block_size],
                            page_destination=table[p // round_.target._block_size], page_table=table,
                            segment_length=round_.stride if j == 0 else 0,
                            sample_ordinal=ordinal+j, mtp_position=p,
                            mtp_page_destination=table[p // round_.target._block_size],
                            mtp_page_table=table, is_decode=True, sampling=sampling))
                result = _execute_pipelined_draft_batch(
                    BatchCommand(step, capacity, tuple(rows)), targets[capacity], providers,
                    hidden, admission, tuple(sorted(targets)), rank=rank, sampler_rank=sampler_rank)
                if result is None:
                    raise RuntimeError("startup warmup did not execute its captured draft transaction")
                if step:
                    samples.append(result.step_ms)
            round_._startup_round_costs[(round_.C * round_.stride, greedy)] = samples
        for owner in owner_ids:round_.pool.metadata[owner]=None
        round_._last_transfer = True
