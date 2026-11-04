# Third-party model adapters

Qwen4Exp PLE table storage can be selected at startup with
`CUBIC_QWEN4_PLE_STORAGE=cpu` (the default). No environment setting is needed:
`bash launch.sh`. Set `CUBIC_QWEN4_PLE_STORAGE=gpu` to keep the table on GPU.
Only lookup-table payloads move to pinned host RAM; projections, convolution,
hash/history state and output features retain their existing GPU execution.
The original packed codes, scales and vocabulary ownership are preserved. GPU
graphs read the selected rows through CUDA mapped pointers, including inside
resident MTP rounds; there is no per-step CPU lookup or full-table GPU copy.
This requires sufficient pinnable host RAM and mapped-host-memory support.
The loader checks the actual execution device before table allocation. If that
device cannot map host memory, it uses GPU storage and prints a `WARNING` to
stdout once per device/process. Allocation or driver-query failures still fail
explicitly. CPU reference execution uses ordinary CPU storage without pinning.
The shared loader exposes this as `LogicalWeight(storage="pinned_host")` for
embedding consumers; format-specific lookup support is still required (plain
FP16/BF16/E4M3 and INT packed PLE paths are supported).
Unrecognized environment values fail at startup. Disk paging is not implemented:
a pinned file mapping still consumes resident RAM, while direct pageable mmap
lookup requires device/OS host-page-table capabilities absent on the local
discrete GPUs. No launch flag or automatic storage fallback is added.

A model family has exactly one directory, named from the checkpoint's canonical
`model_type` (for example `qwen3_5`, not a marketing alias). Aliases and checkpoint
sizes are metadata handled by that adapter; they do not create copied model trees.

## Configuration-driven model support

Adapters describe an architecture, not a known checkpoint repository. Derive layer
counts, dimensions, tensor names and shapes, optional branches, and rank ownership
from the supplied config and checkpoint metadata. Never admit a model by a stored
checkpoint fingerprint, repository path, marketing name, fixed tensor count, or a
complete reference-config equality check. Sample checkpoints are test cases, not
lists of permitted models.

Validate actual architectural relationships and explicit operator capabilities.
An unsupported feature must name the missing capability; it must not be reported
as a mismatch with a local model profile. Tests must vary independent geometry and
TP/PP sizes and still reject missing, extra, or incorrectly shaped weights.
Historical audit receipts cannot qualify a newly loaded instance.

## Model package layout

Keep the core adapter layout small. Model-owned output recipes have one optional
`output.py`; they consume common generation contracts without owning transport:

```text
<model-repository>/cubic_models/<model_type>/
├── __init__.py       # the only public entry
├── model.py          # config semantics, graph, state schema, ModelAdapter
├── weights.py        # checkpoint names, executable slice, TP/PP/EP shard semantics
├── multimodal.py     # optional: modality encoder and language-graph injection
└── output.py         # optional: tokenizer decoder and reasoning/tool grammar
```

- `__init__.py` exports one `MODEL` object implementing `ModelAdapter`. Its
  zero-argument `create_model_adapter()` is only the CLI ABI bridge and returns
  `MODEL`; it must not construct a second implementation or register globally.
- `model.py` normalizes and validates model-specific config, defines the mathematical
  layer graph, declares required semantic ops, and declares KV/recurrent state
  shape, dtype, sharding, and update contracts. It may build a rank-local
  `ModelProgram`, but it does not choose optimized kernels.
- `weights.py` maps checkpoint names to logical parameters, packed shards, and
  column/row/expert axes. It declares the exact executable tensor slice for each
  entrypoint. Generic code owns file discovery, hashing, safetensors I/O, and
  rank-local materialization.
- `output.py`, when present, owns the model-specific tokenizer decoder, output
  markers, reasoning/tool grammar and recipe identity. Shared generation code owns
  the output pipeline; API code owns HTTP/SSE transport and protocol projection.
  Moving a recipe must preserve decoder/parser behavior and its source identity.
  A codec declares its `xgrammar_vocab_type`, optional `tool_model_format` and
  `tool_reasoning` for the shared constraint compiler. The engine does not infer
  a model-specific vocabulary encoding or tool grammar when declarations are absent.
- `multimodal.py`, when present, owns model-specific text/template/tokenizer recipes,
  image/video/audio encoder math, special-token/position rules, feature projection,
  and injection points.
  Generic input code owns URI/byte transport, decoding, request normalization,
  batching, and resource lifetime.

Reuse existing core contracts before adding another abstraction. Small model-local
helpers may hold state geometry or startup bindings; they must not copy runtime
allocation, scheduling, capture, or replay.

## Reference adapter entry

The existing reference adapter ABI remains separate from native serving startup
declarations:

```python
# __init__.py
from cubic_serve.models import MODEL_ADAPTER_API_VERSION, ModelAdapter
from .model import Adapter

CUBIC_MODEL_ADAPTER_API_VERSION = MODEL_ADAPTER_API_VERSION
MODEL: ModelAdapter = Adapter()

def create_model_adapter() -> ModelAdapter:
    return MODEL
```

Load it explicitly; importing a package must never mutate a registry:

```bash
cubic inspect-model \
  --model-code /model-repo/cubic_models/qwen3_5
```

The CLI path is the authority. A directory loads as an isolated package so its
relative imports work without changing `sys.path`; every Python file in the fixed
flat directory contributes its filename and bytes to one source digest. If `MODEL`
and `create_model_adapter()` are both present, the factory must return that exact
object. A single `.py` path remains valid for small adapters. Loader admission
validates the `ModelAdapter` ABI and fails closed on unknown config, weights,
modalities, state, topology, or required ops. External adapters do not require an
edit to Cubic Serve.

## Reusable startup declarations

Shared engine and operator contracts provide the foundation. A model package
declares only its differences and reuses existing parsing, weight mapping, block
semantics and format implementations wherever compatible. It must not copy the
execution engine to adapt a checkpoint.

The immutable `ModelDefinition` groups existing startup functions: `parse_config`,
`logical_weights` and `owner_slices`, with an explicit definition API version.
Each built-in package (`qwen3_5`, `qwen3_5_moe`, `qwen4_exp`) exports its own
`MODEL_DEFINITION`, referencing its original parser and planners with native config
types. Built-in model resolution stays in `models/builtin.py`; runtime resolves
that declaration once and invokes the same logical and rank ownership planners
without its own model-specific weight-planning branches.
Model-owned optional bindings supply frontends, native text constructors, draft
providers, state geometry and startup stage arguments. Compute-dtype semantics
belong to the model parser; format arithmetic and kernel selection remain shared.
Checkpoint I/O, stage execution, qualification, capture, cache ownership and
scheduling stay in their existing common paths.

Derivatives can reuse the entire declaration and override just one changed rule:

```python
from dataclasses import replace
from cubic_serve.models.qwen4_exp import MODEL_DEFINITION as BASE

MODEL_DEFINITION = replace(BASE, parse_config=parse_derived_config)
```

Unchanged functions remain the same objects. A derived parser must still preserve
the architecture and weight contracts of any inherited planners; a new block or
weight layout requires the corresponding semantic change and qualification.

Quantization formats continue to use the existing `QuantizationExtension`
contracts independently of model identity. Shared semantic operations use
`OpSignature` and the existing reference catalog. New format or attention/block
implementations must enter those shared contracts with an independent reference
and qualified native execution, so compatible model derivatives inherit them.

### Explicit native serving adapter

By default, `serve` uses the project's built-in support for `config.model_type`.
It never discovers or executes Python code from checkpoint configuration. Override
that default only with an explicit startup path:

```bash
cubic serve /model-repo --model-adapter /model-repo/cubic_models/my_model
```

That package's `__init__.py` exports `MODEL_DEFINITION`. A compatible derivative
can inherit everything without an engine copy:

```python
from cubic_serve.models.qwen4_exp import MODEL_DEFINITION
```

Use `dataclasses.replace` only for changed semantics, as shown above. A single
`.py` entry also works. The loader snapshots relative Python files recursively;
each loaded instance has its own package namespace, including worker processes
and lazy relative imports. The admitted source digest and config recipe bind the
execution identity. Source ownership lasts through all workers and encoders and
is released after they close. It does not mutate a global registry or `sys.path`.

Declarations may reuse an existing native stage only when their model semantics
meet that stage's contracts. A new attention block or quantization format needs
its corresponding shared operator and independent reference implementation; the
loader does not make unsupported computation qualified. A non-dataclass config
must supply a canonical `program_recipe`. The existing reference-only
`ModelAdapter` API version 1 and `inspect-model --model-code` remain unchanged;
`--model-adapter` is the native `serve` entry, not an eager-mode override.

## Forbidden ownership

Model adapter code must not implement or control:

- scheduler/runtime policy, CUDA Graph capture or bucket selection;
- process groups, collectives, pipeline transport, or cache/state allocation;
- KV/prefix page ownership, COW, commit, rollback, eviction, or admission;
- HTTP, OpenAI/Anthropic payloads, SSE, or request orchestration;
- hardware dispatch, autotuning, `torch.compile`, or device-specific branches.

The model declares semantics; common runtime code owns resources and execution.
An executable weight manifest must close exactly over its graph: text cannot silently
admit unused vision or MTP tensors, and multimodal cannot silently omit its encoder.

## New operators

An absent semantic operation is not embedded in `model.py`. Add the smallest shared
pair under:

```text
src/cubic_serve/ops/eager/<operation>.py       # exact/reference implementation
src/cubic_serve/ops/cuda/<domain>/<operation>.py  # SM80-SM120 candidate
```

Give both implementations the same `OpSignature`. The adapter requests that
signature; the executor selects the implementation. CUDA code requires an eager
oracle, full-output/state comparison, declared reduction-order tolerance only when
unavoidable, and padding-poison coverage before it can enter a performance graph.

## Qualification

Every adapter change must pass, in this order:

```bash
cubic inspect-model --model-code /model-repo/cubic_models/<model_type>
PYTHONPATH=src venv/bin/python -m unittest discover tests/models/<model_type>
PYTHONPATH=src venv/bin/python -m unittest discover tests/guardrail
```

The model suite must prove config rejection, complete executable-weight closure,
rank-local TP/PP/EP ownership, eager prefill/decode correctness, multimodal injection
when declared, and exact cold-versus-prefix-resume KV/recurrent state. Run focused
real-CUDA operator tests only for new CUDA ops. A performance adapter is accepted only
after its retained end-to-end gain passes the same oracle/state checks, multi-M graph
padding poison with no fallback, and the public streaming Chat Completions acceptance
suite for every claimed topology.
