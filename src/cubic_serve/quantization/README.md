# Quantization extensions

A quantization implementation has one owner package. Checkpoint layout, mathematical
reference, CUDA implementation, expert computation and parameter policy live together.
Model code and common operator dispatch do not enumerate checkpoint formats.

## Add a package

Export an immutable `QuantizationExtension` from `extension.py` (or `__init__.py`).
Use a versioned encoding ID: it describes the payload and arithmetic, not merely a
marketing name or exporter. Implement the applicable contracts in `api.py`:

| Module | Owns |
| --- | --- |
| `checkpoint.py` | Exporter declarations, scheme validation, activation conventions |
| `layout.py` | Payload headers, packing alignment, slicing and scalar/global ownership |
| `reference.py` | Independent decoding and activation reference math; no candidate kernels |
| `linear.py` | Native linear/embedding bindings, private scratch and backend choice |
| `moe.py` | Expert operand layout, input preparation and routed arithmetic |
| `policy.py` | Parameter dataclasses, defaults and override validation |
| `kernels.py` or `kernels/` | Package-private implementations, including backend subdirectories |
| `tests/` | Scheme-specific proofs; cross-format contracts also run from `tests/ops/` |

These are modules, not required inheritance hierarchies. Simple implementations can
share existing helpers or re-export a function from another module in their package.
`parse_config` is required for declared checkpoint methods; `parse_group` for declared
group formats. Identity-only schemes need no activation quantizer. Reference execution
never imports or calls an optimized backend. Metadata and policy loading must not
initialize CUDA.

Register an installed package using the standard entry-point group:

```toml
[project.entry-points."cubic_serve.quantization"]
my_format = "my_package.quantization.extension:EXTENSION"
```

Discovery runs in each worker, in a deterministic order. This transports the extension
to spawned TP/PP/EP workers without serializing callables or changing model code.
Embedding a package directly may call `registry.register()` before constructing
checkpoint/config/execution objects; embedded registrations must be installed in each
worker. Conflicting names, encoding IDs, exporter methods and group formats are rejected;
registration never replaces an existing implementation. The builtins list lives only
in `registry.py`.

A grouped compressed-tensors declaration is container syntax. Its group format resolves
to the registered extension, including mixed-format models and embeddings/lm_head/MTP.
Adding a future GGUF reader does not make GGUF one arithmetic scheme. Unsupported
containers remain rejected; no empty GGUF/GPTQ implementation is advertised.

## Execution and reuse

Common bindings own operand preflight, retained compiled artifacts, launch receipts and
static scratch lifetime. Format packages may supply their own native linear binding;
they do not have to use the shared weight-only Triton template. That template accepts a
format-owned loader rather than an integer selector over all supported encodings.

Common MoE code owns routing, compaction, expert ownership and combination. Each extension
supplies a `RoutedOperands` value: arbitrary typed pointer columns, format-local immutable
math constants and a routed compute callback. The route traversal does not know packed
weight suffixes, FP4 levels, Cubic curves or activation encodings. Do not change logical
expert ownership or reduce activation scales over unrelated experts.

The existing packed Triton template has a shared tile parameter schema. Packages may
reuse its validator, or supply their own parameter subclasses and validation, including
new integer/boolean/finite-float/string controls. Linear and expert parameters remain
separate families. Semantic boundaries such as NVFP4's ordered group-16 accumulation
are not tuning controls. Config transport remains `OperatorConfig`, with existing family
names and defaults preserved.

Triton callbacks in artifact identities use their source-closure cache keys; process
addresses cannot identify an implementation. Registration/binding happens before capture.
Graph qualification, topology-wide leases and public serving admission stay with the
engine. A registered extension or passing primitive test does not qualify serving.

Marlin's integer repacker, CUDA 13.0 compiler and vendor pipeline live once in
`common/marlin/`. Format-owned `marlin.py` modules adapt AWQ W4A16 expert gate/up and down,
FP8-token linear and NVFP4-local expert projections. AWQ retains original FP16
scales and execution-dtype weight rounding. FP8 retains original token/channel
FP32 scale boundaries. NVFP4 retains each original expert/projection global scale
and an ordered FP32 addition of each independently formed K16 partial. Native
eligibility follows the tensor ABI and shared-memory resource equation; unsupported
native layouts bind the format's existing CUDA implementation before capture.
There is no eager or full-dequantization fallback during graph replay.

These Marlin bindings use **WNA16 compute mode**: low-bit weights feed 16-bit
multiply operands and FP32 accumulation. FP8/NVFP4 activation quantization still
follows the checkpoint contract before its exact FP16 representation is prepared;
this is not native W8A8 or W4A4 execution. Keep storage encoding, activation
quantization and actual instruction precision separate when naming or pricing a
backend. Native low-precision compute modes require their own measured costs and
qualification; none is implied by a checkpoint's bit width.

### Compute modes and native availability

Storage encoding, activation arithmetic and performance-kernel availability are
separate capabilities. `supports_compute_mode` describes format-owned arithmetic;
it does not qualify a native operator or a serving configuration. A requested
WNA8 projection must fail explicitly when its native implementation is unavailable.
It must never silently execute WNA16 or a Triton core GEMM in performance mode.

INT W2–W8/F16 dense and routed WNA8 projections use the shared native Marlin path.
The INT package owns signed packed codes, group scales and asymmetric zero-point
correction. GGML WNA8 uses native MMQ for its supported original tensor encodings
and preserves Q8_1 block FP16 scale boundaries. These two activation contracts
are different; the engine must not impose one format's quantizer on another.

Cubic WNA16 uses format-owned native Marlin bindings when eligible. Cubic WNA8
has independent reference math and a legacy Triton compatibility implementation,
but no admitted native performance binding. Its ordinary and grouped native
hooks reject WNA8 before preparing expanded weight carriers. This is an explicit
capability gap, not a qualified WNA8 mode or permission for mixed precision.

The legacy compatibility path requantizes original Cubic W2–W8 values to one
INT8 scale per output channel, with `N*K + 4*N` bytes of prepared storage. It
uses overflow-checked INT32 K reductions and preserves FP32 channel/token scale
boundaries before the output cast. Its existing numerical and ownership fixtures
remain explicit compatibility tests; they do not qualify performance serving.
Its partition-supply heuristic depends on N/K, SM count and configured limits,
but parameterization alone does not establish native-kernel availability or
cross-device performance. Real-device measurements remain local to SM120.

Preparation reuses common encoded-byte transport. Grouped INT experts can join
one original operand at a time directly on device, retaining original owners and
final Marlin buffers rather than a CPU concatenation. The format proves packed
word closure and metadata arithmetic; unsupported grouping/padding geometry keeps
its existing WNA8 preparation. This does not change expert order or native GEMM.

Reference preparation uses a separate optional `prepare_wna8_linear` hook. A
format owns activation production and original decoding; the shared executor
batches whole canonical output tiles within an explicit byte budget. Dot shapes,
rounding boundaries and independent reference provenance remain unchanged.

NVFP4 stores its prepared scales as lossless Marlin S0E5M3 carriers. The FP32
epilogue restores the fixed power-of-two bias before the original global-scale
divisions. No small-scale clipping or data-dependent rebias is used. Weight codes
remain packed; tile-local conversion does not materialize a decoded weight matrix.

Native launch controls are `marlin_enabled`, `marlin_block_m/n/k`, `marlin_warps`
and `marlin_stages`, transported through the existing format parameter families.
NVFP4's prefetch K tile is separate from its immutable mathematical K16 group;
its N tile must cover all warps (`N = 64 * warps`) to avoid changing the ordered
partial reduction. These are baseline launch parameters and resource constraints,
not a claim of a calibrated optimal selector for every shape or SM80–SM120 device.
Actual device qualification and complete model audits remain required.

A format may expose `bind_grouped` and `bind_routed` from `moe.py`. The common
executor provides fixed-owner `RoutedRows`; the format returns an immutable binding
and its row tile. `bind_routed` consumes route-indexed activation rows, while grouped
gate/up consumes token rows with an explicit top-k mapping. Neither hook changes
routing or expert ownership. Optional `linear.bind_native` binds a format-owned device representation before
generic materialization/calibration, so native launches are not benchmarked again
under unrelated Triton tile controls.
Optional `linear.bind_grouped(operators, x, weights, outputs, params)` handles
model-declared projections of one unchanged input. Common dispatch partitions a
mixed declaration by encoding and resolved parameter identity, then binds each
compatible group before remaining individual projections. Returning `None` keeps
that group independent. This avoids allocating both grouped and individual packed
representations during startup; plain members reuse the existing BF16/FP16 grouping.

FP8 exposes `fp8_linear.grouped_projections`. Its Marlin group concatenates original
integer codes and FP32 channel scales, quantizes each input row once, and scatters
the final scaled results to the original outputs. Each member must satisfy the
existing K/N tile alignment and shared-memory eligibility; there is no model,
concurrency or device-name selector. Scales are never merged by taking a maximum
or rescaling the checkpoint. Row-parallel packed projections retain their individual
complete-K scale producers and topology collectives. Every destination retains
its original dtype boundary, immutable binding identity and private scratch ownership.

A reusable binding shares only its first scratch tensor within one sealed dispatch;
all additional buffers remain private and are included in graph ownership.

Ordinary packed linear and routed MoE reuse the common resource geometry generator
and bounded cold calibration. Rank-local dimensions, estimated local route population,
device resources and actual compiled operator cost select M/N, warps and stages;
format K reduction quanta and all scale/rounding boundaries stay fixed. Existing
`launch` search limits and each format's explicit tile overrides remain authoritative.
Only a compiler resource-limit rejection removes a cold candidate; numerical failures
remain errors. Calibration uses private inputs, outputs, scales and routing metadata,
never live requests or traffic-driven expert reassignment. New independent backends
can opt out through their parameter class's `generic_geometry` capability; Cubic
linear does so because its existing decoder-aware placement policy already owns
selection. This capability is separate from serialized tuning values.

Native Marlin geometry uses one shared cost model, with coefficients calibrated
from the actual format kernel and device rather than a model/concurrency table:

```
CTA waves = ceil(row_tiles * (N / tile_N) / SM_count)
working bytes = active_experts * packed_weight_bytes + rows * (2*K + output_bytes*N)
time = launch_cost + wave_cost * max(0, CTA_waves - 1)
       + memory_cost * max(0, working_bytes - usable_L2)
```

Routed row occupancy is estimated from the fixed expert population; the selector
does not inspect traffic or change expert ownership. Bounded cold fixtures fit
nonnegative coefficients and usable cache capacity, while separate shapes measure
prediction error. A candidate replaces the baseline only when the uncertainty
bounds and configured minimum gain separate at both the requested population and
its lower neighbor. Explicit format-family overrides remain authoritative. This
governs native geometry, not an assertion that every Triton/Marlin backend choice
already comes from one cross-backend regression. Triton decoding costs remain
format-owned; the common fallback uses bounded resource candidates and cold timing.

Current checkpoint coverage is determined by actual tensor encodings. AWQ and
NVFP4 expert gate/up and down use native Marlin when eligible. Mixed FP8/NVFP4
also binds FP8 ordinary/grouped projections through Marlin. Cubic MoE supports
native expert projections for its normalized metadata representations. Cubic
dense WNA16 dispatch uses its format-owned planned native binding. Compatibility
implementations and old primitive fixtures do not establish performance readiness.
Plain BF16/FP16 members retain the ordinary linear policy.

Stacked expert tensors are model-declared storage views, independent of
quantization method. The model declares [E,N,K], expert index, projection row spans,
fused output partitions and semantic configuration aliases. Admission closes logical
row coverage exactly once for every physical payload, including per-projection scales.

A format supporting stacked serialization supplies the optional layout function
`stacked_payload_name(source, key)`. Existing `payload_headers` describes one complete
expert matrix; admission prepends E, and existing `slice_index` composes the projection
view with TP ownership. Cubic supplies its underscore field names here. Missing or
partially mixed serialization fails closed. Direct checkpoint reads and rank-owner
loading share these coordinates; packed bytes and metadata dtypes remain unchanged,
while plain execution tensors follow the requested FP16/BF16 dtype. No model path,
version, bit-width label, or expert traffic controls loading.

A Cubic input shard wholly contained in one original quantization group may retain
that group's metadata verbatim: every local column indexes local group zero. A shard
crossing group boundaries still requires aligned boundaries, and packed input starts
must remain byte-aligned. The encoding's group size is never rewritten to fit TP.

## Validation

`serve` defaults to `--execution-mode cuda`. Packed performance startup
uses the actual retained graphs for independent original-weight/coherent-producer
qualification, cross-M row permutation and padding checks, cache/state migration,
and prefix forks before readiness. One physical M1 graph and power-of-two anchors
cover the admitted range. The full rank portfolio, including interleaved child
buffers and compute/communication streams, holds one exclusive replay lease until
durable output/state consumption. Missing evidence fails admission; there is no
runtime eager fallback or graph capture. This is fresh qualification on every
startup, not a reusable receipt cache. The measured four-GPU portfolio currently
takes roughly 7–20 minutes to qualify before readiness; it is not an instant
startup path.

The independent oracle can initialize BLAS even when native/Triton serving has
never called it. Its library context and workspace survive ordinary allocator
cache cleanup. Initialize that existing reference path with bounded synthetic
operands before pricing the KV pool; do not expand checkpoint weights or add a
model-specific memory surcharge. The final post-audit reserve check remains
authoritative. SGLang's pinned `utils/common.py:4709` similarly initializes BLAS
with a small matmul; vLLM's pinned `gpu_model_runner.py:6618` materializes execution
and sampling before completing memory profiling.

The packed numerical admission covers BF16 execution, PP1, and legal EP factors
of TP, with attention/activation `auto` and softmax `fp32`. Expert matrix shards
reduce over ETP=TP/EP; owner-local routed outputs complete over EP. Both the
shared text graph and the legacy MoE witness use these topology groups instead
of an EP1/EP=TP whitelist. Each startup still qualifies its actual checkpoint,
format, graph buckets and topology independently. Other arithmetic policies
are rejected as unqualified before loading, rather than misreported as numerical
failures of the low-precision oracle. Explicit FP16 is never silently replaced with BF16: use the independent
`--execution-mode eager` path for that unqualified graph dtype, or explicitly
request `--dtype bf16` (including for AWQ repositories declaring FP16). Quantized
MTP remains rejected. Passing BF16/FP16 primitive tests does not qualify FP16
model graphs, PP, MTP or an unmeasured device/topology. Explicit `debug`
remains available for pre-qualification diagnostics and is not a performance seal.

`tests/ops/test_quantization_extensions.py` constructs an external installed package and
checks discovery, checkpoint resolution, slicing, independent reference math, policy
transport and collision rejection. The packed operator suites exercise all four builtins
in FP16/BF16, including graph replay, padding and routed MoE. FP8 conversion proofs live
in `extensions/fp8/tests/` and retain the existing root test collection entrypoint.

Pinned structural references: vLLM `89af4f1ff792199b7c9260dd4a86feed445fd3d6`,
`vllm/model_executor/layers/quantization/{__init__,base_config}.py`; SGLang
`ba433bb4628f44d5031e910ad32194ad6c145ba8`,
`python/sglang/srt/layers/quantization/{__init__,base_config}.py`; llama.cpp
`458681e1d5d4a29a1463c4732e03226cf384b997`, `ggml/src/ggml.c` type traits.
Those sources demonstrate registry/type-owned implementation boundaries. Cubic's strict
no-replacement registration, explicit graph admission and independent reference audit
are intentional local constraints, not claims of identical upstream architecture.

## Pre-qualification serving test entry

`serve --execution-mode debug` connects the registered formats to the shared
continuous-batch scheduler, batch-fill/sampling path and streaming Chat Completions API.
It is an explicit debugging entry, not a numerical or performance qualification.
Model forward uses only startup-captured power-of-two buckets, including physical M1;
there is no runtime capture or eager fallback. The default `cuda` entry
additionally requires the independent cold qualification described above; a
successful `debug` run does not satisfy that admission gate.

```sh
PYTHONPATH=src ./venv/bin/python -m cubic_serve.cli serve \
  /pubdata/llm_models/Qwen/Qwen3.8-27B-Cubic-4Bit \
  --execution-mode debug --devices 0,1,2,3 \
  --tp-size 4 --pp-size 1 --ep-size 1 \
  --max-running-requests 256 --host 0.0.0.0 --port 30016
```

For MoE, select its checkpoint path and the intended expert group size, for example
TP4/PP1/EP2. The local AWQ repository declares FP16: this BF16 debug entry
requires an explicit `--dtype bf16` override for it; `auto` does not silently change
the checkpoint dtype. Use native eager execution when FP16 is required. Single-card uses `--devices 0 --tp-size 1 --pp-size 1 --ep-size 1`;
TP2/PP2 uses four devices. EP divides TP and does not need to equal it. Capacity remains
subject to the actual resident weights, graph allocations and BF16 KV/state pool; reduce
the admitted concurrency when a single-card topology cannot hold that pool.
An omitted context limit is sized from available memory and capped by the repository
limit. Explicit limits that cannot fit are rejected.

Before allocating the final KV pool, debug measures the complete graph and
rank-staging portfolio against a temporary arena, using the declared context width
for page tables. It releases that arena and portfolio, charges their measured
non-cache residency and construction transients separately from the runtime reserve,
plus the observed allocator slack that cannot be assumed reusable after replacing
the arena, then takes the minimum page capacity across ranks. Final graph bindings reconstruct
the selected overlap partitions without rerunning selection under a smaller memory
budget. Arena allocation closes the payload estimate against its actual device
footprint; a measured deficit removes the corresponding number of pages on all
ranks before graph construction, including observed allocation rounding so a
smaller payload does not remain on the same allocator plateau. Readiness also
checks that the final runtime reserve remains available.
This uses actual operator/topology allocations; M1 workspace size is not an estimate
for all larger buckets, and successful allocation is not numerical qualification.

Current debugging scope: BF16 compute/KV/recurrent state, text, MTP0, shared sampling,
chunked prefill and request cancellation. FP16 graph serving, multimodal execution and
MTP are not advertised by this entry. `--operator-config` and the existing attention,
activation and softmax dtype controls are forwarded unchanged. Default math remains
attention=auto, activation=auto, softmax=fp32. Public SSE timing from this entry is an
unqualified diagnostic; it does not grant readiness to `performance` mode.


Cubic linear parameters include `fixed_split_k` (deterministic FP32 partial
reduction) and a bounded output-tile work estimator. Starting from
`fixed_block_m`/`fixed_block_n`, candidates double the row tile through
`fixed_max_block_m` and grow the column tile proportionally through
`reuse_block_n`. `fixed_warps` applies to the base tile; `reuse_warps` to larger
tiles. At cold binding, the estimator counts device waves of weight decoding,
input loading and MMA work, with tunable `decode_scalar_ops` and
`mma_scalar_ratio`. Actual compiled registers, shared memory and spills then
penalize resource-heavy candidates. These are estimates, not claims of universal
optima. There are no per-model, per-C or architecture timing tables, runtime
autotuning launches, or runtime graph captures. Every candidate keeps the same
block-K, split-K and ordered reduction. Other formats, plain linear and routed
MoE retain their own policies. Equal minimum/maximum row tiles force the base
tile for a controlled comparison.

`cubic_linear.level_table_max_bits` defaults to 4; 0 disables prepared levels.
Compact Cubic weights with 2 through that many bits prepare their exact
dtype-rounded per-group magnitude levels once. Replay unpacks the stored codes,
looks up a magnitude and applies its sign bit, keeping the existing GEMM and
ordered FP32 split reduction. The FP16 coefficient and final weight rounding
boundaries are preserved. Tables are keyed by source matrix, payload versions
and compute dtype, owned by the resident model and shared by its graph buckets;
there is no global cache. Original packed payloads stay intact. Preparation
expands group codebooks rather than full matrices, and increasing the bit limit
increases table memory exponentially. This option affects linear projections;
embedding and routed MoE keep their independently qualified paths.

Split-K workspace follows temporary lifetime, not layer count. A sealed
`PackedProjectionDispatch` owns a private pool sized to the largest opted-in
temporary of each device/dtype. Serial Cubic linear occurrences reuse that pool
only after each occurrence has consumed its partials. Binding and pool growth
finish before the first launch or capture; replay addresses remain fixed.
Outputs and persistent state are excluded. Separate dispatches, graph bodies and
execution lanes never share this pool, and other formats must explicitly qualify
their scratch lifetime before opting in. Allocating permanent split-K partials
for every layer and bucket can otherwise exhaust memory before request admission.

Cubic W4 compact metadata additionally exposes `level_word_lookup` (default true)
and `row_reuse_max_tiles` (1 or 2, default 2). The first loads each group's
eight rounded magnitudes as four packed words before extracting weight indices;
bit selection preserves the exact BF16/FP16 representation. The second lets the
existing work estimator consider sharing one decoded weight tile between two
independent output-row accumulators. Compiled register, shared-memory and spill
costs participate in selection; K partitioning and ordered FP32 merging do not
change with M. The widest configured tile is extended only when real input rows
can reuse it. Other formats and fused MoE retain their own strategies. Set
`level_word_lookup=false` to compare with the direct magnitude-table path, or
`row_reuse_max_tiles=1` to isolate row reuse. These controls do not grant a
quantized performance-mode seal or imply qualification on unmeasured devices.

### Cubic decode selection and resource limits

`extensions/cubic/policy.py` owns cold decoder selection; `linear.py` binds the
selected implementation. Encoding semantics are not tuning parameters. W1 uses
bipolar codes; W2-W8 use the existing signed-code/reserved-zero rule. Prepared
levels currently support only compact A/B metadata and half-sized elements;
curve2 and other metadata retain direct decoding. The four-word loader is a W4
kernel capability, not a threshold that can be raised to enable W6/W8. Its
group/K-tile alignment fallback remains in the loader. A metadata or resource
fallback selects an operator before capture, never an eager serving fallback.

Two optional `cubic_linear` controls limit preparation **before allocation**:

| Parameter | Meaning | Default |
|---|---|---|
| `level_table_max_bytes` | Maximum table bytes per rank-local matrix and compute dtype | 0: no additional limit |
| `level_table_max_packed_ratio` | Maximum table bytes / packed code bytes, excluding original metadata | 0: no additional limit |

For local N/K, group sizes GO/GI, bit width B and element bytes E, the estimate is
`ceil(N/GO) * ceil(K/GI) * 2**(B-1) * E`; packed code bytes are
`N * ceil(K*B/8)`. Either exceeded limit selects direct decoding. Limits are
independent of `level_table_max_bits`, apply to the shared weight-owned table
rather than each graph bucket, and are not a model-wide memory allocator.
Existing defaults preserve decoder and tile selection. For example, setting
`level_table_max_bits=8` with `level_table_max_packed_ratio=1.0` permits only
implemented tables no larger than their local packed code payload; it does not
assert that those tables are faster. W4/W6/W8 need 8/32/128 half values per group,
respectively. Loading all wider tables into registers has no qualified speedup.

Tune the existing tile/warp/reuse and decode-work parameters separately from
these memory bounds. The work coefficients are estimates, not bit-width-specific
measurements. Increasing warp count does not necessarily distribute useful rows:
check the compiled MMA layout against backend instruction shape and M/N tile
coverage, then registers, spills, shared memory and conversion cost. An MMAv2
64-row experiment assigned eight warps along M, covering 128 rows. Explicit
balanced-layout prototypes did not beat the retained implementation and were
not integrated; the current resource score does not explicitly repair this
coverage problem. Do not extrapolate W4 results to other widths or SM families.

Source triangulation for this boundary: vLLM-cubic
`89af4f1ff792199b7c9260dd4a86feed445fd3d6`,
`vllm/model_executor/layers/quantization/cubic.py:574` explicitly restricts its
exact Marlin preparation to W4/G32 and additional dtype/layout constraints;
SGLang `ba433bb4628f44d5031e910ad32194ad6c145ba8`,
`python/sglang/kernels/ops/quantization/awq_triton.py` separates AWQ packing,
group scales and launch tiles (not a Cubic implementation); llama.cpp
`458681e1d5d4a29a1463c4732e03226cf384b997`,
`ggml/src/ggml-cuda/mmq.cuh:1385` derives launch/shared-memory requirements from
the format/configuration (no equivalent Cubic table policy identified).
These are implementation facts, not numerical or throughput oracles. The shared
lesson is to validate format/layout capabilities before selecting a resource
strategy; upstream specialization tables are not imported into this baseline.

Device-specific measured configuration storage/dispatch is deferred. Future
measured configurations should supply this same parameter contract; no model,
concurrency or SM timing table is introduced here. Qualification remains tied
to the actual tested device, tensor geometry and numerical execution contract.

### AWQ grouped W4A16 projections

The AWQ extension owns the Marlin-derived source, startup code repacking and
native binding. The shared routed operator supplies explicit rank-local row
metadata through `RoutedRows`; it contains no AWQ or Marlin selection branch.
Original expert order, FP16 checkpoint scales, execution-dtype weight rounding
and whole-K reduction ownership are preserved. Repacked codes remain quantized
and share the resident checkpoint cache across graph buckets.

`awq_moe` exposes a separate native launch ABI:

- `marlin_enabled` (true) permits the native grouped gate/up binding.
- `marlin_block_m/n/k` (16/64/128), `marlin_warps` (4), and
  `marlin_stages` (4) describe its compile geometry.
- Existing `fixed_*` parameters continue to describe the Triton binding.
- `grouped_projections=false` disables grouped projection execution.

The native binding checks tensor divisibility, scale dtype, quantization groups
and shared-memory capacity before startup binding. Group blocks derive from
checkpoint metadata; shared memory derives from the activation/weight pipeline,
reduction temporary, scales/zeros and row metadata. No device/model/concurrency
timing table is used. These defaults and resource checks are not a calibrated
cost model or qualification for every exposed geometry.

CUDA 13.0 compiles an SM80 PTX baseline at startup; `CUDA_HOME` can select the
baseline toolkit installation. Compilation is content-cached and locked across
processes. Captured execution uses sealed native launches with actual linked
cubin identities. Sources and license notices ship with the package; no adjacent
vLLM checkout, prebuilt vLLM library or decoded full-weight matrix is required.
This runtime compilation path does not distribute AOT binaries or imply
qualification on untested devices. Graph-debug evidence remains distinct from
performance-mode admission and complete model numerical qualification.

### Decode compute/communication overlap

The debug stage can prepare an interleaved decode graph before readiness.
The model body exposes its existing collective boundaries as an iterator; normal
execution consumes that same iterator in order. Interleaving does not duplicate
model arithmetic or alter the ascending-rank reduction. Each child owns its
activation/collective scratch, while parent row metadata supplies disjoint cache
and padding ownership. Prefill, mixed segments, reset and COW stay on the original
startup-captured graph. Runtime selection never captures a new graph or executes
eager code.

A linear binding may declare `row_tile_count` only when its rows are independent
and it contains no hidden communication. Undeclared bindings conservatively
exclude the stage from overlap. Dense TP stages with PP1 currently consume this
capability; routed expert and pipeline dependencies are not qualified by this
implementation. FP8 scale collectives must not be moved implicitly across these
boundaries. This is an operator capability boundary, not a format-name selector.

`collective` parameters bound startup search:

- `decode_overlap_max_parts` (4): candidate-count ceiling; 1 disables overlap.
- `decode_overlap_samples` (3): timing samples per candidate.
- `decode_overlap_min_gain_percent` (5): minimum improvement over the original graph.
- `decode_overlap_max_bytes` (4 GiB): additional live allocation budget per bucket,
  including the currently retained candidate during comparison.

Candidates split the largest power-of-two part, preserving total physical rows.
For example, three parts can use M/2 + M/4 + M/4 without inventing a padded M/3
capacity. Splitting stops before children become smaller than the largest bound
row tile, or when resource limits exclude another candidate. Timings measure the
complete captured execution and use the slowest rank's median; all ranks select
the same plan. No device, model or public-concurrency lookup table is used.
Startup timing is diagnostic selection evidence, never serving qualification;
selected variants still require independent eager/reference, cross-M state and
public streaming qualification. Graph-debug remains explicitly unqualified.

Source comparison: vLLM `89af4f1ff792199b7c9260dd4a86feed445fd3d6`,
`vllm/v1/worker/ubatching.py` shares compute/communication streams and event
boundaries (two batches by default); SGLang
`ba433bb4628f44d5031e910ad32194ad6c145ba8`,
`python/sglang/srt/batch_overlap/two_batch_overlap.py` preserves sequence splits
and treats small-batch overhead separately; llama.cpp
`458681e1d5d4a29a1463c4732e03226cf384b997`,
`ggml/src/ggml-cuda/allreduce.cu` separates copy-engine and compute work for its
two-GPU transport, with no matching generic multi-body graph planner identified.
The common invariant is explicit dependency ownership; their transport, routing
and reduction rules are not interchangeable with Cubic Serve's canonical sum.

Original encoded checkpoint payloads are immutable host snapshots, independent of
the selected device representation. Plain execution tensors still load directly to
the requested GPU/dtype. Native factories run before fallback operand construction;
they may stage bounded integer payloads and retain only their prepared GPU layout.
When a native factory declines, `materialize_weight` caches the original encoded GPU
layout for the existing CUDA backend. These operations occur before capture, never
inside graph replay. The independent reference moves bounded encoded tiles to its
computation device before decoding, preserving the original reference arithmetic.
`named_tensors` includes original and prepared storage for inspection; GPU residency
counts unique GPU allocations and does not count the host reference copy.

Packed shared/routed expert branches reuse `MoeBranchStreams`. The launch policy
`shared_expert_overlap` can disable this fork/join for diagnosis. Eligibility
requires a routed branch with no internal collective and disjoint writable
projection/expert scratch. The post-norm input remains read-only; each branch
retains its own output, original scale/reduction boundaries and checkpoint expert
ownership. Join precedes output collectives, combination and buffer reuse. These
dependency checks contain no model, device or concurrency thresholds; they are
not a claim of performance benefit on every eligible shape.

Upstream comparison: vLLM `89af4f1ff792199b7c9260dd4a86feed445fd3d6`,
`vllm/model_executor/layers/fused_moe/runner/shared_experts.py` supports an auxiliary
shared-expert stream with backend/dependency restrictions. SGLang
`ba433bb4628f44d5031e910ad32194ad6c145ba8`,
`python/sglang/srt/models/cohere2_moe.py` explicitly protects shared input from
in-place expert mutation before fork/join. Cubic's routed input is read-only and
does not require that clone. llama.cpp
`458681e1d5d4a29a1463c4732e03226cf384b997`, `src/models/qwen35moe.cpp` constructs
separate routed/shared computations and then combines them; this source does not
establish a matching CUDA multi-stream policy. Serving and numerical evidence are
qualified separately from these source invariants.

Marlin linear and expert bindings share `common.marlin.compiler.bind_launch`.
The grid is `min(output_tile_capacity, SM_count * resident_CTAs_per_SM)`;
residency uses the loaded kernel's register and shared-memory allocation through
the CUDA driver, rather than an architecture table. `marlin_blocks_per_sm=0`
selects this resource bound; a positive value caps it for controlled comparison.
Routed tile capacity is only an upper bound: device route metadata continues to
determine actual work and checkpoint ownership. The rule preserves whole-K tile
ownership and format-specific reduction order. It does not establish optimal
M/N/K geometry, account for communication stalls, or imply device qualification.
Direct FP32 output must retain the shared row-map lifetime barrier before tile
reuse, even though it bypasses shared output reshuffling.

Pinned comparisons: vLLM `89af4f1ff792199b7c9260dd4a86feed445fd3d6`,
`csrc/libtorch_stable/moe/marlin_moe_wna16/ops.cu`, and SGLang
`ba433bb4628f44d5031e910ad32194ad6c145ba8`,
`python/sglang/kernels/jit/csrc/gemm/marlin_moe/moe_wna16_marlin.cuh`,
also bound CTA supply using registers, shared memory and available work; their
split-K and geometry policies are not imported. llama.cpp
`458681e1d5d4a29a1463c4732e03226cf384b997`, `ggml/src/ggml-cuda/mmvq.cu`,
uses a different shape/type-based row/warp policy rather than this Marlin grid.

Shared/routed branch planning uses `ops.cuda.launch_cost.WorkCost`:
`T = launches * startup + max(F / compute_rate, B / memory_rate) / supplied_fraction`.
The serial estimate is `T(shared) + T(routed)`. The parallel estimate is the
maximum of each branch's critical path and their **summed** compute/memory demand,
plus `launch.branch_fork_join_launches * startup`. Full-device rates cannot be
assigned independently to both concurrent branches. Disjoint writable scratch,
read-only shared input and absence of internal routed collectives remain hard
prerequisites; a lower estimate cannot override them.

Cold-start coefficients are measured on the current device and activation dtype.
The memory probe exceeds L2, and repeated work inside one temporary graph avoids
charging host event-submission gaps as single-kernel latency. Only scalar
coefficients survive calibration. Tensor geometry and actual encoded payload
bytes provide the work estimate. Expected active experts use the uniform routing
prior `E * (1 - exp(-M * top_k / E))`; actual route metadata still controls
execution and checkpoint ownership. This is a planning prior, not a traffic
balancer or an exact predictor of skewed routing. The branch estimate counts
logical operator stages (four shared, six routed), not a complete kernel trace.

Native tile selection uses the shared `common.marlin.cost` regression, rather
than extrapolating a dense GEMM roofline to a quantized decoder:

```
T = intercept + issue_cost * max(ceil(row_tiles * (N / tile_N) / SMs) - 1, 0)
              + cache_cost * max(working_bytes - usable_L2, 0) / 1e8
```

Each candidate runs the real compiled native kernel on private finite operands
at a small set of **independent calibration shapes**, derived from SM supply,
matrix geometry, expert count and cache capacity. Actual register/shared-memory
residency governs its launch; native instruction, decoder and pipeline costs
therefore enter the fitted coefficients. Simultaneous residency does not grant
each CTA its own SM issue units. Usable cache capacity is fitted within a bounded
fraction of physical L2, accounting for imperfect cache residency. The `1e8`
normalization conditions the regression and is not a workload threshold.

Nonnegative coefficients are fit once per device/native arithmetic/geometry and
matrix shape. Bounded reweighted least squares balances relative errors across
sparse and saturated loads; both training residuals and independent sparse/cache/
tile-boundary holdouts enter the rejection bound. A small absolute timing error
must not hide a large relative C1 error. Only scalar models are retained;
there are no checkpoint replicas, live-state probes or shipped SKU/model/C tables.
The requested M is evaluated by the formula, not timed to choose its winner.
For routed work, a uniform Poisson prior estimates occupied experts **and the
probability of additional row tiles beyond the mean**. This is not a traffic
balancer; actual device route counts and ownership remain authoritative.

Candidates preserve prefetch K and the N/warp ratio defining K reduction
ownership. They vary row tile, N/warp supply, and pipeline depth within resource
limits. Explicit matching parameters, including expert/top-k scope on the down
projection, remain constraints. A replacement must clear both held-out error
bounds and the minimum predicted gain at full and neighboring lower population.
An unqualified fit cannot displace the declared baseline; startup remains within
the existing baseline arithmetic and graph qualification contract.

Format-local `marlin_auto_geometry`, `marlin_max_candidates`,
`marlin_calibration_replays`, `marlin_max_error_percent`,
`marlin_min_gain_percent`, and `marlin_cache_fraction_min_percent` expose the
bounded calibration and admission policy. These parameters govern prediction,
not rounding or scale semantics. Numerical and canonical serving qualification
of a retained choice remain separate requirements.

### Cubic group decoding in the shared Marlin pipeline

Cubic compact AB, E5M9/curve2 and ordinary AB expert projections can use
Marlin's W8A16 transport while
retaining their original signed codes and polynomial. This is an **encoded
layout conversion**, not INT8 requantization or dense weight materialization:
W2–W8 codes occupy byte lanes; three FP32 group planes retain the checkpoint's
FP16-rounded `a`, FP16-rounded `b`, and original FP32 decoded scale. The final
weight rounds once to the requested FP16/BF16 type. Original encoded host
payloads remain the independent oracle. W1 and incompatible geometries continue
through their existing Triton implementation.

Metadata normalization belongs to the format frontend. Compact AB retains its
exporter-global scaling and FP16 coefficient rounds. Curve2 widens the low
14 scale bits into FP16 exactly and uses the high two bits to index the four
curves of the original output partition; concatenating gate/up must not merge
those partition domains. Ordinary AB copies its FP32 scale and FP16 coefficients.
All three representations then share the same polynomial, integer repacking,
reduction plan and cost rule. No second kernel or format-specific tile rule is
needed merely because the checkpoint serializes these group values differently.

The additional resident storage is real: each local expert/projection uses
`N*K + 12*N*ceil(K/group_in)` bytes for the prepared representation, with one
copy shared across graph buckets. Compare that footprint with the original bit
packing when assessing capacity; W8A16 transport does not retain the checkpoint's
nominal bits per weight. Preparation is bounded to one expert/projection at a
time and never reconstructs a floating `N*K` matrix.

`common/marlin` owns the group-plane staging, register-buffer lifetime, integer
repacker and device resource accounting. The Cubic extension supplies only its
group arithmetic and preparation. Custom group planes use the same scale
permutation as Marlin. Async prefetch can overwrite a shared group slot before
all its K subtiles finish: retain the current group in both register buffers
until its mathematical boundary, rather than repeatedly reading that shared
slot. G512 with a smaller K tile explicitly exercises this lifetime.

The two-level register pipeline requires an even number of K subtiles. For
WNA16 this is `block_k*block_n` divisible by `threads*64`; groups spanning several
K tiles must close inside the cyclic pipeline. These are geometry invariants,
not GPU-model presets. The Cubic baseline keeps four K partitions and their
reduction order fixed across M; growing N grows the warp count proportionally.
Changing public concurrency must not change the arithmetic reduction plan.

The existing `marlin_*` controls are exposed through `cubic_moe`; `fixed_*`
continues to govern its Triton path. The shared native cost regression accounts
for launch floor, issue waves and the working set beyond usable L2, including
the extra FP32 group planes. It checks held-out shapes and the neighboring lower
population before selecting another launch. Geometry eligibility and a cost fit
are not serving or floating-point qualification.

Source triangulation: vLLM-Cubic `89af4f1ff792199b7c9260dd4a86feed445fd3d6`,
`cubic.py:198,576` and `marlin_template.h` supply the curve2 metadata ABI,
exact-Cubic group-decoder and
pipeline precedent; its W4/G32 exact-level path and A8 carrier paths have distinct
arithmetic contracts. SGLang `ba433bb4628f44d5031e910ad32194ad6c145ba8`,
`fused_marlin_moe.py:230`, supplies route-alignment and fused gate/up precedents.
llama.cpp `458681e1d5d4a29a1463c4732e03226cf384b997`, `ggml-cuda/mmq.cuh:1347`,
uses explicit expert bounds and separate packed MMA storage; it is not a Cubic
numerical oracle. Cubic Serve retains its own original-weight oracle and fixed
expert ownership, and has no runtime dependency on these neighboring checkouts.
