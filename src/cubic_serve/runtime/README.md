# Runtime strategy contracts

## Client error boundary

OpenAI and Anthropic JSON errors, and SSE failures after response start, use
reviewed English client messages. Backend and input-processor exceptions retain
their detailed diagnostic text only in the server log with the request ID and
causal traceback. Backend public messages pass a finite reviewed allowlist;
unreviewed text cannot be promoted to a client error. Visual admission reports
"image or video frame is too large"
to the client; required/admitted rows and the relevant server startup flag stay
in the log. All 5xx responses use generic messages. Request schema validation
may name public request fields but does not echo arbitrary field names or values.

## GPU worker ownership

`--mm-worker-layout colocated` is the default: one persistent GPU worker per
language-model rank. First-stage ranks also own their input encoder, running on
a separate host thread with its own communicator, CUDA stream, graph buffers
and bounded topology lease. The CPU frontend and preprocessing pool remain
separate from GPU execution. Encoded features still complete into immutable
request-owned host rows before language-model admission.

Both serving executors use the same input-factory ownership and transport
contract. Encoder staging is reserved before KV sizing. Encoder threads stop
before worker communicators are destroyed; an encoder execution failure poisons
its device owner and fails closed. `--mm-worker-layout separate` retains dedicated
input-encoder processes for deployment isolation and explicit comparisons.
Separating encoder processes does not separate language prefill from decode;
those still share the language worker's scheduling and graph capacity.

## Model graph and checkpoint storage

The Flash-Next text traversal, rank ownership, state and graph buckets are shared
by original BF16 and encoded checkpoints. Dispatch selects the model graph from
the declared architecture; weight admission then binds each logical weight to
its plain tensor or format-owned payload. Absence of quantization metadata is
not a model-family decision and must not send Flash-Next to the Qwen3.5 loader.
Plain admission retains tensor-set closure, stacked expert ownership and dtype
checks. It does not invent quantization metadata or enable WNA8 for BF16 weights.
Numerical and public-streaming qualification remains specific to the actual
checkpoint, execution arithmetic and topology.

Requests have no engine-imposed wall-clock lifetime. Queueing, prefill, thinking
and answer generation do not share a total expiry deadline. EOS/stop, output and
context limits, client cancellation and shutdown retain their existing semantics.
The common controller's worker-response watchdog bounds each wait for the next
validated result, including intermediate resident receipts; it does not measure
request age. Resident windows remain finite transactions bounded by step, page,
state and token resources. Ending a window hands execution back to the controller
and never terminates a healthy request merely because it has run for a long time.

## KV cache storage contract

`--k-cache-dtype` and `--v-cache-dtype` select storage independently. Values are
`auto`, `bfloat16`, `float16`, `fp8`, `int8`, `int6`, and `int4`; `auto` preserves
the model's compute dtype. Native dtype combinations still require a compatible
attention operator and fail closed when unavailable. Query retains its original
compute precision. There is no second Q-format family or quantized Query path.
Recurrent state and auxiliary indexer fields retain their separately declared
formats; the K/V flags do not reinterpret those tensors.

The common storage spec supplies grouping; each codec owns its arithmetic and
packing. K uses token/column groups and V uses channel/row groups, currently 32
members per group. FP8 means E4M3FN with a separate FP32 group scale. Symmetric
INT8/6/4 uses FP32 scales, round-to-nearest-even, signed two's-complement codes
packed least-significant bits first, and ranges [-127,127], [-31,31], [-7,7].
INT6 consumes six packed bits per member rather than an INT8 allocation.

An incomplete column group stays in the original compute dtype in the owner's
checkpoint state. Only complete groups have durable packed payload and scales.
A query sees earlier committed groups plus its native current token; prefill and
speculative review cannot use a later token to choose an earlier query's scale.
Accepted-prefix commit, rejection, copy-on-write and rollback include the tail
along with recurrent state. Payload, scales and page fields share explicit page
ownership; request checkpoints own tails. Shared prefixes require complete pages
with valid payload/scale inventories. Padding has no request or prefix identity.

`--enable-kv-cache-rotation true|false` enables the shared paired transform and
defaults to `false` during development. A quantized K transforms Query and K
together; a quantized V transforms V and applies the inverse to the attention
result. Query is still stored/computed in its native precision. Unquantized
cache sides do not acquire that side's transform. Rotation is part of cache,
graph and qualification identity; enabling it does not grant qualification.

Capacity accounting includes metadata and the native tail. For a physical page
with B tokens, H local KV heads, D channels, group width G and b payload bits,
the column group count is `H * D * ceil(B/G)` and the row group count is
`B * H * ceil(D/G)`. Each quantized group consumes `ceil(G*b/8) + 4` bytes,
including its FP32 scale; multiply by the local full-attention layer count.
Native storage instead consumes `B * H * D * dtype_bytes` for each side.
Add the exact written mask and model-declared auxiliary page fields. Each
quantized column side also requires `layers * G * H * D * dtype_bytes` per
state slot, including separately reserved checkpoint and scratch slots.
The allocator estimates these actual tensor shapes with per-allocation alignment;
it then retains weights, graphs, workspace and runtime reserve separately.

For complete 32-member groups without other page fields, FP8/INT8, INT6 and INT4
use 1.125, 0.875 and 0.625 bytes per scalar including scales. Those are storage
formulas, not whole-device capacity ratios or throughput claims. Model-specific
auxiliary fields, state/checkpoint counts and available device memory determine
actual admitted capacity. Numerical/state/cross-M/topology proofs and matched
public SSE1024 measurements remain separate acceptance requirements; format
selection alone never certifies an unmeasured combination.

Normal startup reports the actual rank-local weight residency separately for
device, pinned host and pageable host backing stores. Shared views count once;
mapped pinned tables count as host memory. Pageable host bytes describe tensor
backing extents, including file mappings, rather than process RSS. Prepared-weight
caches may contain
optional absent correction tensors and encoded matrices, both of which the
residency ledger handles without reading tensor values or copying payloads.
The same summary reports physical KV page backing, state slabs, other device
use, remaining free bytes and required reserve. Other device use includes graph
buffers, sampling, encoders, communication, allocator slack and CUDA resources;
it is a measured remainder, not projection-level attribution.

The cache-storage line records effective K/V formats, rotation, target/draft
bytes per token, public request limit, graph maximum M, owner slots and scratch
slots. INT4 K/V does not compress weights, GDN recurrent checkpoints, QSA side
fields, graph buffers or reserves. In particular, speculative target review can
require M greater than public concurrency C. TP capacity is bounded by the
tightest rank after these fixed allocations; it is not total VRAM divided by
the packed K/V payload size. These messages explain the existing admission;
they do not change its budget, context limit, guards or graph portfolio.

The shared text executor separates request admission from resident graph
capacity. `--cuda-graph-max-tokens` defaults to1024 physical rows and includes
every power-of-two anchor up to the largest one within that limit. Neither the
request ceiling nor a per-request prefill cap expands it. Speculative capture
owner capacity is derived from these rows and the maximum declared complete
review, then bounded by the admitted shared bytes. Bootstrap owner storage,
the compound-round portfolio and sampling priors follow that graph envelope,
not the number of permitted client streams. The existing fair scheduler runs
larger populations through complete fixed-capacity transactions. Every replay
still requires its sealed qualification and exclusive topology-wide lease.
The measured graph/runtime cost is charged before shared KV/state sizing;
there is no permanent recurrent-state reservation for inactive requests.

## Shared TP / EP factoring

For a MoE graph with TP=T and a legal EP=P dividing T, expert tensor parallelism
is Q=T/P. The topology owns both coordinates: expert owner is
`tensor_rank // Q`, and the intermediate shard lane is `tensor_rank % Q`.
Checkpoint expert IDs keep their declared contiguous owner slice. Packed expert
tensors slice both the expert axis and the intermediate matrix axis; unpacked
experts use the same owner coordinates. Format adapters retain their own packed
codes, scales, zero correction and reduction arithmetic.

Attention, shared experts and state continue to use their TP contracts. Routed
matrix partials complete over ETP, then owner contributions complete over EP,
or use a format-declared local fold followed by ordered TP completion. The legacy
BF16 MoE path already uses that latter completion. Qualification follows the
same groups and never substitutes all-TP peers for an expert's ETP peers.

Legal factors share these mechanisms without a model-specific EP whitelist.
Actual checkpoint/dtype/topology graph qualification remains a startup gate.
Fixed expert ownership can make rank workloads unequal: measure each routed
projection and its following collective wait. Local kernel strategies may use
the routed workload; the engine must not change expert ownership to hide waiting.

## Historical Dense strategy and retained equivalent

The last clean Dense revision before MoE is
`ce38b8c3257a43136774fb82028688ad801fb0ce`; the first MoE revision is
`0e90aa9`. Compare linear binding, model execution, graph capacities and draft
planning together; a single kernel selector does not describe the strategy.

| Boundary | Historical Dense | Retained equivalent |
| --- | --- | --- |
| Ordinary BF16 linear | M1/M2 canonical vector; larger M vendor GEMM | Ordered fixed-tile target reduction across M; canonical draft reduction. Actual local N/K selects typed linear parameters. |
| Independent projections | Separate GDN QKV and packed A/B/Z; separate attention Q/K/V | Shared-input launch grouping only when every member resolves to the same enabled plan. Original weights, outputs and rounding remain separate. |
| Routed experts | Absent | Independent MoE family, actual local routed workload and checkpoint ownership; never inherit Dense MLP tuning. |
| Recurrent state | FP32 | BF16 default; explicit `--mamba-ssm-cache-dtype float32` for reference comparison. KV precision is separate. |
| Graphs | Exact small capacities including M6 for K5 | Permanent physical M1 and sealed power-of-two anchors; K4–K7 at C1 pay actual M8 cost. |
| Acceptance | Conditional prefix survival, prior 4/1, decay .99 | Same request-local evidence and one-level transitions; K0 terminal. |
| Cost history | EMA .2 each sample | Configurable EMA and median window; default median of three. |

Do not restore M-dependent reduction switches, arbitrary exact-M capture or expert
remapping to reproduce a historical speed. The equivalent strategy must preserve
current numerical and graph contracts.

## Recorded reproduction

Raw records, exact command lines, checkpoint-config digest, source hashes and clean
worktree status are under `/workspace/artifacts/dense-original/`.
The historical chicken/rabbit prompt is:

> 鸡兔同笼，共有35个头、94只脚。请计算鸡和兔各有多少只，并说明过程。

Historical GPU4–7 fixed K5 measured 161.4 tok/s; dynamic repeats measured
146.9/155.5/158.0. These requested 2048 tokens and ended naturally after 387–474
outputs; they are historical workload reproduction, **not** canonical 1024-token
frontier qualification. On the clean pre-MoE worktree, GPUs0–3 reproduced fixed
K5 at 140.8/156.5/148.4 and dynamic at 139.9/131.7/144.8. Dynamic reached K7/K6/K6.
GPU0–3 crosses NUMA; historical GPU4–7 is within one NUMA node. Do not attribute
a numerical delta solely to code without controlling that difference.

For the historical SVG prompt
`请详细设计一个鹈鹕骑自行车的SVG，并给出完整代码与说明。`, exactly 1024 outputs,
pre-MoE dynamic on GPUs0–3 measured 114.5/115.3 tok/s. Current-code controls using
the identical isolated prompt namespaces measured 97.2/102.5. Current chicken/
rabbit controls measured 125.4/135.0/134.8. All completed without request failures.
These are baseline observations; a retained performance change still needs
forward/reverse controls and applicable numerical audits.

## Model-independent dynamic scoring

For request i and depth d, conditional acceptance is q(i,d), and prefix survival
is S(i,d) = product(q(i,1)..q(i,d)). The planner maximizes

```
sum_i(1 + sum_{d=1..K_i} S(i,d))
--------------------------------------------------------------
target_cost(actual resident capacity, maximum segment depth)
+ sum_depth proposal_cost(actual active requests at that depth)
+ measured complete-round residual (verification, sampling, host and transport)
```

Prefill consumes target rows and time but contributes no decode tokens to the
numerator. Captured communication is already included in graph time. There are no
model-name, parameter-count or GPU-name branches in this formula. Costs belong to
the loaded weights, topology and immutable operator configuration.

Startup target prices come from rank-aligned replays of already-qualified graphs:
Capacity and segment depth have separate prices. All input preparation finishes
before timing; discard one warmup and take the median of three. Correctness-audit preparation skew must not become an unvisited
bucket's permanent price. Proposal prices separately include their sampling
boundary. No synchronization or extra capture is added to the serving replay.

An unvisited complete-round price uses the same target calibration in ordinary
and constrained domains. When the matching target-body startup measurement is
available, only that component receives the live/seed target ratio; proposal,
sampling and other work keep their measured prices. A neighboring complete
observation supplies only the remaining excess after that same component
calibration. An inclusive observation at the actual logical occupancy supersedes
both estimates. Without a matching target-body measurement, retain the existing
conservative whole-round correction.

Positive-K allocation compares a common cohort depth plus at most one extra
proposal wave, clipped by each request's configured bound. Requests with the
largest marginal prefix survival receive that extra wave first. At each common
depth, compare the common cohort with its fullest reachable extra wave inside the
same target bucket. The extra wave cannot promote to a larger target bucket;
intermediate partly filled waves are not additional candidates. This fills unused
rows at non-anchor logical M while avoiding repeated K reversals from pricing
different routed occupancies as one bucket/depth workload. It preserves prefix
closure, resource limits and
one-level moves. Future cohort scoring crosses asynchronous
request cooldowns, but only ready requests move. Mixed prefill cannot promote K
using decode-only prices. In pure decode, the K0 comparison includes all runnable
decode requests, including those that a positive-K batch cannot admit. It requires
a configurable 3% predicted advantage over the positive allocation. Live/seed target slowdown
calibrates that comparison conservatively; measured no-proposal residuals price
K0 without charging speculative preparation. Only ready requests move, one level
at a time; stopped requests never resume drafting.

Once every selected request is terminal K0, the scheduler reuses ordinary owned
one-token continuation. It preserves physical graph capacity, exclusive ownership,
page-crossing updates and durable target-hidden state. Shared state or changed
request membership returns to full metadata binding. Virtual wave capacities
cannot establish a physical resident continuation. Stopped requests allocate no
new MTP pages.

### Scope and limitations

The estimator remains an approximation: target prices separate physical graph
capacity and maximum segment depth, but do not fit every context, segmentation
pattern or expert load. Live costs adapt to those effects. Only consecutive pure
decode rounds train costs; idle and mixed-prefill intervals cannot contaminate
those prices. The result-to-result interval includes controller, transport and
output feeding; subtracting target and proposal time leaves the complete residual
without charging captured communication twice.

Acceptance still requires a same-window ledger that closes actual committed work
against public SSE output. Neither predicted acceptance nor internal graph
throughput is a serving result. The 3% terminal margin is a configurable decision
threshold, not a statistical confidence bound or a promised performance gap.

## Configuration

CUDA geometry belongs to [operator parameters](../ops/README.md).
Runtime policy belongs to `--speculative-config`:

```json
{"method":"mtp","num_speculative_tokens":1,"mode":"dynamic","max_tokens":7,
 "policy":{"decision_tokens":24,"decision_min_rounds":8,
           "prior_success":4.0,"prior_failure":1.0,"history_decay":0.99,
           "cost_ema_alpha":0.2,"cost_sample_window":3,"terminal_stop_margin":0.03}}
```

`max_tokens=5` is the measured MoE launch bound; Dense may use 7. These are launch
configuration values, not model checks inside the planner. Fixed MTP continues to
use an integer `num_speculative_tokens` without dynamic mode.

| Parameter | Meaning and validity |
| --- | --- |
| `decision_tokens` | Positive committed-token interval |
| `decision_min_rounds` | Positive round floor; both decision thresholds must pass |
| `prior_success`, `prior_failure` | Finite positive conditional acceptance pseudocounts |
| `history_decay` | Retained fraction, strictly between zero and one |
| `cost_ema_alpha` | New sample weight, greater than zero and at most one |
| `cost_sample_window` | Median window: 1, 3, 5 or 7; 1 reproduces old EMA updates |
| `terminal_stop_margin` | Finite nonnegative fractional advantage required for global K0 lookahead |

Unknown fields and invalid values fail before startup. Configuration is immutable
and logged; all workers receive the same value. Quantized linear and routed-MoE
families will own separate operand and execution schemas, reusing only configuration
transport and precedence. No setting bypasses numerical or graph qualification.

## Upstream semantic basis

Inspected code facts, not numerical oracles:

- llama.cpp `4d9176092d00586775af140581bb0b558ddc4389`,
  `common/speculative.cpp`: bounded draft confidence and accepted-prefix bookkeeping;
  no matching cost-table allocator. `src/models/qwen35.cpp` keeps original projection
  and recurrent boundaries.
- vLLM `e962733e08d10f7ca65dac4df99e116460b8b174`,
  `vllm/v1/spec_decode/llm_base_proposer.py`: padded proposal graph shapes, without
  Cubic's terminal-K0 allocator.
- SGLang `991368d8800cda6075c9f1b0be256af7cb44b8c8`,
  `python/sglang/srt/speculative/dspark_components/dspark_planner.py` and
  `dspark_sps.py`: expected accepted work over measured graph-tier costs, with
  batch/proposal-dependent cost terms.
- For continuation, vLLM `89af4f1ff792199b7c9260dd4a86feed445fd3d6`,
  `vllm/v1/core/sched/scheduler.py`, clears per-step speculative metadata after
  budget selection. SGLang `ba433bb4628f44d5031e910ad32194ad6c145ba8`,
  `python/sglang/srt/managers/schedule_batch.py`, separates speculative preparation
  from ordinary one-token KV allocation and clears stale decode metadata.
  llama.cpp `8b4b3558f1459c13e4aa38d5c94d306a00dc6acd`,
  `common/speculative.cpp`, bounds proposals per sequence; it has no equivalent
  Cubic resident-graph continuation contract.

Shared invariants are explicit proposal bounds, ordered accepted prefixes, target
verification and state ownership. Rank-aligned startup calibration and terminal-K0
resident reuse are Cubic implementations of measured-cost and state-lifetime
requirements, not copied upstream algorithms.

The cohort candidate constraint was additionally checked against vLLM
`89af4f1ff792199b7c9260dd4a86feed445fd3d6`,
`vllm/v1/spec_decode/dynamic/utils.py` (configured batch-size ranges select one
bounded depth), and SGLang `ba433bb4628f44d5031e910ad32194ad6c145ba8`,
`python/sglang/srt/speculative/adaptive_spec_params.py` (batch-level candidate
depths, acceptance history and hysteresis; unsupported unsynchronized DP-attention
tiers are rejected). Cubic does not adopt their batch-size tuning tables: actual
request survival and complete measured round costs select the cohort and extra
wave. llama.cpp's inspected `common/speculative.cpp` at the commit above retains
per-sequence confidence and length bounds but has no equivalent full-cost cohort
allocator. These are execution-shape constraints, not numerical references.

## Joint acceptance and public counters

C1 remains pending. Assess C1, C4, C8, C16, C32, C64, C128 and C256 as one curve;
revisit low-concurrency behavior when a shared change warrants it. For each model
and topology, C64/C128/C256 must approach or exceed the best same-condition public
streaming result, with at most a 3% deficit. A measured frontier is not a proven
hardware limit. No model/device/concurrency tuning table implements this rule.

The status line's `decode` counter counts actual published output tokens after
EOS truncation; rejected draft tokens never contribute. Its denominator is the
union of admitted busy intervals, including prefill and stalls, excluding idle.
It is an aggregate interval rate, not an individual request's pure decode rate.
The public request benchmark uses `(output_tokens - 1) / (last_SSE - first_SSE)`;
cumulative usage counts retain all tokens when several arrive in one SSE event.
A common steady window additionally closes public output against committed work
and the complete round cost, including controller and transport overhead.

## Prefill budget and owned-state publication

Omitted `--prefill-chunk-tokens` means use the available qualified graph budget.
A positive value is an explicit additional cap; each recurrence segment must also
fit one admitted physical wave. Pure prefill can use the available budget. Mixed
prefill preserves the existing minimum quota and also consumes rows that currently
ready decode transactions cannot use, accounting for each request's actual draft
depth. MTP maintenance remains capped by the admitted proposal graph capacity.
There is no universal 16-token cap and no model-name selection. Decode-only cost
curves exclude mixed/pure prefill: sequential recurrence is a different workload.
An unvisited startup target price may be conservatively calibrated by observed
decode slowdown; an actual measurement of that bucket takes precedence.

MTP maintenance consumes the previous prefix's hidden state before the target
publishes its new segment-final hidden into an exclusively owned slot. The same
production helper handles physical and wavefront target execution. Startup compares
an owned in-place prefix with a COW prefix, including target and MTP cache/state.
A COW-only audit cannot establish source/destination alias correctness.

Checkpointed causal convolution must preserve the original history or an equivalent
rolling, already-rounded history across every output and checkpoint write. An
exclusive destination may alias its source; reloading that overwritten source on
the next token violates the recurrence. Tests must include multi-token aliasing,
not only one-token decode and distinct COW destinations. Independent convolution
math and full-model chunk/state identity are complementary proofs.

Source contracts inspected for these boundaries:

- llama.cpp `8b4b3558f1459c13e4aa38d5c94d306a00dc6acd`:
  `tools/server/server-context.cpp` bounds batch submissions;
  `common/speculative.cpp` maintains per-sequence token/hidden pairing;
  `ggml/src/ggml-cuda/ssm-conv.cu` carries convolution history locally.
- vLLM `89af4f1ff792199b7c9260dd4a86feed445fd3d6`:
  `vllm/v1/core/sched/scheduler.py` separates total token budget and chunk admission;
  `vllm/v1/spec_decode/llm_base_proposer.py` aligns shifted tokens and target hidden;
  `vllm/model_executor/layers/mamba/ops/causal_conv1d.py` reads initial history
  before state update.
- SGLang `ba433bb4628f44d5031e910ad32194ad6c145ba8`:
  `python/sglang/srt/managers/scheduler.py` has separate chunk/decode controls;
  `python/sglang/srt/models/qwen3_5_mtp.py` consumes target hidden with token inputs;
  `python/sglang/kernels/ops/mamba/causal_conv1d_triton.py` reads prior-block history
  before publishing new state.

Their execution/storage layouts differ from Cubic's checkpoint destinations and
leased graphs. They establish semantic constraints, not a numerical oracle.

## Earlier local measurements — not final-code qualification

The preceding implementation was measured through public streaming Chat Completions
on GPUs 0–3 (RTX 6000D), Dense TP4/EP1 and MoE TP4/EP4, with BF16 KV and recurrent
state, seed 1, xhigh and exactly 1,024 requested output tokens. Each model completed
509 requests over the ascending C matrix without failure. Prefix caching is enabled;
the deterministic namespace is reused across levels, so cache history follows the
whole trace. These numbers do not qualify another device or topology.

| C | Dense total SSE tokens/s | MoE total SSE tokens/s |
| --- | ---: | ---: |
| 1 | 118.5 | 152.1 |
| 4 | 366.5 | 360.9 |
| 8 | 577.4 | 578.9 |
| 16 | 928.2 | 784.5 |
| 32 | 1299.1 | 1109.0 |
| 64 | 1869.4 | 1453.4 |
| 128 | 2260.6 | 2124.2 |
| 256 | 2220.3 | 2897.9 |

These historical measurements predate the multi-token owned-prefill alias fixes
and do not qualify the current code. They remain performance targets, never
numerical references. They are common steady-window public SSE totals, closed against committed work
and complete measured round cost. Whole-request C1 decode rates are 117.2 and
151.2 tokens/s; the distinction is intentional. Dense reaches a plateau at C128–256.
MoE C64 remains about 17% below the strongest intermediate candidate (1748.3);
this is a remaining allocation-quality limitation, not an established optimum.
The matrix has no concurrency cliff, but should not be described as improving
every earlier point. Device power, compute activity, memory activity and clocks
were recorded over every public request interval.

For 30,000 requested outputs, the final 2,048-token/initial 2,048-token rate ratios
were Dense 0.9542 (112.9 to 107.7 tokens/s) and MoE 0.9163 (167.3 to 153.3).
Both passed the 0.7 decay guard. At C32, staggered long-prefill requests had 5,177
Dense / 5,135 MoE input tokens; existing streams' maximum output gaps were under
95 ms. This tests the required 4K–8K arrival range, not an exactly 8,192-token
prompt. A 60,000-token horizon was not measured.

Raw SSE timelines, same-window cost breakdowns, GPU telemetry, commands and source
manifest are under `/workspace/artifacts/four-hour-recovery/qualified-matrix/`,
`RESULTS.json`, and `retained-source-manifest.json`. Numerical evidence combines
independent Dense BF16 leaf witnesses, full-48-layer MoE causal/eager witnesses,
and separate cross-M/padding/row-permutation and frozen speculative state checks.
Same-body execution identity is not presented as an independent numerical oracle.


## Earlier pre-audit measurements and remaining margin

Records are under `/workspace/artifacts/prefill-batch-recovery/`. `final-matrix`
and `host-profile-matrix` use the same retained runtime/operator code, GPUs 0–3,
BF16 KV/recurrent state, Dense TP4/EP1 or MoE TP4/EP4, seed 1, xhigh, the same
isolated SVG prompt namespaces and exactly 1,024 requested outputs. Both enable
full draft tracing. The second MoE run additionally times planner/scheduler wall
and thread CPU time without changing their inputs or decisions. Each complete
matrix has 509 successful requests and zero failures. Rates below are common
steady-window public SSE outputs, with complete round-cost closure; they are not
internal graph rates or sums of per-request rate averages.

| C | Dense final | MoE final | MoE timed repeat |
| --- | ---: | ---: | ---: |
| 1 | 115.12 | 157.12 | 156.08 |
| 4 | 371.73 | 359.88 | 353.96 |
| 8 | 566.30 | 547.06 | 522.81 |
| 16 | 959.86 | 789.61 | 791.59 |
| 32 | 1354.81 | 1220.52 | 1093.56 |
| 64 | 2020.49 | 1694.40 | 1700.92 |
| 128 | 2340.12 | 2191.31 | 2189.66 |
| 256 | 2226.31 | 2919.34 | 2920.08 |

C1 remains pending in the joint curve. Whole-request C1 decode rates in the final
matrix are Dense 116.23 and MoE 155.59 tokens/s. Corrected fixed K3 controls gave
127.72 and 160.28; those controls and their same-window totals remain separate in
`fixed-correct`. Dynamic is not claimed to dominate every fixed K or every repeat.

Against the strongest recorded high-C targets, Dense C64/C128/C256 deficits are
0.59%/0.26%/2.20%. MoE C128/C256 deficits are about 1.43%/1.41%. MoE C64 spans
3.08% to 2.71% below the historical 1748.31 target: one full repeat passes the 3%
line, but the first misses it. This is borderline repeatability, not a stable
performance reserve. Historical results preceding the owned-prefill state fixes
are retained as conservative speed targets, never numerical oracles. The corrected
fixed K3 C64 control is 1695.60 tokens/s. Continue assessing low-C variation along
with the high-C curve; a measured frontier is not a hardware limit.

The C64 timed repeat measured 1.50 ms of planner CPU time per round, within
4.53 ms of batch construction wall time. This is an actual host cost; the tiny
wall/CPU difference does not support blaming GIL waiting for the planner cost.
A physical-shape price-cache candidate reduced standalone planner CPU time but
was rejected after complete public matrices: MoE C64 measured 1688.94 tokens/s,
without improving the critical gap, and some low-C points regressed. It is absent
from the retained source. CPU microbenchmarks alone cannot qualify serving gains.

Current cold long-prompt TTFT is Dense 3.49 s for 5,177 input tokens and MoE 2.51 s
for 5,135. Initial 16-token-cap controls took 8.39/10.31 s; those controls predate
the state fixes and diagnose the cap rather than qualify model numerics. During
C32 staggered long-prefill tests, the largest existing-stream output gaps were
295/213 ms, with no failed request or seconds-long global decode stall. Device
power, compute/memory activity, clocks, request timing and KV occupancy accompany
the public runs. Both final servers exited on SIGINT without an error traceback.

Earlier local numerical evidence included Dense full-K witnesses on sampled output
columns at **logical M1** in physical M1/M8 (497 occurrences per rank). Physical M8
is not a C8 concurrent audit. It also includes single-request full-layer MoE causal/eager
and MTP oracle tests,
independent CUDA convolution/GDN tests, and all-rank 273-token chunk identity at
1/16/64/256. Separate graph audits cover 48 target boundary/row-map cases,
48 MTP cases across seven depths, and 89 frozen review/repair cases per rank for
both models. Graph/state identity and independent numerical evidence are distinct.
The old standalone MoE graph-oracle harness timed out; that execution failure is
preserved separately and is not reported as a floating-point verdict.

`tools/qwen3_5/concurrent_audit.py` captures actual serving review frames and compares
their logits and BF16 KV/recurrent/conv state with independent target reference math.
Request IDs must match the public load cohort; participating requests, logical rows,
and physical graph capacity are separate quantities. Committed-token records witness
the consumed prefix, not numerical correctness. This diagnostic synchronizes and its
timings do not qualify serving throughput. A selected target frame with frozen proposal
tokens does not qualify the complete speculative algorithm or the complete serving trace.
Proposal/verify/repair, padding poison, row permutation and applicable reduction proofs
remain separate requirements. Unexplained differences remain unqualified; successful
requests, graph identity and a completed audit process never imply a numerical pass.


On that earlier code, both models completed exactly 30,000 streamed outputs.
Dense first/last 2,048-token rates were 114.60/94.99 tokens/s (ratio 0.8289);
MoE rates were 172.91/137.95 (ratio 0.7978). These pass the existing 0.7 tail-ratio
guard but represent larger decay than the earlier, numerically different run.
Maximum output gaps were 51.4/48.9 ms. Intermediate token-rate dips must be read
with accepted work: near 26K MoE outputs, mean acceptance fell from 4.45 to 2.23
while target time decreased from 24.05 to 19.65 ms. This is fewer committed tokens
per round, not evidence of a blocking prefill or a sudden target-kernel slowdown.
Long-run records and per-2,048-token cost intervals are in `final-long`.

C1 and the joint curve remained subject to review. In particular, MoE C64 did not
yet have stable margin within the 3% historical-target band; those measurements
did not establish complete acceptance.

## Concurrent speculative audit provenance

`tools/qwen3_5/concurrent_audit.py` captures each selected public target review's
actual pre-review BF16 KV, convolution and recurrent state. For an unfinished COW
page, the source page is part of that snapshot even when the review's destination
page has not yet been populated. The independent reference resumes from this
frozen prefix and consumes the actual proposal-substituted review tokens. A
separately recomputed cold prefix is a diagnostic; its accumulated reduction
history cannot stand in for the fixed prefix of a speculative transaction.

Audit snapshots reserve memory before KV admission, are immutable while the
forward runs, and are read only after completion and identical ordered rank
schedules. Auxiliary shared-expert operands must be cloned on their producing
stream before the normal join. These observers are audit-only. Harness failures,
independent numerical differences, graph identity, and speculative decision/state
qualification remain separate outcomes. Raw non-exact output cannot become a pass
without its applicable named causal witnesses; acceptance rate or final text
cannot supply those witnesses.


## Earlier concurrent qualification after initial rounding corrections

The independent concurrent suite and exact retained-source manifest are in
`/workspace/artifacts/concurrent-final/qualification.json`. Both local TP4
configurations (Dense EP1 and MoE EP4) pass selected public C8/C32/C33 review
transactions: full-output/full-K linear witnesses, target state and nonlinear
boundaries, complete MTP proposal/catch-up forwards, independently reproduced
Philox accept/reject/recovery decisions, and durable BF16 state repair. Target
and MTP output/KV/state identity, routed expert IDs/weights and TP collective
results additionally pass changed physical graph capacity, poisoned padding and
permuted request row maps. C33 target reviews include logical M67/66 replayed at
physical M128 and M256. Physical capacity does not stand in for public concurrency.

The independent reference starts from the frozen pre-review prefix. The corpus
is the selected actual transactions and their complete logical rows, not an
assertion that every output token of every workload has been numerically replayed.
Raw non-exact reference files remain available; the composed verdict also requires
all named causal witnesses and graph/state/route/collective identity checks. The
new convolution SiLU, gated RMSNorm division and MoE intermediate-rounding
regressions reject their previous implementations.

Canonical 1024-token streaming C1 through C256, staggered 5K prefill and 30,000-token
long-generation results for the corrected operators are stored alongside that
suite under `performance/`. Numerical qualification and throughput-frontier
qualification remain separate. C1 remains pending in the joint curve; the record
includes the Dense C1 decrease instead of treating a numerical fix as proof of a
performance improvement.

## Earlier concurrent qualification and cohort contract

`/workspace/artifacts/cohort-final/NUMERICAL.json` binds the retained Python source
hashes to the final Dense TP4/EP1 and MoE TP4/EP4 C8/C32/C33 numerical verdicts.
Both pass independent frozen-prefix target and complete MTP proposal/catch-up
math, independent Philox decisions, and all-rank BF16 state repair. Each model's
C33 review has logical M66 in physical M128 and also passes M256, poisoned padding
and permuted physical request rows. Route IDs/weights and TP results are checked
explicitly alongside output, KV and recurrent state. Captured request identities
match the corresponding public concurrency cohorts. Dense's cross-M target/MTP
identity is replayed again after the final MoE-only routing correction.

This includes the actual C33 routing-softmax midpoint regression: fast exponential
approximation produced a wrong BF16 route weight even with exact route IDs and
cross-M identity. Full-precision exponential fixes that mathematical boundary;
the old implementation fails the retained independent regression. The preceding
convolution SiLU, gated RMSNorm division and MoE intermediate-rounding regressions
remain active. The corpus comprises selected complete transactions, not every
output token of every workload.

That revision's canonical streaming matrix, staggered prefill, 30,000-token decay and
fresh-server fixed/dynamic C64 controls are recorded separately in
`/workspace/artifacts/cohort-final/qualification.json` and `RESULTS.md`. Earlier
planner controls before the final routing correction remain diagnostic history;
they are not substituted for final-code throughput. C1 remains pending in the
joint curve. The general planner uses the cohort-plus-extra-wave contract above,
not a model/device/concurrency lookup table.


## Retained resident-fill cohort qualification

Final records are `/workspace/artifacts/fill-final/NUMERICAL.json`,
`qualification.json` and `RESULTS.md`. Both local TP4 configurations pass C8, C32
and C33 independent target/MTP mathematics, fixed Philox decisions and durable
BF16 repair, plus cross-M output/state/route/collective identity with poisoned
padding and permuted request rows. The actual public transaction corpus is held
fixed across the planner-only change; every reference and candidate audit is
rerun on the retained source. C33 has logical M66 in physical M128 and M256.

The cohort's extra wave now only fills reachable space inside its current target
bucket. Intermediate partly filled waves had caused repeated K reversals. With
the same corrected arithmetic, the complete-trace MoE C64 SSE rate improves from
1535.45 to 1660.93 tokens/s. Fresh-server dynamic C64 is 1691.31 versus fixed K3
1696.95. Fixed K3 does not execute the changed dynamic planner.

Canonical public SSE totals, exactly 1024 requested/produced tokens per request:

| C | Dense TP4/EP1 | MoE TP4/EP4 |
| ---: | ---: | ---: |
| 1 | 104.47 | 155.23 |
| 4 | 380.33 | 371.81 |
| 8 | 559.30 | 568.98 |
| 16 | 929.83 | 761.64 |
| 32 | 1395.46 | 1211.77 |
| 64 | 2015.98 | 1660.93 |
| 128 | 2335.18 | 2191.38 |
| 256 | 2222.68 | 2913.19 |

Each model completes 509 matrix requests without failure. C64/C128/C256 are
within 3% of the listed valid references; those reference identities and gaps
are explicit in the report. C1 remains pending in the joint curve, not claimed
to be at its best frontier. Both 30000-token runs pass the existing 0.7 tail/head
guard: Dense 111.26 to 82.28 tokens/s, MoE 172.93 to 152.82. Staggered 5K prefill
results and per-device power/activity/clocks are retained with the SSE timelines.
A 60000-token horizon and other devices/topologies are not qualified by this run.


## Packed serving startup and explicit numerical acceptance

Ordinary packed `performance` serving loads the original checkpoint payloads,
prepares kernels, measures the complete graph memory portfolio before sizing KV,
and captures the final graphs at their actual cache addresses. Each startup still
checks same-body eager/capture identity, padding and cache/state transitions before
readiness. Missing graph resources fail closed; runtime eager fallback is unchanged
and forbidden. Those instance checks are not an independent numerical oracle.

The expensive independent original-weight and cross-M numerical suite is an
explicit acceptance operation: add `--startup-numerical-audit` to the quantized
performance invocation. It retains the existing oracle, reduction bounds,
concurrent/padded row permutations and distributed checks; failure prevents
readiness. Normal serving does not install its observer bindings, initialize its
otherwise unused BLAS context, run the suite, or manufacture a numerical pass.
Independent numerical qualification and canonical streaming measurements remain
release/development requirements when arithmetic, strategy or topology changes;
ordinary startup success is not fresh qualification. This separation follows the
user's correction that production startup must not silently rerun debugging audits.

Loading progress starts only from the loader's real materialization callback,
after worker/CUDA/communication initialization. Zero percent does not represent
process spawning. Completion follows shard validation. Kernel preparation, graph
memory measurement, final capture/replay checks, and state checks have separately
labeled elapsed times, in seconds to two decimal places.

Upstream startup references (implementation facts, not numerical oracles):
vLLM `89af4f1ff792199b7c9260dd4a86feed445fd3d6`,
`vllm/v1/worker/gpu_worker.py` (`load_model`, memory profiling and
`compile_or_warm_up_model`); SGLang
`ba433bb4628f44d5031e910ad32194ad6c145ba8`,
`python/sglang/srt/model_executor/model_runner.py` (`load_model`,
`init_cuda_graphs`); llama.cpp
`458681e1d5d4a29a1463c4732e03226cf384b997`,
`ggml/src/ggml-cuda/ggml-cuda.cu` (`ggml_cuda_graph_evaluate_and_capture`).
These inspected startup/capture routines prepare executable resources without
running an independent full-model numerical oracle. llama.cpp permits runtime
graph updates; Cubic deliberately retains startup-only capture and graph leases.


### Worker launch metadata and graph memory measurement

Spawn each worker with small control arguments and live multiprocessing handles.
The controller serializes the admitted metadata once into its private temporary
directory; workers load that engine-owned file after concurrent process imports.
Do not place the complete checkpoint inventory/ownership object in every spawn
pipe: a payload larger than the pipe can block `Process.start()` until the child
finishes importing and drains it, serializing otherwise independent imports.
The metadata file is not a checkpoint, user-supplied pickle, persistent cache or
expert remapping mechanism. Its lifetime follows the owning controller; a read
failure notifies the parent before any worker collective can start.

The shared text-graph entry admits the complete logical checkpoint once for
the vision and language modules. Reuse checks the model root, complete logical
plan, config/index content and shard stamps. A worker payload view keeps the
original encoded bindings, file evidence and global semantic/inventory digests;
it omits unused tensor-header records and cannot replace complete admission.
Vision workers receive only the vision partition. Language workers retain the
complete ownership plan used by distributed execution identity.

Startup logs distinguish metadata discovery, metadata serialization, worker
bootstrap, metadata restoration, CUDA/collective initialization and loading.
The controller queries startup free memory through CUDA-visible device UUIDs
and NVML without creating a primary CUDA context. GPU execution workers apply
allocator limits before loading. With separate vision TP workers, each card has
two GPU workers; the control process does not occupy a third GPU context.
This layout does not establish a throughput advantage over colocated encoders.

The packed graph memory phase currently constructs and captures all temporary
buckets, measuring executor buffers, operator workspaces, graph residency and
construction peaks before allocating final KV. It is a full portfolio measurement,
not a cheap graph-size query. Per-bucket logs separate construction and capture
cost. The final arena has different addresses, so those temporary graphs cannot
be replayed as final graphs. vLLM's pinned `gpu_model_runner.py` at
`89af4f1ff792199b7c9260dd4a86feed445fd3d6:profile_cudagraph_memory` samples the
largest two descriptions per mode in a shared profiling pool and extrapolates
incremental graph overhead. Cubic's independently allocated bucket executors do
not establish that shared-pool bound. Removing this phase requires accounting for
the full independent workspace portfolio or changing its ownership/allocation
lifecycle; neither a fixed memory surcharge nor copying the two-sample estimate
is a proved replacement. This revision does not claim to eliminate duplicate
portfolio construction.

Process orchestration references: vLLM at the same pin,
`vllm/v1/executor/multiproc_executor.py:WorkerProc.make_worker_process`, and
SGLang `ba433bb4628f44d5031e910ad32194ad6c145ba8`,
`python/sglang/srt/entrypoints/engine.py` scheduler-process launch: both pass
configuration/control metadata and initialize workers separately. They do not
pass Cubic's admitted complete checkpoint graph through spawn. llama.cpp
`458681e1d5d4a29a1463c4732e03226cf384b997` CUDA graph backend runs in process,
so it has no equivalent rank-spawn metadata protocol.

## Shared visual execution across checkpoint formats

Multimodal adaptation follows the model architecture: establish media processing,
vision execution, feature placement and position/state semantics with the original
checkpoint, then reuse that path for quantized derivatives. Each visual tensor
binds through the common checkpoint and format/operator contracts. The visual
module may itself be quantized; language quantization is not a reason to assume
BF16 visual weights. A missing format capability belongs in the shared operator
layer, not in a second visual pipeline for a particular model and quantization.
Automatic graph connectivity still requires numerical and serving validation of
each actual checkpoint arithmetic and topology.

The shared CUDA/debug serving backend constructs `QwenVisionGraph` before
language-graph KV admission. It reuses the Qwen visual reference traversal and
`CheckpointWeights` logical inventory: plain projections use their native
floating-point kernels, encoded projections and learned positional embeddings
bind through `PackedLinearOperators`. Encoded matrices retain their packed
payloads. Each startup bucket seals the existing projection occurrence dispatch
before capture; public requests neither bind projections nor capture new graphs.
Visual startup applies the existing `configure_reference_math` policy before
the first CUDA allocation, including disabled truncated BF16/FP16 reductions.
Plain biased projections retain the library's actual rounding boundaries; an
audit must account for a BF16 matmul storage boundary before bias when present.

Dense INT Marlin split-K partials opt into the existing sealed dispatch scratch
reuse contract: each merge consumes the complete partial buffer before a serial
successor can overwrite it. Activation codes, scales, persistent outputs and
routed metadata retain their own lifetimes. A MoE shared branch that can run on
another stream binds private projection scratch; it cannot borrow the serial
main-stream pool. Reuse changes storage ownership, not the kernel or reduction
plan. Runtime KV admission logs free memory, graph/state costs, reserve and the
local page budget before the topology-wide minimum is chosen.

The shared CUDA/debug backend runs the visual encoder with weight TP over the
first pipeline stage's tensor group. QKV shards own complete heads in Q/K/V
order; attention output, MLP and merger row projections reduce rank partials
before adding their bias once. Patch embedding, positions, norms and residuals
remain replicated. Original and encoded checkpoints use the same ownership rules;
format-owned slice materialization preserves the declared encoding and values. Encoded
MLP cuts respect input scale groups where possible and retain the partial tail.
There is no implicit switch to a replicated data-parallel encoder or TP1.
An incompatible head or projection geometry rejects startup explicitly.

The encoder primary device defaults to the first `--devices` entry. An explicit
`--device` must belong to that input-stage tensor group and selects its primary
rank. Persistent encoder ranks use a separate collective group; their exclusive
topology lease spans shared host input fill through every rank's completion and
the immutable result copy. Decoder KV sizing sees all encoder residents and
reserves staging separately on each physical device. Encoder and decoder
collective ordering are independent; this does not establish useful GPU overlap.
`--multimodal-memory-gib` bounds each encoder device's graph/staging allocations.
Startup reports the actual encoder TP size and devices separately from the
language topology. The existing single-device eager/reference entrypoint remains
a diagnostic execution path and does not claim encoder TP qualification.

The inspected implementations are vLLM `89af4f1ff792199b7c9260dd4a86feed445fd3d6`,
`vllm/config/multimodal.py` and `vllm/model_executor/models/qwen3_vl.py`
(default weight TP, separately selected input-data parallelism), SGLang
`ba433bb4628f44d5031e910ad32194ad6c145ba8`,
`python/sglang/srt/models/qwen3_vl.py` and `python/sglang/srt/server_args.py`
(default attention-TP group, optional replicated DP encoder), and llama.cpp
`8b4b3558f1459c13e4aa38d5c94d306a00dc6acd`, `tools/mtmd/clip.cpp`
(one selected accelerated encoder backend plus CPU; no encoder tensor split).
These are source facts, not upstream numerical or performance oracles.

Startup
logs name vision metadata admission, checkpoint payload loading, kernel
warmup/calibration and each captured vision graph before language weight loading.
The `weights/vision` bar advances by locally materialized vision tensors.
Shard-file counts are omitted because a file can contain many vision and language
weights and is not the progress denominator. Bar prefixes identify the module:
`weights/vision`, `weights/main_model`, and `weights/mtp`; the formatter does not
repeat that identity. A future separate draft loader can use `weights/draft_model`.
Main-model counters retain the loader's actual unit, tensors or bytes shown as GiB.
The common `inputs.multimodal` policy applies to input encoders independently of model,
modality and checkpoint format. All public memory amounts use GiB (2**30 bytes).
`--gpu-memory-gib` is the per-device application envelope: weights, input encoders,
graphs, workspace, state and KV share it. Its default `auto` uses memory available
at startup; an explicit amount above availability is rejected. Separate GPUs do
not pool this budget. `--gpu-reserve-gib` (default `1`) remains unallocated inside
that envelope for runtime work. CUDA Torch allocators are limited before weights
load, and actual driver-visible residency closes the graph/cache admission.
`--multimodal-memory-gib` (default `1.5`) is a sub-budget for startup encoder graphs
and bounded input work, clipped by remaining global headroom. It is an upper
bound, not a preallocation. Encoder weights also consume the global envelope.
Positive decimal GiB amounts such as `1.5` and `2.2` are accepted and converted to
integer bytes; the GPU application budget still defaults to `auto`.
Weights and resident graphs remain visible to decoder KV sizing. The startup
controller also reserves each physical device's admitted peak encoder input work
before deciding BF16 KV capacity. An independent image or video frame that
exceeds the startup-captured visual graphs runs through the same native visual
operators in eager mode if its estimated working set fits that sub-budget.
Admission happens before dispatch to any TP rank; an item beyond the budget is
rejected without changing the image or poisoning the encoder workers. The encoder
lease starts before upload and covers graph replay or eager execution through
completion of immutable BF16 host feature rows. Queued requests and the byte-
bounded feature cache hold no frontend device features. `--mm-feature-cache-gib`
(default `0.25`, `0` disables caching) controls that host cache. This encoded-feature
cache is independent of language prefix/KV/state caching: a hit saves encoder work
while uncached language prefill still executes. The old fraction and MiB CLI flags
are removed rather than assigning their old values new units.

`--mm-preprocess-workers` (default `2`) bounds actual image/video/audio input
processing workers. HTTP cancellation does not release a permit until its real
worker finishes; text-only input does not wait behind the media gate. This bounds
active preprocessing, not queued HTTP payload bytes or all live CPU request
features. A cache eviction drops cache ownership, not a live request's reference.

Each model declares independent encoder segments. Qwen temporal patch groups
have independent visual attention, so videos pack complete groups into bounded
existing graph buckets, preserve their order and original timestamp/position
recipe, and copy each group's features to host before reusing device scratch.
Total video patches can exceed the largest bucket; one spatial segment must fit.
No runtime capture, exact-size graph, frame pruning or floating-weight expansion
is introduced. Startup logs resident bytes and maximum admitted segment capacity.
The repository's preprocessing pixel limit is an upper bound, not a promise that
the whole pixel range fits a captured portfolio on every device. Raising the
budget admits larger buckets and reduces memory available to decoder KV/state.

The common media decoder samples the complete video at `--video-fps` (default
`2`) and admits at most `--video-max-frames` (default `768`), subject also to the
decoded pixel and encoded byte limits. It rejects over-limit sampling rather
than returning a truncated prefix. Lower FPS can cover longer clips within the
same frame budget. The decoder selects source frames across the complete clip,
retains their actual relative PTS, and disables synthesized CFR duplicates.
The cadence, source duration, timestamps and repeated-tail indices remain in
trusted metadata, including rates below 1 FPS. Unknown or
incorrect duration is checked with one extra bounded decoded frame, which is
never admitted. This is uniform cadence sampling, not a guarantee to observe
every short event between samples. The resource layer also accepts
audio encoder costs and declared independent windows, but only a model with an
implemented audio processor/encoder can accept audio input. Flash-Next has no
audio encoder. Output-modality generation is outside this input mechanism.

Immutable request commands carry BF16 visual feature bytes, their semantic
identity, prompt positions and affine decode continuation. Feature width is the
model's input width H, independent of the decoder's internal working width.
Flash-Next broadcasts each H-wide feature to its hyperconnection streams after
the lexical embedding TP sum. The reference sparse indexer preserves the actual
three-axis position of each compression group's first token. Request admission
checks both token extent and the complete requested position-axis continuation.

`--startup-numerical-audit` additionally exercises feature-bearing inputs and
non-ordinal three-axis positions in the existing M1 causal arithmetic/state
witness and cross-M ownership, row permutation and padding checks. These witnesses
retain their existing scope: they do not establish an independently advancing
whole-model floating-point trajectory error bound. The shared visual traversal
currently requires an empty deepstack layer list; legacy eager backend support
and other visual architectures are separate qualification scopes.

Cross-M observers still retain only immutable GPU snapshots during the forward.
After every rank completes and proves the same event schedule, the auditor copies
the required physical-row span once and restores explicit row-map order on CPU.
Durable owner/page reads remain on the observing thread. Only immutable completed
CPU snapshots are hashed in parallel, with at most four pending rows and an
affinity/rank-bounded worker count. Field inventory, exact digest bytes and
protected write-set checks remain unchanged. Explicit startup numerical auditing
has a longer bounded readiness deadline to cover its reference and snapshot work;
child failure still aborts immediately, and readiness awaits the complete audit.

Mapped-host ordered sums initialize their existing rank-owned payload planes
before any peer registers the complete shared arena with CUDA. The calling
thread temporarily uses allowed CPUs local to its GPU when available, then
restores its original affinity; an all-rank capability decision precedes writes
and a rendezvous precedes registration. Rank zero initializes the flag tail.
This cold allocation step prevents arbitrary registration order from selecting
different NUMA page placement for calibration and retained buffers. It adds no
per-token host copies or affinity changes. The native kernel, generation flags,
rank-ordered FP32 addition and final half cast are unchanged. Caller affinity
and process memory policy still bound placement, so another topology requires
actual mapping observations and matched streaming evidence. Event durations
that include peer waiting must not be labelled pure transfer latency.
