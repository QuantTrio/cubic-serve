# Startup operator parameters

The [quantized execution contracts](QUANTIZATION.md) describe implemented native
eager FP8, dynamic NVFP4, AWQ and Cubic operators, including embedding, lm_head
and MTP weight consumers, and their remaining performance qualification boundary.

Use `cubic serve MODEL ... --operator-config /path/operators.json` in performance
mode, or pass `OperatorConfig.from_file(path)` as `operator_config` to the Python
executor's `start`. The parent parses the file once and sends an immutable value
to every rank. No file reads or policy selection occur during graph replay.
Omitting the option retains the existing launch parameters.

```json
{
  "version": 1,
  "profiles": [
    {
      "operator": "linear",
      "parameters": {"fixed_block_m": 16, "fixed_block_n": 64, "fixed_block_k": 64}
    },
    {
      "operator": "moe",
      "parameters": {"block_m": 16, "block_n": 64, "block_k": 128}
    }
  ]
}
```

Profiles may add `device` (exact CUDA device name), `model` (exact admitted model
ID printed at startup), and `shape`. Matching precedence, from lowest to highest:
built-in defaults, unscoped profiles, device profiles, model profiles, profiles
with both device and model, then top-level `overrides`. Within each tier, later
matching profiles override only the fields they specify. Shape specificity does
not imply precedence: put a narrower shape after a broader one in the same tier.

`shape` matches exact rank-local tensor dimensions. Linear accepts `n`, `k` for
`x[M,K] @ weight[N,K].T`. MoE accepts `hidden`, `intermediate`, `experts` (local),
`global_experts`, `top_k`. Model and device identities only select data; they do
not create model-specific operator implementations. Ordinary projections inside
MoE models use the same linear family as Dense models. Expert routing and grouped
GEMM parameters belong exclusively to the MoE family.

Supported parameters and built-in defaults:

| Linear | Default |
| --- | ---: |
| fixed_block_m / fixed_block_n / fixed_block_k | 16 / 64 / 64 |
| fixed_max_block_m | 32 |
| fixed_warps / fixed_stages | 4 / 3 |
| canonical_max_rows / canonical_warps | 8 / 8 |
| grouped_projections | true |

| MoE | Default |
| --- | ---: |
| block_m / block_n / block_k | 16 / 64 / 128 |
| compact_down_block_n | 128 |
| gemm_warps / gemm_stages | 4 / 3 |
| compact_min_routes_per_expert | 1 |

The compact path activates when physical M times top-k is at least global expert
count times `compact_min_routes_per_expert`; the kernels still consume actual
local route counts. This is the existing coarse heuristic, not a dynamic routed
workload tuner. Checkpoint expert ownership is unchanged.

For an explicit final override use, for example,
`"overrides": {"linear": {"fixed_block_n": 64}}`. Both keys and values are
validated before workers start. Tiles are powers of two from 16 to 256 (M at most
64); warps are 4 or 8, stages 1 through 4, canonical row grouping 1/2/4/8, and the
compact threshold a positive integer. These ranges describe accepted inputs,
not a claim that every combination fits or is qualified on every device.

`grouped_projections` is a boolean ordinary-linear option. When enabled, independent
GDN QKV + A/B/Z or attention Q/K/V projections sharing an input can share one
launch. Each member first resolves its own actual rank-local N/K profile. Grouping
requires every member to enable the option and resolve to identical parameters;
otherwise binding retains separate projections. This preserves explicit per-shape
tuning rather than resolving a fictitious combined N. It applies to ordinary
projections in both Dense and MoE; routed experts keep their separate family.

The grouped kernel calls the same ordered-K fixed-tile mathematical body for each
original weight/output pair. It introduces no packed weight copy or workspace and
preserves each BF16 output boundary. Group membership and resolved geometry enter
the artifact signature; leaf observers identify the actual grouped artifact.
An explicit configuration is:

```json
{"version":1,"overrides":{"linear":{"grouped_projections":true}}}
```

The corresponding upstream projection contracts were inspected at the pins below:
llama.cpp `src/models/qwen35.cpp` keeps GDN QKV/Z as separate projections;
vLLM `vllm/model_executor/layers/linear.py` and SGLang
`python/sglang/srt/layers/linear.py`, respectively,
provide QKV/merged column projections with explicit TP head ownership. Cubic retains
its original separate tensor layouts and head ownership while grouping launches.

M-based profile matching, kernel/reduction switching, precision changes, expert
remapping and audit bypasses are not configuration options. Target and draft
retain their existing reduction plans. Changing tile geometry or warps can still
change floating-point reduction, resource usage and performance: configuration
is not qualification. Every changed plan needs independent eager/reference,
cross-M row permutation, padding, state and distributed audits, followed by the
canonical streaming acceptance protocol. No tuned device/model presets ship here.

Startup logs the model, device and config content digest. Bound launch signatures include
resolved geometry; migration audit receipts and graph evidence include config
identity, preventing reuse of another configuration's cached audit. Modifying a
file after startup has no effect; restart to construct and qualify new graphs.

Upstream source basis: llama.cpp `4d9176092d00586775af140581bb0b558ddc4389`,
`ggml/src/ggml-cuda/mmvf.cu` separates matrix-vector launch geometry from operands;
vLLM `e962733e08d10f7ca65dac4df99e116460b8b174`,
`vllm/model_executor/determinism/batch_invariant.py`, and SGLang
`991368d8800cda6075c9f1b0be256af7cb44b8c8`,
`python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py` parameterize tiled
matmul while retaining an ordered K loop. Their tuning is not numerical evidence
or permission to import device-specific presets. There is no identical shared
configuration schema in the inspected paths; the precedence above is Cubic's
configuration contract.

## Quantized operator extension boundary

Version 1 preserves the existing `linear` and `moe` parameters and adds
`awq_linear`, `fp8_linear`, `nvfp4_linear` and `cubic_linear` with a shared typed
`PackedLinearParameters` schema. These families are accepted in explicit
quantized eager serving. Plain native FP16/BF16 linear uses `linear`; the sealed
BF16 fused-MoE implementation owns `moe`. There is no implicit quantized-to-BF16
fallback. See the linked contract for runnable syntax and qualification limits.

Keep three things separate when adding a format:

1. **Operand contract**, derived from the admitted checkpoint and actual tensors:
   weight and activation formats, packing, group/block scale geometry, scale dtype,
   zero-point convention, accumulation and output dtype. These are not arbitrary
   tuning overrides: they must agree with the loaded representation. An INT4 label
   alone does not establish compatibility between packing or quantization schemes.
2. **Execution parameters**, owned by that operator family: applicable kernel
   candidates, tile geometry, warps/stages, packing/repacking execution choices,
   workspace bounds and expert routing thresholds where relevant. Share parameter
   definitions only where their units and meaning are identical. Unsupported
   combinations fail during binding, before graph capture/readiness.
3. **Common configuration plumbing**: immutable parsing, scope precedence, exact
   shape matching, worker transport, startup diagnostics and evidence identity.
   Reuse this module's mechanism; add one typed schema and binder for a new family.
   Do not put format or model dispatch into the shared executor.

A mixed-precision model may contain several families in the same forward. Only
matching family profiles apply. BF16 overrides must never tune FP8/INT4 kernels,
and linear overrides must never modify routed-expert kernels. Qualification must
bind the format/layout contract together with resolved execution parameters;
changing either invalidates prior evidence. Reference math for a quantized
operator must follow its declared quantization semantics, not treat the original
BF16 model as a bit-exact oracle.

Additional inspected upstream sources at the pins above:
vLLM `vllm/model_executor/layers/quantization/base_config.py` separates weight
creation/processing and application by quantization method; SGLang
`python/sglang/srt/layers/quantization/awq/awq.py` validates bit width, group size
and zero-point configuration and distinguishes linear and MoE schemes;
llama.cpp `ggml/src/ggml-cuda/mmq.cuh` defines format-specific packed scale/data
layouts separately from MMVF. These support the boundary above; they do not
establish support for any new quantized kernel in Cubic.

## Rounding boundaries are not tuning parameters

Execution parameters may choose tiles, grouping and pipeline stages. They may not
remove an operator's declared BF16 intermediate or change nonlinear rounding.
In routed MoE, each weighted expert output rounds to BF16 before the next BF16
partial addition. The shared expert's gated output likewise rounds before routed
addition. Both combine launches disable FP fusion: Triton can otherwise lower
apparently explicit BF16 casts to `fma.rn.bf16x2`, skipping a required rounding.
Unowned expert scratch may contain stale values; route ownership masks, rather
than zero-filled scratch, determine which values participate.

Causal convolution evaluates SiLU after its FP32 product reduction. A fast
sigmoid approximation is not a reduction-order allowance. The convolution uses
full-precision exponential and correctly rounded FP32 division. Gated RMSNorm
also uses correctly rounded division for its FP32 SiLU gate: the checkpoint's
FP32 norm weight can expose a gate error at the final BF16 midpoint. BF16 and
FP32 checkpoint weight boundaries remain distinct; this does not change ordinary
linear or routed GEMM parameter selection.

Routing softmax also uses the full-precision exponential. A real concurrent C33
review exposed a fast-exp error crossing a BF16 route-weight midpoint, despite
unchanged expert IDs and exact cross-M identity. The independent selected-logit
softmax and a retained midpoint regression distinguish this nonlinear error from
denominator reduction order. This correction belongs only to the MoE routing
family and does not change ordinary linear kernels.

Regression coverage is in `tests/ops/test_cuda_gated_delta_decode.py`,
`tests/ops/test_cuda_qwen_pointwise.py`, and `tests/unit/test_cuda_moe.py`.
It includes real review midpoint operands, in-place state, C8/C32/C33 logical
rows, graph replay, padding poison and stale unowned expert routes. The regression
assertions fail with the previous nonlinear/combine implementations.

Semantic triangulation: llama.cpp `8b4b3558f1459c13e4aa38d5c94d306a00dc6acd`,
`src/models/qwen35moe.cpp` and `ggml/src/ggml-cuda/unary.cuh`; vLLM
`89af4f1ff792199b7c9260dd4a86feed445fd3d6`,
`vllm/model_executor/layers/fused_moe/fused_moe.py` and
`vllm/model_executor/layers/mamba/ops/{causal_conv1d,layernorm_gated}.py`;
SGLang `ba433bb4628f44d5031e910ad32194ad6c145ba8`,
`python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py` and
`python/sglang/kernels/ops/{mamba/causal_conv1d_triton,attention/fla/layernorm_gated}.py`.
They preserve the semantic order of expert weighting/combination and
normalization/gating, but differ in accumulation dtype, intermediate storage and
nonlinear approximation. Cubic's independent eager definition is the numerical
oracle; upstream implementations do not waive Cubic's explicit rounding steps.
The route normalization contract was also inspected in vLLM's
`vllm/model_executor/layers/fused_moe/router/fused_topk_router.py` and SGLang's
`python/sglang/srt/layers/moe/topk.py` at those same pinned commits. Both separate
route selection and normalized weights; their FP32 output/storage choices differ
from Cubic's BF16 route-weight boundary.


### Row reuse rule for linear kernels

`fixed_block_m` is the minimum row tile and `fixed_max_block_m` bounds row
reuse. At startup, for a captured input capacity `M`, the shared selector uses
`max(minimum, min(maximum, 2**floor(log2(M))))`. Thus a wider tile is used only
when enough input rows exist. An explicit minimum wins if it exceeds the upper
bound; setting both to the same value restores a fixed row tile.

The rule uses tensor capacity and operator parameters, never a device name,
model identifier, or public concurrency. Ordinary and grouped BF16 projections,
and plain BF16/FP16 tensors in packed execution share the selector. Cubic owns
its decoded-weight work estimator in `../quantization/extensions/cubic/policy.py`.
Their parameter families remain separate (`linear` versus `cubic_linear`);
routed/fused MoE retains its own strategy. K tiling, partition
order, FP32 accumulation and output rounding do not change with the row tile.
Cross-M output/state identity must still be qualified on the retained kernels.

A K-by-N indexing expression is not proof of poor coalescing. Direct BF16/FP16
loads and decoded packed operands can compile differently; measure both. Layout,
K splitting and row reuse have distinct tradeoffs, and a packed-format speedup
must not be copied into ordinary linear kernels without its own evidence.

### Canonical TP transport cost

By default, `collective.transport_calibration` measures ordered gather/sum and
partition-exchange/sum/gather once per actual process group, dtype and policy,
before graph capture. Only CPU coefficients are cached with weak group ownership.
Interleaved graph measurements use the slowest rank and retain neither probe
graphs nor GPU buffers. Two positive affine bandwidth regimes describe changing
channel efficiency; independent payloads validate the lower cost envelope. The
rule is shared by ordinary linears, MTP sums and routed-MoE completion. It does
not depend on public concurrency, tensor row buckets or a GPU/model name.

`transport_calibration_bytes` and `transport_calibration_replays` bound the cold
work; `transport_min_gain_percent` and `transport_max_fit_error_percent` control
selection margins. Invalid fits, inconclusive differences and payloads outside
the measured interval retain the analytical rule below. Explicitly setting
`exchange_latency_bytes` takes precedence and disables automatic transport fitting.
Numerical or execution failures are never treated as unusable timing evidence.

For standalone FP16/BF16 completion, `collective.host_sum` additionally admits
mapped-host ordered reduction when all peers can register the same private host
arena. `host_sum_threads` bounds the vector-copy CTA (default128 threads); block
count follows payload size and SM count. The same positive-affine cost family
estimates this transport, with its own held-out error. It must beat the
selected NCCL path by `transport_min_gain_percent` inside the measured interval.
Fused residual/RMSNorm and routed completion retain their own completion ABIs.

Decode interleaving prices collective geometry together with the child GEMMs.
Unless `host_sum_threads` is explicitly configured, the bounded candidate set
includes the original geometry and the next wider supported CTA; all ranks use
its common intersection. A wider CTA can reduce SM coverage for the same payload
and avoid interference even when its isolated reduction is slightly slower.
The selection score is `min(control_before, control_after) /
max(candidate_first, candidate_second)` over restored identical operands and the
slowest rank, using `decode_overlap_samples` and `decode_overlap_min_gain_percent`.
This is a measured complete-round cost comparison, not a device or concurrency
threshold. Sequential packed candidates release obsolete graphs before testing
the next resource variant within `decode_overlap_max_bytes`. The selected parts,
linear strategy and collective threads survive final cache rebind as one immutable
plan; explicit configuration and arithmetic boundaries are preserved. Numerical
qualification and canonical public streaming measurements remain separate gates.

`collective.host_sum_partition` admits an additional two-shot host path. Each
rank reduces disjoint output stripes in the same ascending-rank FP32 order,
rounds each completed element once, then shares those completed stripes. A
whole CTA serves each rank's stripe: partitioning must not leave most threads
idle inside every CTA. The launch grid derives from element count, rank count,
vector width, configured threads and available SMs. Publication and consumption
use identical stripe ownership, including incomplete final vectors. The extra
publication phase and reduced replicated reads are priced by the same affine
cost family, with a separate held-out error. This path must beat the already
selected NCCL or one-shot host path; there is no fixed message-size, model, SM
generation or concurrency threshold. Disabling it preserves the one-shot path.

The host kernel polls independent peers in parallel and keeps separate ready and
consumed generations. Producer system fences precede publication; no peer may
overwrite a generation until every reader finished it. Each primitive owns its
mapping, epoch buffer and immutable pointer descriptor through graph/lease
completion. File unlink waits for the CPU-observed collective completion proving
all peers opened and registered it; an enqueued NCCL call alone is insufficient.
Native operators share content-addressed CUDA13.0 compilation with Marlin while
keeping separate source manifests and cache namespaces. This is baseline PTX
linked for the current device, not an AOT multi-device qualification claim.

`collective.exchange_latency_bytes` expresses the extra exchange startup as
equivalent transferred bytes. For payload B and group size R, full all-gather
transfers `(R-1)*B` bytes per rank; partition exchange plus completed-partition
gather transfers `2*(R-1)*B/R` with one extra collective startup. The lower
estimated cost chooses the transport before graph capture. There is no upper
payload cutoff that reverses the decision. This parameter uses the same startup
operator configuration as linear and MoE, and is independent of their kernels.
Both transports retain ascending source-rank FP32 addition and one BF16 result
rounding. No NCCL reduction tree replaces that arithmetic contract.

For expert routes `[M,K,H]`, `cuda/moe/route_completion.py` can finish whole row
partitions and apply the original weighted top-K fold before gathering `[M,H]`.
Each route still observes the original rank-ordered FP32 sum and low-dtype store;
each weighted product and each addition still observes its original rounding.
This is temporary transport ownership, not expert redistribution. Ordinary
linear reductions and paths with a different completion contract retain their
own strategies.

Complete-K activation scale reductions are a separate producer contract. A
row-parallel FP8 expert down projection must retain its cross-rank MAX scale
even when row completion replaces the legacy route-sum callback. The presence
of that callback alone cannot identify tensor-parallel arithmetic.

With `P = ceil(M/R)*R` and element size `s`, this path's transport estimate is
`(R-1)*P*H*s*(K+1)/R`. If `P != M`, price the three input copies and output trim
separately, including the actual route-ID width (INT32 or INT64). All extra rows
are private scratch. Workspaces belong to one graph/stage lane; only sequential
bindings in that lane share them. Both kernels and all operand addresses are
bound and checked before the first transfer.

`RouteCompletionCost` fits startup plus inverse bandwidth on the actual process
group, activation dtype, index dtype and tensor geometry. Cold probes measure
the slowest rank and release their graphs and tensors afterwards. A smallest-row
gather witness prevents a bandwidth-region intercept from mispricing C1. Held-out
measurements check the domains where each transport actually executes; a padding
fit does not disable a path that performs no padding copies. Unusable fits retain
the original ordered transport. Runtime dispatch consults no device/model/C table.

The `collective` parameter family exposes `route_completion`,
`route_completion_calibration_bytes`, `route_completion_calibration_replays`,
`route_completion_min_gain_percent`, and `route_completion_max_fit_error_percent`.
Sampling count reuses `launch.calibration_samples`. These control a cold sampling
budget or a decision margin, not production batch thresholds. Fits are shared
within the owning worker; per-capacity workspaces remain separate. A primitive
or cost fit is not a full-model numerical or performance-mode qualification.

Calibration repetitions are captured inside one timing graph. Submitting a
separate tiny graph for each repetition from Python prices host submission gaps
as transport work, distorting the startup intercept. This timing rule applies
to these immutable private probe operands; repeating stateful model steps would
require their own reset contract. Keep both actual selected paths and fit
validity in the evidence, since a rejected fit can select the original transport.
