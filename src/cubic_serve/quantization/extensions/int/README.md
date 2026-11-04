# INT WNA16 and WNA8

The public extension is `int`, with encoding `int-pack-v1` and parameter families
`int_linear` / `int_moe`. It implements ordinary signed INT2–INT8 affine weights,
including the minimum signed code. No Cubic codebook, reserved-code rule, or
Cubic quantization execution implementation is used. WNA8 is admitted for
supported original packed INT2–INT8/F16-scale linear and routed MoE weights
through the shared native Marlin path.

WNA16 retains integer weight storage, reconstructs `(q - zero) * scale` in FP32,
rounds to the checkpoint scale dtype, then converts to the FP16/BF16 tensor-core
operand dtype. Accumulation is FP32. This is weight-only quantization, not INT8
activation MMA. Symmetric weights have zero = 0; asymmetric packed zeros use
the same offset-binary codes as weights. F16, BF16 and F32 scales are admitted.

WNA8 quantizes each projection input to signed INT8 with a row scale, executes
the packed-weight integer MMA in native Marlin, and applies the format-owned
scale and zero arithmetic at the output boundary. WNA8 does not promise an
INT8 hidden-state flow between model operations. A requested WNA8 projection
cannot silently dispatch WNA16; an unsupported scale carrier or geometry fails
at binding. Dense A8 reuses the shared measured Marlin row-tile selector, including
a resource-compatible BM64/stage2 candidate for larger physical M. K partitions
remain a separate startup plan; adding partitions is not assumed beneficial
when N already supplies enough output tiles.

## Checkpoint ABI

The compressed-tensors `pack-quantized` container is supported. The root format
can supply groups that omit their own format. Static group/channel weights,
identity activations and no activation ordering are required. Logical matrices
are `[N,K]`; packed weights are INT32 words along K, scales are `[N,ceil(K/G)]`,
and asymmetric zeros are packed along N. `weight_shape` is INT64 `[2]` and its
values must agree with the model geometry.

Two physical layouts exist in compressed-tensors upstream:

* `legacy`: floor(32/bits) offset-binary codes per word, unused high bits.
* `dense`: contiguous bitstream, including codes crossing word boundaries.

Header dimensions infer the layout when unique. Ambiguous odd-bit headers require
an explicit `weights.packing` (`dense` or `legacy`); they are never guessed.
For divisors of 32 both layouts are identical. `weights.scale_dtype` may declare
the scale carrier; otherwise admission derives it from headers. These optional
fields are Cubic Serve loader hints, not claims about the upstream CT schema.
For interchange with a particular CT release, emit its actual packing ABI rather
than relying on another loader accepting extra configuration fields.

The Qwen3.8-Flash-Next-Int-4Bit checkpoint inspected here uses legacy packing,
including 5- and 6-bit weights; interpreting those as dense corrupts the codes.
This is an exporter-version compatibility distinction, not evidence of corrupt
weights. Explicit packing metadata is recommended for unambiguous interchange.
TP/EP slices preserve declared row/expert ownership and original group scales.
Unaligned K slices rebase scale groups by their greatest common divisor without
requantization. Row-only materialization copies packed rows without expansion.

## Execution and limits

Dense and routed MoE execution use the shared Marlin WNA16 template with an
INT-owned affine decoder. Repacking preserves B-bit weight storage for every
B in 2..8. Original F16 scales and exactly representable integer zeros share
one four-byte carrier per group/column; Marlin half2 decoding preserves the
original F16 rounding before conversion to the operand dtype. Original BF16/F32
scales retain separate FP32 scale/zero planes. Geometry
selection reuses the shared measured cost model, prices metadata by the original
group-to-pipeline-quantum ratio, and uses no model/device patch table.
The shared MoE ABI requires a homogeneous encoding across the local experts of
each projection; gate, up and down may have different encodings.
The within-CTA K reduction factor (`warps * 64 / block_n`) and N/warp geometry
remain invariant across M.
Dense linear work can additionally split K across CTAs. Its sparse-row partition
plan is selected at startup from tensor geometry and measured device costs. INT
WNA8 dense bindings then cap that plan for each existing physical graph capacity:
`tiles = ceil(M / block_m) * (N / block_n)` and required partition supply is
`ceil(SM_count / tiles)`. Choose the smallest legal K divisor meeting that supply
without exceeding the original plan or an explicit user cap. A single row tile
retains the original plan. This removes excess partial-write/merge traffic rather
than retaining more intermediate values; it is not a model, SKU or client-C table.
Each graph seals its reduction before replay, and warm binding restores its plan
without timing searches. Independent cross-M anchors restore the actual bound
tile and partition recipe instead of repricing it as an M1 workload.

WNA16 uses the same capacity rule inside complete residual-stream mixer blocks.
The common graph builder scopes this startup binding hint to the complete block;
the INT backend owns legal divisions and records its actual reduction recipe.
Other WNA16 blocks retain their existing plans. This requires no model/GPU name
branch, environment switch or alternate serving entry point. WNA8 keeps its
existing capacity plans and audit inventory identities.

Legal counts divide `K / lcm(metadata_quantum, block_k)`. The affine decoder
indexes the original group by absolute K, so a compute partition does not
require a new scale group or any change to serialized weights. Candidate counts bracket
one SM wave and the thread/shared-memory residency bound. The common calibrator
fits nonnegative coefficients for `time = base + issue * extra_waves + traffic *
bytes`, including packed weights, metadata, cache capacity and partial merge.
Independent held-out shapes bound prediction error. Selection minimizes the
worst uncertainty-adjusted relative cost across one row, a full row tile and
SM-covering rows. The sparse-row plan cache prevents a new logical M from
refitting the reduction order; WNA8 capacity caps above are also cached and are
immutable within each bound graph. `marlin_max_k_partitions=0` derives the search bound; a positive
value is an explicit user cap. `marlin_max_candidates` limits calibration work,
not the maximum partition count. This rule needs real-device qualification for
each advertised SM family.

Compatible grouped projections align their combined output width to
`lcm(64, block_n)`, adding integer-zero rows only after the final member.
Codes and metadata are packed over that complete axis, preserving expert order;
N160 gate/up therefore uses N320 rather than two independently padded N192
members. Routed groups with no trailing padding write their output directly.
Linear output widths are padded with integer-zero rows to the native N tile;
the result is sliced back to its original logical width. Original scales and
codes are preserved, and selection costs use the padded physical geometry.
For dense linear, an unaligned K first derives a smaller legal tile with
`block_k = gcd(K, base_block_k)`. The N tile must satisfy
`block_k * block_n` divisible by `threads * 64`; the least compatible multiple
of the base N tile supplies that alignment. This reuses the existing Marlin tile
family and respects explicit K/N tile overrides. No K padding is introduced.
The physical pipeline group quantum is at least 16. Linear and routed MoE
supports original boundaries aligned to eight K elements by carrying separate
scale/zero metadata for the two K8 halves of a tensor-core fragment. G40/K320
can therefore use K64/N128 tiles, and aligned G72 uses the same decoder hook.
The serialized scale groups and integer codes remain unchanged. K32/N256 is
another legal instance of the same load-alignment equation, covering K160.
Mixed gate/up encodings use separate Marlin affine decoders and original weight
storage, then copy their outputs into the shared projection buffer. Their row
tile comes from the common configured geometry so both consume the same route
prefix. Homogeneous gate/up retains the fused projection launch.
INT GEMMs require a Marlin binding: an unsupported native geometry fails at
startup with its shape, encoding and parameters rather than entering a generic
Triton GEMM. Such a failure is an adaptation gap, not a supported execution path.
Embedding lookup and projection-output copies remain format-owned auxiliary
kernels; they are not matrix multiplications.
Packed embeddings use one shard-owner lookup kernel, preserve explicit row
activity and emit zero for active rows owned elsewhere. The caller owns reduction.

## Qwen3.8 TP4 WNA8 qualification on local SM120

The final TP4/PP1/EP1 graph on physical GPUs 4–7 has 389 dense and 96 routed
native INT A8 Marlin launches per rank at both M1 and M128, with no quantized
WNA16 GEMM. PLE hash/table lookup and embedding remain lookup work; PLE key/value
projections use A8 Marlin. The independent original-weight audit passed all four
ranks, physical M1–M256, reversed row maps and state checks at the automatically
admitted 262,144-token context. The complete public streaming Chat Completions
C1–C256 matrix requested 1,024 outputs per request and completed every request.
C1 repeated at 136.47 and 136.05 output tokens/s/request; C128 measured 32.68
and C256 21.18, versus matched WNA16 controls 28.90 and 17.99 at C128/C256.
One 30,000-token xhigh request completed at 126.20 tokens/s with a tail/head
ratio of 0.929. During an added 6,095-token prefill, 32 active decode streams
had at most 39.2 ms between output tokens. These measurements qualify this local
TP4 topology, not other devices. The TP2 startup on GPUs 6–7 loaded weights but
failed closed before serving because the required BF16 KV/state pool did not fit
after graph residency; TP2 needs a separate memory-offload design and audit.
Evidence: `/workspace/artifacts/wna8-attention-profile-20260925/` files
`final_tp4-public-comparison.json`, `final_wna16_vs_wna8_perop.json`,
`final_wna8_tp4-c256-audit-server.log`,
`wna8_tp4_long_stagger-{long,stagger}-public.json`, and
`wna8_tp2_feasibility-server.log`.

The earlier operator-only measurements below are not serving-performance
qualification. Operator tests cover integer
codes, owner slicing, FP16/BF16 linear and MoE, routing/SwiGLU/down, padding,
permutations, graph replay and embeddings on SM120. CUDA 13 compilation of the
baseline linear templates for SM80/86/87/88/89/90/100/103/110/120 proves compilation
only; other hardware families still require real-device numerical/performance
audits. Canonical full-model streaming, state and topology qualification must be
reported separately from these tests.

The adaptation baseline is pure TP: TP4/PP1/EP1. The historical four-rank
measurements below used TP4/PP1/EP4 and do not qualify pure TP. EP1 preserves
all logical experts on each TP rank and partitions their matrices; its startup,
numerical audits and public streaming results must be measured separately.

Pure TP routed projections reuse integer-zero N padding and the linear gcd/lcm
K-tile rule for local TP shapes (including N=160 and K=160/288). Dense TP K spans
whose endpoint bit offsets are both divisible by 32 retain the serialized word
interval directly; scale-group normalization remains unchanged. The retained
SM120 TP4/PP1/EP1 implementation passes four-rank independent original-weight and
M1/M2/M4 cross-M/state audits, all 17 INT tests, and 280 ten-SM CUDA13 builds for
routed K32 with two/four metadata planes. Public C1 at 1024 requested tokens and
context4096 measures 79.14/78.77 tokens/s; the near-100 target and full concurrency
frontier remain unqualified. Rank weight loading decreased from about 258–270 s
to 115–117 s in the recorded startup runs; these phase timings are not a claim
about total warm startup or serving throughput. Evidence:
`/workspace/artifacts/int-tp-marlin-fix-20260920/qualification.json`.

The N-padding, dense-partition and K8-linear revisions of the inspected checkpoint
each passed an independent original-weight
M1 audit on TP4/EP4 SM120, followed by four request corpora across physical
M1/M2/M4, boundary logical row counts, reversed row maps and poisoned private
padding. Common outputs, routes, collective results, BF16 KV/recurrent state
and durable write sets agree bit-for-bit. This bounded result uses FP32
activation/attention/softmax and BF16 WNA16 operands; it does not qualify other
topologies, larger buckets, long contexts or the complete concurrency frontier.
The K8-linear revision measures 83.30 public C1 tokens/s in the debug graph server
and 80.40 in the audited CUDA server launched under inactive Nsight instrumentation,
using identical 1,024-token requests. These results remain below the requested
target near 100 tokens/s.

The required-Marlin revision records 252 distinct rank/geometry/encoding/M
bindings across four ranks and physical M1/M2/M4: all INT GEMMs are native.
INT2-8 routed G40/G72 and mixed gate/up tests pass, as do the independent
four-rank original-weight and cross-M/state audits. The deferred typed audit
reuses the existing `route_fp32_exp_sum_division` boundary proof on independently
proved actual logits before consuming stored route probabilities; stable expert
IDs and cross-M route/state identity remain exact. CUDA13 compilation passes
280 routed G40/G72 combinations across the ten baseline SM targets, which is
still compilation-only evidence outside SM120.
Canonical C1 measures 81.65 tokens/s under the audited CUDA server with an
inactive Nsight launcher. This does not meet the near-100 target and must not be
mixed with debug-mode reference results. Scope and artifacts are recorded in
`/workspace/artifacts/int-c1-100-20260919/marlin-required-qualification.json`.

The affine metadata decoder compiles its immutable physical N and original group
count from the bound N/K geometry. Calibration uses the same dimensions; only
active experts and rows vary. This lets the compiler simplify flattened-address
division without changing metadata storage, original scale boundaries or K
reduction. The generated-source cache includes those constants. Runtime logical
M, routing and checkpoint expert ownership remain unchanged.
On SM120 TP4/EP4, paired 1,024-token public C1 debug controls average 86.80 tokens/s
and the candidate averages 90.10 (+3.79%). The final retained CUDA revision passes
four-rank independent original-weight and M1/M2/M4 cross-M/state audits and measures
89.29/89.07 tokens/s. All 15 INT operator tests and 140 CUDA13 builds across the ten
baseline SM targets pass. First-use compilation now creates geometry-specific
artifacts; this cold-start cost is separate from steady throughput. The near-100
target, full concurrency frontier and other physical devices remain unqualified.
Evidence: `/workspace/artifacts/int-register-pipeline-20260920/qualification.json`.

## Pinned implementation comparisons

* compressed-tensors `72aad1603356daff9d27775041d725cec4f59842`,
  `src/compressed_tensors/compressors/pack_quantized/{base,helpers}.py` and
  `src/compressed_tensors/quantization/lifecycle/forward.py`: dense current packing,
  scale-dtype reconstruction. Helpers at `9c3dd050d29b8b2489527889c78a77b2391d7909`
  use legacy whole-word packing.
* vLLM `41c4a3ed4e45792739bd8a5bbcd01269b06eae2c`,
  `vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16.py`
  and `vllm/model_executor/layers/quantization/utils/marlin_utils.py`: WNA16 integer
  dispatch and narrower native group/asymmetric contracts. `marlin_padded_nk`
  pads physical tiles while preserving logical output dimensions. Qwen4 ngram embedding
  keeps original table row ownership; upstream lookup accepts plain/FP8 tables.
* SGLang `9e2298e913b32548d7381977159d676871424ffb`,
  `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16.py`
  and `python/sglang/srt/layers/quantization/marlin_utils.py`: packed WNA16 and
  routing remain separate contracts; tile padding preserves complete scale groups.
* llama.cpp `1af554f8fc78ba029665a47b839484d9763e2a75`,
  `conversion/qwen4exp.py` and `src/models/qwen4exp.cpp`: Qwen4Exp logical embedding
  and expert identity. GGUF is a different ABI, not a CT packed-weight decoder.

These are source comparisons, not numerical oracles. Tests use independent
integer extraction and Cubic Serve's eager/reference arithmetic.

Relevant upstream PR review also includes vLLM
[#52079](https://github.com/vllm-project/vllm/pull/52079) at
`a47948a94ac599eba860061e1e51218219e3608d` (Blackwell BF16 GEMM-RS, M>=128,
one NVLink domain), and [#50484](https://github.com/vllm-project/vllm/pull/50484)
at `4b4c1d0214d9467fac4cfe4aec0b9275443afddc` (direct-final DCP query publication,
system-scope epoch synchronization, empty-shard/padded-row LSE handling).
These contribute dataflow and buffer-lifetime lessons, not a portable replacement
for native INT Marlin. Detailed source facts, review constraints and related
SGLang/llama.cpp PRs: `/workspace/artifacts/int-tp-marlin-fix-20260920/pr-review.json`.
