# Text activation precision

`serve --activation-dtype auto` (default) follows the text activation storage
dtype. The native BF16 exponent instruction requires SM90; on SM80–SM89 the
activation computation uses FP16. `serve --activation-dtype fp32` forces the
wide activation rule. The setting does not change weight encoding, KV/state
storage, normalization weights, matrix accumulation, or MRoPE's FP32 rotation.

`--activation-dtype auto --softmax-dtype fp32` keeps SiLU/sigmoid in the
model's low precision while computing attention and router softmax in FP32.
The independent `softmax_dtype` OperatorConfig field defaults to `fp32`.
An explicit `auto` inherits `activation_dtype`. Explicit CLI values override JSON values. Both
fields participate in configuration digests; attention fingerprints and MoE
signatures bind the effective softmax arithmetic before capture. Route weights
still round at their declared storage boundary. Sampling remains FP32.
This mixed mode is an explicit diagnostic option; its full-model qualification
must be evaluated separately from either uniform precision mode.

The same startup `OperatorConfig` carries `activation_dtype` through reference
execution, quantized eager worker threads, and performance rank processes. It
is part of the configuration digest and bound CUDA specialization/signature.
An optional JSON `activation_dtype` field accepts `auto` or `fp32`; an explicit
CLI setting overrides it. Arithmetic is not selected by model name, M, C, or a
linear/MoE tuning profile.

## Arithmetic boundaries

- Native sigmoid uses `exp(-abs(x))` and sign-dependent normalization, so its
  exponential never overflows. Native SiLU multiplies the resulting rounded
  gate in the selected activation precision. A finite BF16 positive tail is
  preserved when the FP16 fallback cannot represent that input. The 65504
  range comparison is wide: comparing a BF16 tensor directly with that scalar
  can round the cutoff to 65536 and miss the first overflowing BF16 value.
- Native exponentiation uses PTX `ex2.approx.f16` or
  `ex2.approx.ftz.bf16`, with a low-precision base-conversion product. These
  instructions have documented approximation errors; BF16 explicitly flushes
  subnormal exponential results. They are not bitwise substitutes for FP32 exp.
- PTX has no half-precision division operation. Scalar activation division is
  explicitly FP32 RN internally, followed immediately by the selected low
  precision boundary. This exception is not advertised as native FP16 division.
- Softmax uses a wide max subtraction, low-precision exponentials, and FP32
  sums/normalization. Paged attention also keeps online rescaling, split merging
  and value accumulation in FP32. The wide-dot attention path has no
  low-precision probability boundary. The low-dot path explicitly rounds each
  tile's unnormalized probability to V dtype before its MMA; these two
  mathematical rules must not share an oracle or reduction-only witness.
  Stored MoE route weights have their declared storage boundary.
  Pure FP16 denominator storage overflows for 65,536 equal logits.
- MoE selection ranks original logits before normalization. Probability rounding
  must not create new expert-ID ties or alter expert ownership.
- Sampling's distribution/CDF, recurrent decay/softplus, and normalization
  reductions are separate stability rules; this switch does not lower them.
- Following SwiGLU/output-gate multiplies use the activation storage dtype.
  Gated RMSNorm retains a checkpoint-declared FP32 weighted operand as FP32.

The eager implementation independently evaluates staged low-precision rules
with PyTorch; it does not call the candidate CUDA activation implementation.
Its exp2 result is the mathematical reference, not a claim about native
instruction selection. Native approximation requires an independent numerical
interval witness in addition to capture and cross-M identity.

## Qualification scope

The local SM120 operator evidence includes every finite FP16/BF16 input for the
exponential/sigmoid/SiLU stages, documented exp2 error enclosures, explicit
post-exponential rounding, captured/permuted M1/M8/M33 SwiGLU, and 64K/128K
uniform-context paged attention. Historical midpoint/state/reduction tests
explicitly exercise the forced FP32 mode. These two evidence scopes must not
be conflated.

Native low-precision full-model CUDA Graph, speculative and distributed
qualification is separate and remains required. Old FP32 nonlinear audit
receipts cannot qualify `auto`; a reduction-only error allowance cannot excuse
native exponential approximation. No throughput gain is inferred from an
instruction count, operator test, or eager model audit.

Source pins and diagnostic evidence are recorded under
`/workspace/artifacts/activation-dtype/`. PTX ISA source:
https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#half-precision-floating-point-instructions-ex2

## Attention and GDN operands

`--attention-dtype auto` selects matching FP16/BF16 Q/K/V operands for QK and
P×V Tensor Core products, with FP32 dot accumulators. It is independent of
`activation_dtype` and `softmax_dtype`. `--attention-dtype fp32` retains the
wide scalar-product control path. `auto` is the default. The JSON field
`attention_dtype` accepts the same values; an explicit CLI argument wins.
The resolved choice is immutable before capture and participates in config,
artifact, launch and numerical fingerprints.

For the requested isolated comparison use:

```
--activation-dtype auto --softmax-dtype fp32 --attention-dtype auto
```

The low-dot kernel groups up to 16 query heads sharing one KV head; this rule
has no model, device-generation, concurrency or logical-M branch. The fixed
32-way KV partition and 16-token tiles are unchanged. Each head independently
publishes its split result and acquires all other splits before merging.
Workspace ownership and graph leases remain the caller's responsibility.

Only dot operands change: no FP32 Q/K/V expansion occurs in the candidate.
The MMA product is not separately rounded to FP16/BF16 before FP32 addition.
Softmax scaling, max subtraction, denominator, online rescaling and split
normalization retain their explicit safe arithmetic contract; these are not
advertised as low-precision products. Output and KV storage are unchanged.
The independent eager oracle models low-operand products with wide scalar
math and explicitly rounds the unnormalized P operand at the same boundary.

Local tests check actual MMA operand/accumulator instructions, independent
staged-reference agreement, FP16 product overflow avoidance, and graph identity
with poisoned padding and permuted M1/M8/M33 rows. These are operator diagnostics,
not full-model numerical or canonical streaming performance qualification.
The historical FP32 reassociation witness fails closed for the new P boundary.

## GDN recurrence precision

The default serving policy is `attention_dtype=auto`, `activation_dtype=auto`,
`softmax_dtype=fp32`. Explicit flags or JSON can override each field. Startup
logs do not emit separate precision-policy notices.

The same attention switch controls both full attention and GDN. GDN `auto`
uses model-dtype Q/K/V and state multiplication operands. Stable Q/K
normalization and softplus/exp coefficient generation retain FP32 arithmetic;
normalized Q/K, the decay coefficient and beta are converted to the operand
dtype before recurrence products. State decay, residual subtraction, beta
multiplication and query scaling have explicit low-dtype boundaries. Packed
BF16/FP16 FMA implements the pointwise operations, including BF16 on SM80+.
State×K and state×Q use low MMA operands and FP32 accumulation. The rank-one
state update keeps its low-MMA product in FP32 registers and performs an explicit
FP32 RN addition in the same kernel, without an intermediate low-dtype product
rounding. A nonzero MMA accumulator does not guarantee IEEE RN addition and
can choose the wrong BF16 state at a midpoint; it is not used for this boundary. The generic tile does not depend on model, SM, M or concurrency.

GDN value-row tiling is bound through the existing operator configuration:

```json
{"version": 1, "overrides": {"gdn": {"native_value_tile": 64}}}
```

`native_value_tile` accepts 16, 32, 64 or 128, defaults to 64, and is clamped
to the padded value dimension (at least 16 for MMA). Shape-scoped profiles
can match `q_heads`, `value_heads`, `key_dim` and `value_dim`; these are the
actual rank-local tensor dimensions. There is no concurrency selector. The
FP32 control path is unaffected. This changes independent value-row grouping,
not the K reduction, operand precision, state rounding or ownership rules.
The actual tile is included in the primitive's launch signature.

Larger tiles amortize the native path's matrix-layout conversions, barriers
and block scheduling, but can reduce parallelism at low M. They do not remove
the zero-filled MMA columns used for matrix-vector contractions or the padded
rank-one update. Low operand precision alone therefore does not imply a faster
GDN implementation; qualify both numerical identity and public serving gains.

After the update, BF16 state is rounded before both current output and next
step consumption. A retained FP32 state remains FP32 storage; `auto` still
converts its multiplication operands to model dtype. The explicit `fp32`
attention option retains the historical wide recurrence with separate RN
multiply/add operations. Changing activation precision alone does not choose
recurrence precision. Checkpoint ownership, segment continuation, COW and
speculative checkpoint destinations are unchanged.

The independent eager implementation models the same low boundaries. It uses
wide products of already-low operands for reduction reference math and FP64
product-plus-add followed by FP32 rounding to witness the fused rank-one update;
those wide oracle temporaries are not candidate CUDA computation. Validation
must distinguish allowed changes of arithmetic from implementation errors, and
must retain the agreed fusion/precision policy while repairing such errors.

PTX instruction boundary reference:
https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#half-precision-floating-point-instructions-fma
