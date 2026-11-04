# Quantized execution contracts

The implemented functionality is explicit **native eager text execution**, on
one CUDA device, for AWQ GEMM, channel FP8, compressed-tensors NVFP4 and Cubic
packed checkpoints. These are weight/activation formats, not model-name
selectors. Dynamic NVFP4 is the intended term; NVFP8 was a typo.

Performance CUDA Graph execution, TP/PP/EP quantized execution, multimodal
execution and speculative serving are not qualified by this implementation.
They fail admission instead of falling back to eager execution. MTP weights and
MTP forward math are included in the independent audits; that does not qualify
the speculative acceptance/repair algorithm. Short SSE answers are functional
smoke evidence, never throughput measurements.

## Ownership and retained implementation

| Responsibility | Source | Contract |
| --- | --- | --- |
| Model graph and logical weights | `models/qwen3_5/weights.py`, `models/qwen3_5_moe/weights.py` | Model-owned aliases, original Q/K/V partitions, language/head/MTP/vision ownership. Only the MoE planner expands logical expert IDs. |
| Format semantics | `models/quantization.py` | Immutable `QuantizationScheme`, typed weight/activation formats, exporter-specific rule resolution. |
| Physical admission | `models/quantized_checkpoint.py` | `LogicalWeight` → `EncodedWeight`; complete physical tensor closure; immutable `QuantizedMatrix` payload binding. |
| Native linear/lookup | `ops/cuda/linear/quantized.py` | Separate packed format kernels; FP16/BF16 plain linear; no full decoded weight allocation. |
| Independent oracle | `ops/eager/quantization.py` | Bounded decoded-weight tiles, explicit codes/scales and arithmetic. Not a serving executor. |
| Loading and native eager execution | `runtime/quantized.py` | Original packed payloads, compute dtype for plain members, shared token mailbox and sampling policy. |
| Public chat transport | `runtime/token_chat.py` | Format-independent token, template, codec, cancellation and SSE handling; `performance_chat.py` retains compatibility names. |
| Reference construction | `runtime/quantized_reference.py` | Explicitly selects independent operators; has no public executor. |
| Audit | `guardrail/quantization.py`, `tools/quantization/audit_checkpoint.py` | Deferred immutable observations, unrestricted independent forward, full-output/full-K causal checks, state and row identity. |

Ordinary linear in Dense and MoE uses the same operation interface. Expert
selection, expert identity/order, shared-expert gating and rounded combination
remain model-owned MoE math. The eager quantized implementation executes these
ordinary expert linears separately; it does not pretend to implement a fused
quantized MoE kernel or inherit the BF16 fused-MoE tuning strategy. No EPLB,
expert replication or traffic-derived remapping is introduced.

The canonical model attention, normalization and recurrent-state math are reused.
Quantization adaptation does not require replacing attention. In particular,
a batch-dependent SDPA dispatch is unsuitable for the required common-row
identity. Cross-M equality remains a separate check from independent numerical
qualification.

## Dtype and representation

`--dtype fp16` / `float16` makes the ordinary execution path FP16; `bf16` /
`bfloat16` selects BF16. `auto` uses the repository text dtype. Ordinary linear
weights, activations, hidden outputs and KV follow this execution dtype.
Model-defined FP32 stability tensors and accumulation are preserved. Recurrent
state has its explicit `--mamba-ssm-cache-dtype` boundary; both current output and
subsequent input consume the rounded durable state.

Packed payload dtypes and scale/curve dtypes remain format-defined. They are
not globally converted by `--dtype`. Plain linear loads real FP16/BF16 weights
and activations and accumulates in FP32. Expanding a plain weight to FP32 and
calling a reference matrix multiply is **not** the native serving operation.
Independent FP32/FP64 audit math is confined to reference/guardrail code and
bounded weight tiles.

| Format | Serialized representation and arithmetic |
| --- | --- |
| AWQ GEMM | I32 `[K,N/8]` words, output nibble order `[0,4,1,5,2,6,3,7]`, direct packed zero points, FP16 group scales. Decode `(q-z)*s` inside a tile, round W to execution dtype, then A16 GEMM. |
| Channel FP8 | E4M3 `[N,K]` weights and FP32 `[N,1]` scales. CT dynamic `strategy=tensor` activation metadata maps to its per-token dynamic E4M3 runtime rule; scale floor is `1/(448*512)`. Preserve rounded activation codes and scales. |
| CT NVFP4 | Adjacent low/high FP4 nibbles, E4M3 block scales over 16 inputs, FP32 global inverse scales. Dynamic activation is local group-16 E4M3 scale plus FP4 ties-to-even codes. Accumulate scaled group products with original global divisors. This is not GGUF's NVFP4 layout. |
| Cubic | Little-endian signed 1–8-bit codes; reserved most-negative code is zero (binary ±1 for one bit). Decode the declared polynomial with original output-partition metadata, then round W to the execution dtype for A16 GEMM. |

Cubic supports FP32-scale/FP16-coefficient metadata, compact signed INT8 scales
and signed INT4 a/b with partition globals, and E5M9-scale/curve2 FP16 metadata.
Compact a/b pass through their declared FP16 coefficient boundary. Q/K/V
partitions never collapse independent globals into one approximate scale.
The local Cubic repository actually uses 4/6/8-bit matrices and both compact
and curve2 metadata.

The portable eager FP8/FP4 implementation may represent exact small codes in
FP16 inside an MMA tile. It preserves the required activation quantization and
format scales; it does not substitute A16 activations, requantize stored weights,
or create a full FP16/FP32 weight matrix. Native instructions and tile choices
are implementation details; additional device families require real-device
qualification. No architecture-specific branch or retained AOT artifact is added.

Embedding lookup consumes integer IDs and decodes selected rows to the requested
output dtype; it does not quantize IDs as floating activations. The FP8 and Cubic
repositories contain quantized embedding, lm_head and MTP weights. All are bound
by their own checkpoint declarations. The AWQ repository keeps these members
plain, consistent with AWQ's default unquantized lm_head convention. A consumer
whose physical payload cannot close its declared scheme rejects admission.

## Admission and configuration

Conflicting root/text quantization declarations, conflicting best-matching
rules, missing or unowned physical tensors, malformed shapes/dtypes, unsupported
transforms and unsupported activation policies reject admission. Nested Cubic
text rules match the text model namespace, including model-owned merged aliases.
No filename or model identity infers a quantization scheme.

The physical safetensors inventory remains authoritative. Some exporters retain
a stale index `metadata.total_size`; quantized admission reports that summary
and uses the validated physical tensor byte count. This narrowly scoped choice
does not relax shard/name/shape/offset ownership and does not populate the strict
BF16 inventory cache. Shard identity is rechecked before loading its payload.

Use the existing version-1 `--operator-config` parser and precedence. The new
families are `awq_linear`, `fp8_linear`, `nvfp4_linear`, and `cubic_linear`.
They share `PackedLinearParameters` because the units agree:
`fixed_block_m/n/k`, `fixed_warps`, `fixed_stages`, `embedding_block_k`.
NVFP4 fixes K=16 at the format accumulation boundary and rejects K overrides.
Plain native linear uses the existing `linear` family. The existing `moe` family
continues to own the BF16 fused-MoE implementation. Changing one quantized family
does not change another family or the ordinary BF16 baseline parameters.

```json
{
  "version": 1,
  "profiles": [
    {
      "operator": "cubic_linear",
      "shape": {"n": 5120, "k": 5120},
      "parameters": {"fixed_block_n": 64}
    }
  ],
  "overrides": {"awq_linear": {"fixed_block_k": 64}}
}
```

This is an example of configuration syntax, not an accepted performance preset.
The configuration is immutable and its digest is logged. Profiles can select
exact device/model identities and N/K geometry through the existing mechanism;
the default has no device/model/concurrency table. Public C and logical M are
not reduction selectors. Bits, packing, activation quantization, scale dtypes,
rounding, expert ownership and global-scale interpretation are not tuning knobs.
Changing execution parameters requires the corresponding numerical qualification.

An explicit functional launch uses, for example:

```sh
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src ./venv/bin/python -m cubic_serve.cli serve \
  /pubdata/llm_models/Qwen/Qwen3.8-27B-Cubic-4Bit \
  --execution-mode eager --device cuda:0 --dtype fp16 \
  --mamba-ssm-cache-dtype bfloat16 --prefill-chunk-tokens 16 \
  --host 127.0.0.1 --port 30015
```

The eager worker uses request-local state and the existing Chat Completions,
sampling, output codec, cancellation and signal lifecycle. It serializes model
work and makes no continuous-batching claim. MTP serving is rejected, and its
unused resident weights are omitted from this public path. Complete checkpoint
admission still covers those physical members. Vision execution is rejected.

## Qualification boundaries

`tests/ops/test_cuda_quantized.py` checks FP16/BF16 native formats against
independent arithmetic, logical M1/M8/M33, masked tile tails, row permutations
and lookup values. Native execution is checked with the reference weight decoder
blocked. Scalar checkpoint probes independently interpret actual serialized
codes and metadata.

The model audit freezes prefix state, completes the observed forward, then runs
an unrestricted independent reference forward and causal producing-boundary
checks. Every linear output is checked over its full K using an operand-derived
FP64 anchor and the named FP32-dot/format-scale/output-RNE enclosure. Intermediate
inputs, non-linear boundaries and final output/state consumption must agree in
the causal replay. Raw independent logit differences are reported; a global
absolute tolerance does not stand in for this proof. Repeated expert occurrences
may reuse a full-K proof only with identical weights, operands and rounded
outputs. Separate target/MTP cross-M and physical-row-permutation checks remain
required. These audits do not issue CUDA Graph or speculative algorithm seals.

Current evidence is stored in `/workspace/artifacts/quantization-adaptation/`.
See the final qualification manifest there for the exact model, dtype, rows,
retained sources and status of each run. A missing, interrupted or OOM run is
unqualified, not a floating-point pass. CUDA Graph padding poison, distributed
ownership/collectives, speculative proposal/target/K/RNG repair, canonical
1024-token SSE concurrency curves and long-generation decay remain separate
work before performance serving can be enabled.

## Pinned upstream basis

These are source facts, not numerical or throughput oracles. Working-file hashes
are recorded in the external inventory and source manifest.

- llama.cpp `458681e1d5d4a29a1463c4732e03226cf384b997`:
  `src/llama-quant.cpp`, `ggml/src/ggml-quants.c`, and
  `ggml/src/ggml-cuda/mmq.cuh` establish per-tensor formats, GGUF-specific FP4
  storage, and packed tiled execution. No Cubic adapter was found.
- Local vLLM `89af4f1ff792199b7c9260dd4a86feed445fd3d6`:
  `vllm/model_executor/layers/quantization/{base_config.py,auto_awq.py,awq_triton.py,cubic.py,cubic_kernels.py}`
  and `compressed_tensors/{compressed_tensors_embedding.py,schemes/compressed_tensors_w8a8_fp8.py,schemes/compressed_tensors_w4a4_nvfp4.py}`
  define consumer matching, AWQ ordering/A16 rounding, Cubic codes/coefficients,
  FP8 embedding and dynamic-token activation conventions, and inverse NVFP4
  global scales. The local working tree is not treated as a clean commit alone.
- SGLang `ba433bb4628f44d5031e910ad32194ad6c145ba8`:
  `python/sglang/srt/layers/quantization/base_config.py`, `awq/awq.py`, and
  `compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py` separate linear
  and fused-MoE consumers and bind format-specific storage. Global scale handling
  differs from the newer vLLM path, so the actual checkpoint convention must be
  established rather than copied blindly. No Cubic adapter was found.

Shared invariants are explicit consumer identity, authoritative serialized
metadata, exact group/partition ownership, format-correct activation arithmetic
and accumulation/output boundaries. Layout differences and unavailable consumers
are explicit admission boundaries. Cubic Serve implements those common contracts
with its existing model graph, configuration and token transport.
