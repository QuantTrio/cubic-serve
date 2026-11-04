# Marlin source provenance

Pinned source: vLLM commit `89af4f1ff792199b7c9260dd4a86feed445fd3d6`,
`csrc/libtorch_stable/quantization/marlin/`,
`csrc/libtorch_stable/moe/marlin_moe_wna16/`, and `csrc/core/scalar_type.hpp`.
Original copyright and Apache-2.0 notices are retained in the files.

Cubic Serve changes to `marlin_template.h`:
- grouped scales retain their original FP16 storage and reconstruct each weight
  with FP32 multiplication followed by one execution-dtype rounding;
- whole-K stripe ownership does not depend on logical row count; no atomic or
  cross-block split-K reduction is enabled by this binding.
`repack.cuh` retains only the upstream CUDA kernel, without the PyTorch host wrapper.
Compilation instantiates W4 with FP16/BF16 activations through the SM80 PTX baseline.
Geometry and group size are compile parameters, not model or device identities.
No compiled binary is distributed here. Runtime compilation and device qualification
are distinct; local evidence does not advertise untested hardware qualification.

Triangulation: SGLang `ba433bb4628f44d5031e910ad32194ad6c145ba8`
`python/sglang/kernels/jit/csrc/gemm/marlin_moe/marlin_template.h` and
`python/sglang/kernels/ops/quantization/awq_marlin_repack.py` use the same
repacked W4A16 family. llama.cpp `458681e1d5d4a29a1463c4732e03226cf384b997`
`ggml/src/ggml-cuda/mmq.cuh` uses resource-constrained tiles and packed reads,
but its Q8 activation contract is intentionally not imported as AWQ arithmetic.

The shared template also supports a raw FP32 accumulator epilogue and an ordered
K16 mode. FP8's exponent-bias correction and original FP32 token/channel scales
are applied by its format binding before the sole output cast. NVFP4 uses exact
FP16 carriers for FP4 values times original E4M3 scales, zero-initializes each
K16 MMA partial and adds it in group order; its original global-scale divisions
remain outside the GEMM. Neither path imports upstream scale recoding/clipping.

Additional inspected sources at the same pinned commits:
- vLLM `vllm/model_executor/layers/quantization/utils/marlin_utils_fp8.py`
  and `marlin_utils_fp4.py`: upstream W8A8 is rejected; weight-only scale casts
  and FP4 scale recoding are intentionally different contracts.
- SGLang `python/sglang/srt/layers/quantization/marlin_utils_fp8.py`
  and `compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4_moe.py`:
  format/backend-owned scale preparation, not a shared numerical oracle.
- llama.cpp `ggml/src/ggml-cuda/common.cuh` and `quantize.cu`:
  distinct FP4 quantizer and scale-selection rules, not imported here.

`gptq_repack.cuh` retains the integer K-packed repack kernel from the same
vLLM commit, `csrc/libtorch_stable/quantization/marlin/gptq_marlin_repack.cu`,
without its framework wrapper. Compared with SGLang's pinned
`python/sglang/kernels/jit/csrc/gemm/marlin/gptq_marlin_repack.cuh`, both
convert packed K-axis words through shared staging directly to MMA layout.
Cubic selects the source packing axis, not a model/quantization name; W4 and W8
codes are never reconstructed as floating weights. No activation-order permutation
is enabled. llama.cpp's packed `mmq.cuh` layout has a different ABI and is not
substituted. Preparation runs before capture, separate from graph execution.

Cubic's NVFP4 binding uses the vendor S0E5M3 scale carrier with an explicitly
lossless positive-E4M3 contract. Multiply original scales by 128 in FP16, encode
the exponent/mantissa bytes, and pair them with raw FP4 half carriers biased by
2**-14. Restore 128 in the FP32 epilogue before the original IG/WG divisions.
All 2,032 signed-FP4 / nonnegative-finite-E4M3 products, including zero and
subnormal scales, retain their original half result. The fixed K16 reduction
order remains unchanged. Negative/nonfinite scales are rejected as in the
independent format reference. Unlike the pinned vLLM scale preparation, Cubic
never zeros scales below 2 after rebiasing, and never chooses a data-dependent
rescaling factor. The constants belong to the carrier formats, not device,
model, topology or concurrency. Operator witnesses do not constitute full-model
qualification. Scalar/vector original-E4M3 tile decoders were rejected for
measured overhead; they are not retained as alternate execution paths.

The shared `marlin_template.h` selects the preserved architecture declaration.
`template/configuration.cuh` owns compile-time types and geometry;
`kernel.cuh` owns CTA-local storage and connects routing, operand staging,
metadata, MMA, reductions, output stores, and execution. Each stage is a complete
force-inlined device function with explicit reference operands; no header is
included inside a function body. Small local adapters retain upstream call order
and defaults without hiding the stage's dependencies. Fragment arrays are passed
by reference, never copied into an opaque runtime context. Every header can be
compiled independently, as checked by `tests/ops/test_marlin_headers.py`.
The native compiler hashes and copies every nested `.cuh` into its immutable
source receipt. A structural refactor still requires code-generation, resource,
numerical and latency checks; inlining alone does not prove performance identity.

The upstream pre-SM75 empty declaration lives in `template/legacy.cuh`; SM75's
arithmetic branches remain in `template/configuration.cuh` and `kernel.cuh`.
Both are preserved. Their
presence does not change the documented compilation baseline or assert new
hardware qualification. FP8, NVFP4, AWQ and Cubic still share this implementation.

Removed one Cubic Serve-only leftover: the extra NVFP4/FP16-scale template
admission from the retired half-scale carrier experiment. The current NVFP4
binding in `extensions/nvfp4/marlin.py::_kernel_source` instantiates the lossless
E4M3 carrier, and no in-tree caller instantiates the retired combination. The
original upstream FP4 scale-type admission and all architecture branches remain.
