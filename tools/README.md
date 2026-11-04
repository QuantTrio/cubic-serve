# Tools

Run tools from the repository root in the serving environment. For a source
checkout, use `PYTHONPATH=src` if the package is not installed.

| Location | Purpose |
| --- | --- |
| `build_release.py` | Build and validate a CUDA wheel. |
| `build_cuda.py`, `cubic_build_backend.py` | CUDA bundling and PEP 517 build hooks. |
| `check_install.py` | Validate wheel contents or an installed package. |
| `benchmarks/chat_stream.py` | Public streaming Chat Completions benchmark and device telemetry. |
| `cuda/check_graphs.py` | Conditional CUDA graph and collective diagnostics, without model weights. |
| `quantization/audit_checkpoint.py` | Quantization format and checkpoint correctness probes. |
| `qwen3_5/` | Qwen3.5-family model audits and graph diagnostics, including adapted Qwen3.8 checkpoints. |

The packaging entrypoints stay at the root to preserve the release command and
build-backend paths:

```bash
CUDA_HOME=/usr/local/cuda-13.0 ./venv/bin/python tools/build_release.py
```

## Qwen tools

| Tool | Purpose |
| --- | --- |
| `concurrent_audit.py` | Capture and independently audit concurrent target review frames. |
| `graph_audit.py` | Audit decode CUDA graphs against the eager reference. |
| `graph_bench.py` | Time raw decode graph replay. |
| `linear_witness.py` | Audit sampled linear projections with a CPU FP64 reference. |
| `mtp_graph_audit.py` | Audit the MTP layer's eager and captured execution. |
| `tp2_pp2_audit.py` | Audit distributed decode across TP/PP ranks. |
| `tp2_pp2_bench.py` | Time raw distributed graph replay. |
| `tp2_pp2_leaf_audit.py` | Audit distributed linear occurrences and reduction witnesses. |

Raw graph timings are diagnostics. Use `benchmarks/chat_stream.py` for serving
throughput. Each command's `--help` lists its inputs and options.
