# Cubic Serve

**Built with AI, with human direction and review.**

Cubic Serve focuses on **Cubic, GGUF, and custom quantization formats** for
**high concurrency on a single node**. It is an experimental engine, not a
complete professional serving stack.

## Features

- Cubic quantization and GGUF formats, with partial support depending on the checkpoint and encoding.
- Single- and multi-GPU serving with CUDA Graphs, shared KV/state memory, and prefix caching.
- INT4/INT6/INT8 K and V caches, with independent dtype selection.
- Dynamic MTP for supported checkpoints.
- Streaming OpenAI/Anthropic-compatible APIs, tool calling, and structured outputs.

Support varies by model, format, and hardware.

## Features to be added

Under consideration:

- KV cache offload to system RAM.
- Strata mode.
- Support for more popular models.
- Additional draft mechanisms, such as dspark and dflash2.

## Make it your own

Customize it for your models, hardware, and workloads. **Floating-point
guardrails, independent references, and state audits** help catch regressions
as AI iterates on the implementation.

Validate your configuration after making changes.

## Installation

The current installation targets are **Linux x86-64, Python 3.10–3.14, and
CUDA 13.0**. See the [installation guide](docs/installation.md) for prerequisites,
compatibility limits, and the reproducible wheel build process.

Install a CUDA wheel:

```bash
python -m pip install ./cubic_serve-0.0.0+cu130-py3-none-linux_x86_64.whl
```

Or install from a clone:

```bash
git clone https://github.com/QuantTrio/cubic-serve.git
cd cubic-serve
python -m pip install .
```

Or install directly from Git:

```bash
python -m pip install 'git+https://github.com/QuantTrio/cubic-serve.git'
```

For development, install a clone with `python -m pip install -e '.[dev]'`.
Structured generation uses the optional `xgrammar` extra; see the installation
guide for setup.

## Launch

```bash
export CUBIC_SERVE_CACHE_DIR=./cache
cubic serve /path/to/model --devices 0 --tp-size 1
```

Choose compute mode, cache dtypes, parallelism, and speculative decoding settings
for your checkpoint and hardware. Inspect the available options with:

```bash
cubic serve --help
```

`CUBIC_SERVE_CACHE_DIR` is the single root for writable caches, including plans,
calibration results, and compiled/JIT artifacts. Precompiled artifacts shipped
inside the installed Python package remain there.

Logs go to the console by default; no log file is created. To enable file logging:

```bash
--log-config '{"dir":"./logs","rotation":"100 MB","retention":"7 days"}'
```

Startup output appears on the console and in `logs/cubic-serve.log`. After HTTP
startup succeeds, subsequent output goes to the file. See
[logging configuration](docs/installation.md#logging) for the available rules.

## Status

This is a project for users who want a focused, customizable quantized inference
engine and are willing to validate their setup. Hardware and model coverage are
incomplete; a successful run on one configuration does not qualify another.

## Credits

[CubicQuant: Parametric Non-Uniform Codebooks for High-Throughput LLM Inference with 1–8-Bit Weights](https://arxiv.org/abs/2608.06763)

Thanks to these projects and their contributors for the open-source work and
inference engineering ideas this project learns from:

- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [vLLM](https://github.com/vllm-project/vllm)
- [SGLang](https://github.com/sgl-project/sglang)
- [NInfer](https://github.com/Neroued/ninfer)

Included third-party components retain their upstream license notices.
