# Installation

Cubic Serve supports Linux x86-64, CPython 3.10–3.14 with the GIL enabled,
and NVIDIA CUDA devices. Install the **CUDA 13.0 toolkit**, including `nvcc`,
and a CUDA 13-compatible NVIDIA driver. CUDA runtime libraries installed by
PyTorch do not supply the native compiler. Set `CUDA_HOME` to the 13.0 toolkit;
an ambient newer toolkit does not replace this build baseline.

Install a C compiler and development headers matching the venv's Python.
Triton compiles Python host helpers at runtime even with precompiled CUDA kernels.
For Python 3.12 on Debian/Ubuntu:

```bash
sudo apt-get update
sudo apt-get install python3.12-dev build-essential
```

Use the corresponding `python3.10-dev` through `python3.14-dev` package for other
supported interpreters. Custom Python installations need their matching headers.

```bash
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
```

Use one of the three installation routes below. Installing from Git requires the `git` command.
No recursive submodule checkout or neighboring upstream repository is required.

## Install directly from Git

```bash
python -m pip install 'cubic-serve @ git+https://github.com/QuantTrio/cubic-serve.git'
# For a reproducible installation, append @ followed by a full commit SHA.
```

## Install a clone

```bash
git clone https://github.com/QuantTrio/cubic-serve.git
cd cubic-serve
python -m pip install .
```

For development use `python -m pip install -e '.[dev]'` instead. Normal and
editable installations expose the same `cubic` command. `PYTHONPATH=src` is
unnecessary after installation.

## Install a wheel

```bash
python -m pip install /path/to/cubic_serve-0.0.0+cu130-py3-none-linux_x86_64.whl
```

The **same Cubic wheel** is used on Python 3.10, 3.11, 3.12, 3.13 and 3.14.
Pip selects Python-specific wheels for dependencies such as PyTorch and Triton.
Python 3.15+ is not advertised until those dependencies support it and the
installation matrix passes. Free-threaded Python is not qualified.

The release wheel bundles precompiled CUDA kernels and PTX for the shared native
baseline across all ten SM targets. It also includes native CUDA sources,
Marlin/GGML headers and their license notices. CUDA code objects use the Driver
API and have no CPython extension ABI.

CUDA wheels carry a build-variant version suffix: `+cu130` identifies CUDA 13.0.
The builder derives it from the selected CUDA compiler; the wheel filename,
distribution metadata and installed `cubic --version` agree. The source version
remains independent of CUDA. A future CUDA 12.8 build would use `+cu128`, but the
currently qualified build pipeline still requires CUDA 13.0. CUDA-labelled wheels
are for direct downloads or a separate wheel index; PyPI does not accept local
version suffixes. Ordinary source distributions retain the base version.

The catalog covers format-owned baseline arithmetic, bit widths, activation and
scale types, dense/routed projections and speculative reduction variants.
Matrix dimensions and client concurrency are runtime operands. Adaptive or
explicitly overridden tiles outside the advertised catalog compile from the
same source owner before graph capture. Triton coordinates its own kernels.
Thus the toolkit remains required; the wheel does not promise that every possible
configuration is compilation-free. Missing or corrupt advertised images fail
closed. Source extensions retain all existing operator and qualification guards.
Installing a wheel does not create model qualification evidence or warm plans.

The installation format does not restrict SM80, SM86, SM89, SM90, SM100, SM103,
SM110 or SM120; SM87/SM88 are also part of the CUDA artifact baseline. Individual
operators retain their instruction/resource requirements and fail explicitly
when unavailable. Cross-assembly and an installable wheel are not real-device
model or throughput qualification; current serving evidence is on local SM120.

## Verify and launch

Run from any directory, including outside the clone:

```bash
cubic --help
cubic serve --help
cubic serve /path/to/model --execution-mode cuda --devices 0 --tp-size 1
```

Keep the model-specific options in your existing launch script. Replace
`PYTHONPATH=src ./venv/bin/python -m cubic_serve.cli` with `cubic` when using an
installed environment, or invoke that environment's `python -m cubic_serve.cli`.
Logging defaults to the console, with no automatic log file. See [logging](#logging)
for optional file output.

Use `--served-model-name` to give the model a public API name:

```bash
cubic serve /path/to/model --served-model-name my-model --devices 0 --tp-size 1
```

The name appears in `/v1/models` and in OpenAI/Anthropic responses. Multiple names
can be provided, for example `--served-model-name my-model my-model-v1`: requests
may use either name, while responses use the first. This changes the API name
only; checkpoint paths, internal model/weight identity and inference settings
keep their original values. Without the option, the checkpoint model ID is used.

`cubic serve --help` groups options into Server, Model and generation, Devices and
parallelism, Memory and cache, Scheduling, Multimodal input, and Advanced, audit
and debug. `cubic qualify --help` uses the same groups.

To require an API key, add `--api-key YOUR_KEY`. Clients send
`Authorization: Bearer YOUR_KEY`; Anthropic `/v1/messages` clients may instead
send `x-api-key: YOUR_KEY`. Missing or invalid credentials return HTTP 401 before
request admission. `/health` remains public; `/v1/health` requires the key.
Authentication is disabled when `--api-key` is omitted. The key is redacted from
startup logs. No API-key environment variable is used.

All writable caches use `CUBIC_SERVE_CACHE_DIR`, defaulting to
`${XDG_CACHE_HOME:-~/.cache}/cubic-serve`. Checkpoint metadata and operator plans
live directly under this root; native, Triton, CUDA-driver and Torch compiler
caches live under `compile/`. Set the variable before launching. Existing
per-compiler cache variables do not redirect these caches outside the root.
Read-only CUDA artifacts installed with the wheel remain in the Python package.
The installed package directory need not be writable. Cache compatibility and startup guards remain
mandatory. Engineering qualification remains an explicit `cubic qualify` step.

Install `cubic-serve[xgrammar]` (or `'.[xgrammar]'` in a clone) for structured
output support. Constraints are never silently ignored when an extra is absent.

## Logging

Console output is the default for every installation method, including containers
and services whose supervisor collects stdout/stderr. To enable file logging:

```bash
cubic serve /path/to/model --log-config '{"dir":"./logs","rotation":"100 MB","retention":"7 days"}'
```

Providing `dir` enables file logging to `cubic-serve.log` inside that directory.
The directory is created if needed. Relative paths resolve from the launch
directory; installation location is irrelevant. Logs are operational output,
separate from `CUBIC_SERVE_CACHE_DIR`.

| Key | Default | Meaning |
| --- | --- | --- |
| `dir` | Required for file logging | Output directory. |
| `rotation` | `"100 MB"` | Positive size or duration, e.g. `"1 day"`; `"daily"`, `"hourly"`, `"weekly"`, `"monthly"`, `"yearly"`; integer bytes; or `null` to disable rotation. |
| `retention` | `"7 days"` | Positive duration, integer rotated-file count (including zero), or `null` to keep all files. |
| `console` | `"startup"` | `"startup"`: console and file until HTTP startup succeeds, then file only. `"always"`: console and file throughout. |

The file includes frontend and inherited worker stdout/stderr, including errors
and shutdown output. One frontend writer uses Loguru for rotation and retention;
workers do not rotate files independently. Existing files append across restarts.
Retention cleanup runs when Loguru rotates a file (also on close when rotation
is disabled); durations do not schedule a background cleanup job. Separate server
instances should use separate log directories.

Invalid config or an unwritable output directory fails before model loading.
If file writing fails during operation, an error is printed to stderr and output
continues on the console; shutdown reports the logging failure. Without file
logging, console output continues throughout serving. No `log.out` is created.

## Build distributions

Ordinary Git/clone/source-wheel builds require neither a GPU nor Torch nor a CUDA
toolkit at build time. They include kernel sources for startup compilation.

```bash
python -m pip install build
python -m build
python tools/check_install.py --wheel dist/cubic_serve-0.0.0-py3-none-any.whl
```

For the CUDA release, use the checked publisher pipeline from a clone. Prepare
its environment once:

```bash
python -m pip install '.[dev]' 'setuptools>=75' 'packaging>=24' 'pip>=23'
```

Then run:

```bash
CUDA_HOME=/usr/local/cuda-13.0 python tools/build_release.py
```

The command checks the compiler and dependencies, runs packaging/loader contracts,
creates a source archive, builds the CUDA wheel **from that archive**, validates
every advertised PTX/cubin digest and packaged source, and tests installed imports,
CLI entry points and dependency consistency in a disposable environment. Only
then does it publish the wheel, source archive, `SHA256SUMS` and `release.json`
under `dist/`. It performs no downloads and does not modify the publisher's
installed packages. Publication copies are checked before replacing existing
files; `release.json` is the final completion marker.

To check the same wheel on additional installed Python environments, repeat
`--verify-python`. Each supplied environment must already contain the project
dependencies for that Python version; the pipeline reads them and creates its
own temporary installation:

```bash
python tools/build_release.py \
  --verify-python /path/to/py310/bin/python \
  --verify-python /path/to/py311/bin/python \
  --verify-python /path/to/py312/bin/python \
  --verify-python /path/to/py313/bin/python \
  --verify-python /path/to/py314/bin/python
```

The build never initializes a GPU or reads models, runtime compiler caches,
serving plans or qualification records. Its exact-source/compiler cache lives
under `CUBIC_SERVE_CACHE_DIR/compile/wheel/`.
Each completed recipe has an atomic, checksum-verified receipt. An interrupted
build retains valid completed recipes; incomplete or corrupt entries are rebuilt.
After fixing a failure, rerun the same command with the same cache. Failed stages
and logs remain under `--work-dir` (default `CUBIC_SERVE_CACHE_DIR/release/`).
The pipeline exits unsuccessfully and leaves existing published artifacts in
place when a build or validation stage fails.

`--jobs` controls compiler parallelism (default at most eight), further bounded
by CPU count and available host/container memory. `SOURCE_DATE_EPOCH` defaults
to the current Git commit time. Wall-clock measurements live in build logs and
receipts, outside the wheel. `payload_sha256` in `release.json` witnesses every
wheel member independently of ZIP timestamps; the ordinary SHA256 also witnesses
the exact wheel file.

For integration with another PEP517 frontend, the underlying native build hook
remains `python -m build --wheel --config-setting cuda-bundle=true`. The checked
pipeline is the release entry point.

`python -m build` also verifies a source archive can build its wheel. The package
CI installs one source wheel on every supported Python version outside the checkout.
Native release checks additionally verify every advertised image and install the
same CUDA-bearing wheel across the Python matrix. These checks do not issue
numerical model or GPU-family qualification receipts.

For release, update the version in `pyproject.toml` and
`src/cubic_serve/__init__.py` together, run the installation matrix, retain the
wheel checksum, and choose a project license before public distribution.
Bundled third-party license notices remain in their original directories.
