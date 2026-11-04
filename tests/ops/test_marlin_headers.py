"""Native stage headers must parse without an enclosing kernel's local scope."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

from cubic_serve.executor.runner import configure_reference_math
from cubic_serve.quantization.common.marlin import compiler


@pytest.mark.skipif(shutil.which("c++") is None, reason="C++ compiler required")
def test_marlin_accumulator_initializer_preserves_adjacent_storage(tmp_path):
    root = Path(compiler.__file__).with_name("native")
    source = (root / "template/kernel.cuh").read_text()
    initializer = "auto zero_accums = [&]() {" + source.split(
        "auto zero_accums = [&]() {", 1
    )[1].split("};", 1)[0] + "};"
    # Compile the actual initializer with both fragment layouts and independent
    # adjacent sentinels. This is a storage proof, not a CUDA numerical oracle.
    unit = r"""
#include <algorithm>
#include <iterator>
struct Fragment { float values[4]; };
template<int M, int Columns> bool check() {
  constexpr int thread_m_blocks = M;
  struct Storage {
    Fragment frag_c[M][Columns][2];
    float protected_values[M * 4 * 2 * 4];
  } storage;
  auto& frag_c = storage.frag_c;
  for (auto& row : frag_c) for (auto& column : row)
    for (auto& fragment : column) for (float& value : fragment.values) value = 1;
  std::fill(std::begin(storage.protected_values), std::end(storage.protected_values), 1);
INITIALIZER
  zero_accums();
  for (auto& row : frag_c) for (auto& column : row)
    for (auto& fragment : column) for (float value : fragment.values)
      if (value != 0) return false;
  return std::all_of(std::begin(storage.protected_values), std::end(storage.protected_values),
                     [](float value) { return value == 1; });
}
int main() {
  return !(check<1,2>() && check<4,2>() && check<8,2>() &&
           check<1,4>() && check<4,4>() && check<8,4>());
}
""".replace("INITIALIZER", initializer)
    binary = tmp_path / "accumulator_bounds"
    subprocess.run(
        [shutil.which("c++"), "-x", "c++", "-std=c++17", "-O0",
         "-Wno-unknown-pragmas", "-o", str(binary), "-"],
        input=unit, text=True, capture_output=True, check=True,
    )
    subprocess.run([str(binary)], check=True)


@pytest.mark.skipif(
    os.environ.get("CUBIC_TEST_QUANT_CUDA") != "1", reason="explicit CUDA audit required"
)
def test_marlin_headers_are_self_contained():
    configure_reference_math()
    # Native linking requires an already established caller-owned CUDA context.
    context_owner = torch.empty(1, dtype=torch.int32, device="cuda:0")
    root = Path(compiler.__file__).with_name("native")
    headers = [root / "marlin_template.h", *sorted((root / "template").glob("*.cuh"))]
    for header in headers:
        source = (
            f'#include "{header.relative_to(root).as_posix()}"\n'
            '__global__ void header_probe(int* output) { output[0] = 0; }\n'
        )
        _, entry = compiler.compile_module(source, context_owner.device)
        assert "header_probe" in entry, header
