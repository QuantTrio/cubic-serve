"""Optional CUDA bundle commands; ordinary source installs remain lightweight."""
import os
from pathlib import Path

from setuptools import setup

from build_cuda import BuildPy, distribution_version

commands = {"build_py": BuildPy}
if os.environ.get("CUBIC_BUILD_CUDA") == "1":
    from build_cuda import BinaryWheel

    commands["bdist_wheel"] = BinaryWheel

setup(version=distribution_version(Path(__file__).parent / 'src/cubic_serve',
                                   bundled=os.environ.get('CUBIC_BUILD_CUDA') == '1'),
      cmdclass=commands)
