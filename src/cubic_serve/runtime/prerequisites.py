"""Host requirements shared by CUDA startup and installed-wheel validation."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys
import sysconfig
import tempfile


class RuntimePrerequisiteError(RuntimeError):
    """An actionable installation error detected before model loading."""


def _python_include() -> Path:
    # Match Triton's Debian scheme handling and the active venv's interpreter.
    scheme = sysconfig.get_default_scheme()
    if scheme == 'posix_local':
        scheme = 'posix_prefix'
    return Path(sysconfig.get_paths(scheme=scheme)['include'])


def require_host_build_prerequisites() -> dict[str, str]:
    """Check files and executables only; no GPU, compilation or downloads."""
    include = _python_include()
    compiler = os.environ.get('CC')
    compiler = (shutil.which(compiler) if compiler is not None else
                shutil.which('gcc') or shutil.which('clang'))
    missing = []
    version = f'{sys.version_info.major}.{sys.version_info.minor}'
    if not (include/'Python.h').is_file():
        missing.append(f'Python {version} development headers (Python.h)')
    if compiler is None:
        missing.append('a C compiler')
    if missing:
        raise RuntimePrerequisiteError(
            'Missing '+ ' and '.join(missing)+'.\n\n'
            'On Debian/Ubuntu, run:\n'
            '  sudo apt-get update\n'
            f'  sudo apt-get install python{version}-dev build-essential\n\n'
            'Then retry your launch command.'
        )
    return {'python_include': str(include), 'c_compiler': compiler}


def check_triton_host_compilation() -> dict[str, str]:
    """Explicit installation check: cold compile/load through Triton's host ABI.

    Bypass compiler caches so preexisting helpers cannot hide missing headers.
    This runs only in installation validation, never normal serving startup.
    """
    result = require_host_build_prerequisites()
    from triton.runtime.build import _build, _load_module_from_path

    name = '_cubic_install_host_probe'
    source = f'''#include <Python.h>
static PyObject* version(PyObject* self, PyObject* args) {{
  return PyLong_FromLong(PY_MAJOR_VERSION*100+PY_MINOR_VERSION);
}}
static PyMethodDef methods[] = {{{{"version", version, METH_NOARGS, NULL}}, {{NULL}}}};
static struct PyModuleDef module = {{PyModuleDef_HEAD_INIT, "{name}", NULL, -1, methods}};
PyMODINIT_FUNC PyInit_{name}(void) {{return PyModule_Create(&module);}}
'''
    with tempfile.TemporaryDirectory(prefix='cubic-host-check-') as temporary:
        path = Path(temporary)/f'{name}.c'
        path.write_text(source)
        library = _build(name, str(path), temporary, [], [], [], [])
        module = _load_module_from_path(name, library)
        if module.version() != sys.version_info.major*100+sys.version_info.minor:
            raise RuntimePrerequisiteError('Python development headers do not match the active interpreter.')
    return {**result, 'triton_host_compilation': 'passed'}
