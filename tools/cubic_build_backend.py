"""PEP517 hooks for source installs and Python-ABI-independent CUDA wheels."""
from contextlib import contextmanager
import os
from pathlib import Path

from setuptools import build_meta as _backend
from setuptools.config.pyprojecttoml import read_configuration


def _cuda(settings):
    value = (settings or {}).get("cuda-bundle", "false")
    if value not in ("true", "false"):
        raise ValueError("cuda-bundle must be true or false")
    return value == "true"


@contextmanager
def _options(settings):
    old = os.environ.get("CUBIC_BUILD_CUDA")
    os.environ["CUBIC_BUILD_CUDA"] = "1" if _cuda(settings) else "0"
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("CUBIC_BUILD_CUDA", None)
        else:
            os.environ["CUBIC_BUILD_CUDA"] = old


def get_requires_for_build_wheel(config_settings=None):
    with _options(config_settings):
        requires = _backend.get_requires_for_build_wheel(config_settings)
    if _cuda(config_settings):
        # Reuse format-owned source generators, including their transitive
        # runtime imports. Keep one dependency contract for isolated publishers
        # instead of assuming an already-populated development environment.
        project = read_configuration(Path(__file__).parents[1]/"pyproject.toml",
                                     expand=False)["project"]
        requires = [*requires, *project["dependencies"]]
    return list(dict.fromkeys(requires))


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    with _options(config_settings):
        return _backend.prepare_metadata_for_build_wheel(metadata_directory, config_settings)


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    with _options(config_settings):
        return _backend.build_wheel(wheel_directory, config_settings, metadata_directory)


def get_requires_for_build_editable(config_settings=None):
    if _cuda(config_settings):
        raise ValueError("Editable installs use source compilation, not a bundled wheel")
    return _backend.get_requires_for_build_editable(config_settings)


def prepare_metadata_for_build_editable(metadata_directory, config_settings=None):
    with _options(config_settings):
        return _backend.prepare_metadata_for_build_editable(metadata_directory, config_settings)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    with _options(config_settings):
        return _backend.build_editable(wheel_directory, config_settings, metadata_directory)


get_requires_for_build_sdist = _backend.get_requires_for_build_sdist
build_sdist = _backend.build_sdist
