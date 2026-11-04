"""Read-only installed CUDA artifacts, independent of all disposable caches.

This module is also imported by the wheel builder; it must not import Torch,
create a CUDA context, inspect a model or select performance parameters.
"""
from __future__ import annotations

from functools import cache
import hashlib
import json
from pathlib import Path
from collections.abc import Mapping

BUNDLE_SCHEMA = 'cubic.native-bundle.v1'


def source_key(namespace: str, sources: Mapping[str, bytes], *, minimum_isa: int) -> str:
    if not namespace.isidentifier() or minimum_isa not in (80, 90):
        raise ValueError('Invalid native source namespace or instruction contract')
    if 'entry.cu' not in sources:
        raise ValueError('Native sources omit their entry unit')
    value = hashlib.sha256()
    value.update(json.dumps([BUNDLE_SCHEMA, namespace, minimum_isa]).encode())
    for name, data in sorted(sources.items()):
        path = Path(name)
        if (path.is_absolute() or '..' in path.parts or not name
                or type(data) is not bytes or not data):
            raise ValueError('Invalid native source file')
        # Explicit lengths avoid ambiguous file-name/content concatenations.
        for part in (name.encode(), data):
            value.update(len(part).to_bytes(8, 'little')); value.update(part)
    return value.hexdigest()


def _read(root: Path, name: str, digest: str) -> bytes:
    path = (root/name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise RuntimeError('Installed native artifact path is invalid')
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise RuntimeError('Installed native artifact digest differs from its manifest')
    return data


@cache
def _manifest(root: Path):
    path = root/'manifest.json'
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    if (data.get('schema') != BUNDLE_SCHEMA or not isinstance(data.get('units'), dict)
            or not isinstance(data.get('architectures'), list)
            or any(type(x) is not int or x < 80 for x in data['architectures'])):
        raise RuntimeError('Installed native bundle manifest is invalid')
    if 'allow_source_extensions' in data:
        keys = data.get('catalog_keys')
        if (type(data['allow_source_extensions']) is not bool or not isinstance(keys, list)
                or any(not isinstance(key, str) for key in keys)
                or len(keys) != len(set(keys)) or set(keys) != set(data['units'])):
            raise RuntimeError('Installed native bundle differs from its declared recipe catalog')
    return data


def installed_image(namespace: str, sources: Mapping[str, bytes], *, minimum_isa: int,
                    sm: int, root: Path | None = None):
    """Return a verified image, or an explicitly declared source extension.

    Every advertised recipe/image fails closed on omission or corruption. A
    source-extension bundle may additionally compile an unlisted exact source
    before capture, with the same operator and qualification contracts.
    """
    root = Path(__file__).resolve().parents[2]/'_native' if root is None else root
    manifest = _manifest(root)
    if manifest is None or sm not in manifest['architectures']:
        return None
    key = source_key(namespace, sources, minimum_isa=minimum_isa)
    unit = manifest['units'].get(key)
    if unit is None:
        if manifest.get('allow_source_extensions', False):
            return None
        raise RuntimeError('Installed CUDA bundle omits a required operator recipe')
    if unit.get('namespace') != namespace or unit.get('minimum_isa') != minimum_isa:
        raise RuntimeError('Installed CUDA operator identity is inconsistent')
    image = unit.get('images', {}).get(str(sm))
    if image is None:
        raise RuntimeError('Installed CUDA bundle omits an advertised device image')
    ptx = _read(root, unit['ptx'], unit['ptx_sha256'])
    cubin = _read(root, image['path'], image['sha256'])
    return ptx, cubin, unit['entry'], unit['compiler_metadata']
