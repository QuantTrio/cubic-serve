"""Standard-library identities on supported Python versions."""
import hashlib
import sys
from collections.abc import Callable
from typing import Any, BinaryIO

if sys.version_info >= (3, 11):
    from enum import StrEnum as StrEnum
else:
    from backports.strenum import StrEnum as StrEnum

try:
    from hashlib import file_digest as file_digest
except ImportError:
    def file_digest(fileobj: BinaryIO, digest: str | Callable[[], Any]) -> Any:
        """Hash binary input with bounded scratch on Python 3.10."""
        result = hashlib.new(digest) if isinstance(digest, str) else digest()
        if hasattr(fileobj, "getbuffer"):
            result.update(fileobj.getbuffer())
            return result
        if not hasattr(fileobj, "readinto") or not fileobj.readable():
            raise ValueError("file must be readable binary IO")
        buffer = bytearray(256 * 1024)
        view = memoryview(buffer)
        while True:
            count = fileobj.readinto(buffer)
            if count is None:
                raise BlockingIOError("file read blocked")
            if count == 0:
                break
            result.update(view[:count])
        return result
