"""Process-scoped ownership of a completed runtime's static Python heap.

Freeze after cold staging is released and before serving starts. Reference
counting still frees acyclic objects, and transient request cycles remain in
the normal collector. The last owner restores visibility before teardown.
This is a process-wide runtime policy, matching the engine's worker lifecycle.
The interpreter may already have a permanent generation at process startup;
that does not mean the engine's subsequently allocated runtime is frozen.
"""

from __future__ import annotations

import gc
from contextlib import contextmanager
from threading import RLock

_lock = RLock()
_owners = 0


@contextmanager
def frozen_startup_heap():
    global _owners
    with _lock:
        if _owners == 0:
            gc.collect()
            gc.freeze()
        # A second runtime must not freeze another runtime's live requests.
        _owners += 1
    try:
        yield
    finally:
        with _lock:
            _owners -= 1
            if _owners == 0:
                gc.unfreeze()
