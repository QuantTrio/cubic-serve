import os
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize("case", ["nested", "transient", "disabled", "exception"])
def test_static_heap_lifetime_in_an_independent_process(case):
    # Never freeze pytest's process or rely on mocked collection semantics.
    code = """
        import gc, weakref
        from contextlib import ExitStack
        from cubic_serve.runtime.heap_lifetime import frozen_startup_heap
        class Cycle:
            def __init__(self): self.link = self
        gc.unfreeze()  # Isolated subprocess: remove interpreter startup freezing.
        before = gc.isenabled()
        if CASE == 'nested':
            original = Cycle(); witness = weakref.ref(original)
            first = ExitStack(); first.enter_context(frozen_startup_heap())
            second = ExitStack(); second.enter_context(frozen_startup_heap())
            del original; gc.collect(); assert witness() is not None
            first.close(); assert gc.get_freeze_count() > 0
            second.close(); assert gc.get_freeze_count() == 0
            gc.collect(); assert witness() is None
        elif CASE == 'transient':
            with frozen_startup_heap():
                current = Cycle(); witness = weakref.ref(current)
                del current; gc.collect(); assert witness() is None
            assert gc.get_freeze_count() == 0
        elif CASE == 'disabled':
            gc.disable(); before = False
            with frozen_startup_heap():
                current = Cycle(); witness = weakref.ref(current)
                del current; gc.collect(); assert witness() is None
            assert gc.get_freeze_count() == 0
        else:
            try:
                with frozen_startup_heap(): raise RuntimeError('test')
            except RuntimeError: pass
            assert gc.get_freeze_count() == 0
        assert gc.isenabled() == before
    """
    subprocess.run(
        [sys.executable, "-c", "CASE = " + repr(case) + "\n" + textwrap.dedent(code)],
        env=os.environ.copy(),
        check=True,
        timeout=30,
    )
