"""Runtime compatibility shims.

Import this module BEFORE importing ``qdrant_client`` or ``ragas`` anywhere.

Two independent breakages on Python 3.14 are neutralized here.

1. ``google._upb`` C extension
   Python 3.14 ships a broken ``google._upb`` C extension for protobuf <= 4.25.x.
   Importing ``qdrant_client`` therefore dies with
   ``TypeError: Metaclasses with custom tp_new are not supported``.
   Neutralizing the two ``google._upb*`` module entries makes protobuf fall back
   to its pure-Python implementation, which works on 3.14.

2. ``nest_asyncio``
   ``ragas/executor.py`` calls ``nest_asyncio.apply()`` at module level, so merely
   importing ragas patches asyncio for the whole process. That patch replaces
   ``asyncio.Task``/``asyncio.Future`` with the pure-Python classes and swaps the
   event loop methods. On 3.14 the result is that
   ``asyncio.tasks.current_task()`` returns None inside
   ``loop.run_until_complete()``, and every RAGAS metric raises:

       RuntimeError: Timeout should be used inside a task

   ``Metric.single_turn_ascore`` wraps its work in ``asyncio.wait_for`` (ragas
   0.2.15, ragas/metrics/base.py:534), which needs a current task. Reproduced
   with no API call:

       asyncio._nest_patched is True after `import ragas`
       loop.run_until_complete(asyncio.wait_for(coro, timeout=10))
         -> RuntimeError: Timeout should be used inside a task

   nest_asyncio 1.6.0 is the latest release, so there is no upstream fix, and
   there is no ``unpatch()``; ``_patch_asyncio`` overwrites
   ``asyncio.tasks._CTask`` and ``asyncio.futures._CFuture``, so the C
   implementations cannot be restored in-process.

   ``nest_asyncio`` only exists to make asyncio reentrant inside a notebook,
   where a loop is already running. ragas applies it unconditionally, and its
   only runtime use is guarded by ``is_event_loop_running()``
   (ragas/executor.py:200), which is False in a script. ``Executor.results()``
   then runs ``asyncio.run(self._process_jobs())``, which needs nothing extra.
   Scripts and pytest are not notebooks, so the patch is neutralized here.
"""

from __future__ import annotations

import sys
import types

_UPB_MODULES = ("google._upb", "google._upb._message")


def _neutralize_upb() -> None:
    for name in _UPB_MODULES:
        if name not in sys.modules:
            sys.modules[name] = None


def _neutralize_nest_asyncio() -> None:
    """Make ``nest_asyncio.apply()`` a no-op for this process."""
    try:
        import nest_asyncio  # noqa: F401
    except ImportError:
        # Not installed. ragas will still import, and its guarded apply() calls
        # will raise the same ImportError they always would.
        return

    if getattr(sys.modules.get("asyncio"), "_nest_patched", False):
        # Already applied before this module was imported. The C Task/Future
        # replacements cannot be undone, so nothing further can be done here.
        return

    stub = types.ModuleType("nest_asyncio")
    stub.apply = lambda *args, **kwargs: None
    stub.unpatch = lambda *args, **kwargs: None
    stub.__doc__ = (
        "No-op stand-in installed by src._compat. The real nest_asyncio "
        "breaks asyncio.wait_for on Python 3.14."
    )
    sys.modules["nest_asyncio"] = stub


_neutralize_upb()
_neutralize_nest_asyncio()