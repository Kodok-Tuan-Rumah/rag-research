from __future__ import annotations

import sys
import types

_UPB_MODULES = ("google._upb", "google._upb._message")


def _neutralize_upb() -> None:
    for name in _UPB_MODULES:
        if name not in sys.modules:
            sys.modules[name] = None


def _neutralize_nest_asyncio() -> None:
    try:
        import nest_asyncio  # noqa: F401
    except ImportError:
        return

    if getattr(sys.modules.get("asyncio"), "_nest_patched", False):
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
