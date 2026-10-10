"""Runtime compatibility shims.

Python 3.14 ships a broken ``google._upb`` C extension for protobuf <= 4.25.x.
Importing ``qdrant_client`` therefore dies with
``TypeError: Metaclasses with custom tp_new are not supported``.
Neutralizing the two ``google._upb*`` module entries makes protobuf fall back
to its pure-Python implementation, which works on 3.14.

Import this module BEFORE importing ``qdrant_client`` anywhere.
"""

from __future__ import annotations

import sys

_UPB_MODULES = ("google._upb", "google._upb._message")


def apply() -> None:
    for name in _UPB_MODULES:
        if name not in sys.modules:
            sys.modules[name] = None  


apply()
