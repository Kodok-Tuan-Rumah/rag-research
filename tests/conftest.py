"""Test-wide isolation from the optional response cache.

The API creates its Redis client at import time. If a developer has Redis
running and REDIS_ENABLED=true, cached answers leak between tests and make the
suite order-dependent. Tests that exercise cache behavior inject a fake cache
explicitly; the real Redis backend is not needed here.
"""

import os

os.environ["REDIS_ENABLED"] = "false"
