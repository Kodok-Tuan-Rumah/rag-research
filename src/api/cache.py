from __future__ import annotations

import os
import json
import logging
from typing import Any, Optional

logger = logging.getLogger("rag.cache")

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class RedisCache:
    def __init__(self):
        self.client = None
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        enabled = os.getenv("REDIS_ENABLED", "true").lower() == "true"

        if REDIS_AVAILABLE and enabled:
            try:
                self.client = redis.Redis.from_url(redis_url, socket_timeout=1.0, socket_connect_timeout=1.0)
                self.client.ping()
                logger.info("[CACHE] Connected to Redis successfully at %s", redis_url)
            except Exception as e:
                logger.warning("[CACHE] Redis connection failed (%s). Caching disabled (fallback mode).", e)
                self.client = None
        else:
            logger.info("[CACHE] Redis caching disabled or package unavailable.")

    def get(self, key: str) -> Optional[Any]:
        if not self.client:
            return None
        try:
            val = self.client.get(key)
            if val:
                return json.loads(val.decode("utf-8"))
        except Exception as e:
            logger.debug("[CACHE] Redis GET error for key %s: %s", key, e)
        return None

    def set(self, key: str, value: Any, ttl_seconds: int = 300) -> None:
        if not self.client:
            return
        try:
            self.client.setex(key, ttl_seconds, json.dumps(value, default=str))
        except Exception as e:
            logger.debug("[CACHE] Redis SET error for key %s: %s", key, e)


# Global cache instance
cache = RedisCache()
