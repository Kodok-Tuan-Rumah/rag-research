from src.api import cache as cache_module


class FakeRedis:
    def __init__(self):
        self.store = {}

    def ping(self):
        return True

    def get(self, key):
        return self.store.get(key)

    def setex(self, key, ttl, value):
        self.store[key] = value.encode("utf-8")

    def delete(self, key):
        self.store.pop(key, None)


def test_redis_cache_round_trip(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setenv("REDIS_ENABLED", "true")
    monkeypatch.setattr(cache_module.redis.Redis, "from_url", lambda *a, **k: fake)

    c = cache_module.RedisCache()
    c.set("k", {"answer": "ok"}, ttl_seconds=60)

    assert c.get("k") == {"answer": "ok"}
    c.delete("k")
    assert c.get("k") is None


def test_redis_connection_failure_disables_cache(monkeypatch):
    monkeypatch.setenv("REDIS_ENABLED", "true")

    def fail(*args, **kwargs):
        raise cache_module.redis.exceptions.ConnectionError("redis down")

    monkeypatch.setattr(cache_module.redis.Redis, "from_url", fail)
    c = cache_module.RedisCache()

    assert c.client is None
    assert c.get("k") is None
    c.set("k", {"answer": "ok"})  # must not raise


def test_redis_invalid_json_is_a_miss(monkeypatch):
    fake = FakeRedis()
    fake.store["bad"] = b"not-json"
    monkeypatch.setenv("REDIS_ENABLED", "true")
    monkeypatch.setattr(cache_module.redis.Redis, "from_url", lambda *a, **k: fake)

    assert cache_module.RedisCache().get("bad") is None
