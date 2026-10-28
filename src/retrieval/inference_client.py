import time

import httpx
import numpy as np


class InferenceError(RuntimeError):
    pass


class InferenceClient:
    def __init__(self, base_url, timeout_ms=5000, connect_timeout_ms=500, max_retries=1, transport=None):
        self.base_url = base_url.rstrip("/")
        self.timeout_ms = timeout_ms
        self.connect_timeout_ms = connect_timeout_ms
        self.max_retries = max_retries
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(
                connect=connect_timeout_ms / 1000.0,
                read=timeout_ms / 1000.0,
                write=timeout_ms / 1000.0,
                pool=connect_timeout_ms / 1000.0,
            ),
            limits=httpx.Limits(max_connections=32, max_keepalive_connections=16),
            transport=transport,
        )
        self._failures = 0
        self._open_until = 0.0
        self._ready_cache = None
        self._ready_time = 0.0

    def _parse_error(self, response):
        code = "http_error"
        message = response.text
        try:
            data = response.json()
            err = data.get("error", {}) if isinstance(data, dict) else {}
            code = err.get("code", code)
            message = err.get("message", message)
        except Exception:
            pass
        return code, message

    def _request(self, method, path, **kwargs):
        if time.monotonic() < self._open_until:
            raise InferenceError("circuit open")
        last_exc = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.request(method, path, **kwargs)
                if response.status_code in (502, 503, 504):
                    code, message = self._parse_error(response)
                    raise InferenceError(f"{code}: {message}")
                response.raise_for_status()
                self._failures = 0
                self._open_until = 0.0
                return response.json()
            except (httpx.TransportError, InferenceError) as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    time.sleep(0.05 * (attempt + 1))
                    continue
                break
        self._failures += 1
        if self._failures >= 5:
            self._open_until = time.monotonic() + 10.0
        if isinstance(last_exc, InferenceError):
            raise last_exc
        raise InferenceError(str(last_exc))

    def embed(self, texts):
        texts = list(texts)
        data = self._request("POST", "/embed", json={"texts": texts})
        embeddings = data.get("embeddings", [])
        dim = data.get("dim", 0)
        arr = np.asarray(embeddings, dtype=np.float32)
        if arr.size == 0:
            arr = np.zeros((len(texts), dim), dtype=np.float32)
        if arr.shape != (len(texts), dim):
            raise InferenceError("invalid response shape")
        if not np.all(np.isfinite(arr)):
            raise InferenceError("non-finite embedding")
        return arr

    def rerank(self, query, documents):
        docs = [{"id": str(d[0]), "text": d[1]} for d in documents]
        data = self._request("POST", "/rerank", json={"query": query, "documents": docs})
        scores = data.get("scores", [])
        by_id = {str(s.get("id")): float(s.get("score", 0.0)) for s in scores}
        return [by_id.get(str(d[0]), 0.0) for d in documents]

    def ready(self):
        now = time.monotonic()
        if self._ready_cache is not None and (now - self._ready_time) < 5.0:
            return self._ready_cache
        try:
            response = self._client.get("/ready", timeout=2.0)
            value = response.status_code == 200
        except Exception:
            value = False
        self._ready_cache = value
        self._ready_time = now
        return value

    def close(self):
        self._client.close()


class RemoteEmbedder:
    def __init__(self, client, dim=384, batch_size=64):
        self.client = client
        self.dim = dim
        self.batch_size = batch_size

    def encode(self, sentences, batch_size=None, show_progress_bar=False, normalize_embeddings=False):
        single = isinstance(sentences, str)
        if single:
            sentences = [sentences]
        sentences = list(sentences)
        bs = batch_size or self.batch_size
        if bs <= 0:
            bs = self.batch_size
        chunks = [sentences[i:i + bs] for i in range(0, len(sentences), bs)]
        arrays = []
        for chunk in chunks:
            arr = self.client.embed(chunk)
            if arr.shape[1] != self.dim:
                raise InferenceError("wrong dimension")
            arrays.append(arr)
        if arrays:
            out = np.vstack(arrays).astype(np.float32)
        else:
            out = np.zeros((0, self.dim), dtype=np.float32)
        if single:
            return out[0] if out.shape[0] > 0 else np.zeros(self.dim, dtype=np.float32)
        return out


class RemoteReranker:
    def __init__(self, client, max_seq_length=256):
        self.client = client
        self.max_seq_length = max_seq_length

    def predict(self, pairs):
        if not pairs:
            return []
        pairs = list(pairs)
        groups = {}
        for idx, (query, _) in enumerate(pairs):
            groups.setdefault(query, []).append(idx)
        result = [0.0] * len(pairs)
        for query, indices in groups.items():
            docs = [(str(i), pairs[i][1]) for i in indices]
            scores = self.client.rerank(query, docs)
            for pos, idx in enumerate(indices):
                if pos < len(scores):
                    result[idx] = float(scores[pos])
        return result

    def backend_name(self):
        return "service"
