"""The retriever must not carry per-request state between searches.

`src/api/main.py` builds one `MasterHybridRetriever` at startup and dispatches
every request through it with `run_in_threadpool`, so FastAPI serves concurrent
requests on the same object. `_run_search` used to resolve the expansion flags
into instance attributes:

    self.use_multi_query = plan.allow_multi_query if use_multi_query is None ...
    self.use_hyde = plan.allow_hyde if use_hyde is None ...

and `_expand_and_hyde_parallel` read them back. Two concurrent requests could
therefore overwrite each other's flags between the write and the read.

Impact today is zero: §9 forbids re-enabling multi-query and HyDE, so both
resolve to False on every path. The defect is latent, not absent — and §13.3
graceful degradation needs per-request control, which mutable instance state
structurally cannot provide.
"""

from __future__ import annotations

import inspect
import pathlib
import threading

import pytest

from src.retrieval.hybrid_retriever import MasterHybridRetriever

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _source() -> str:
    return (REPO_ROOT / "src" / "retrieval" / "hybrid_retriever.py").read_text(
        encoding="utf-8"
    )


def test_retriever_never_assigns_expansion_flags():
    """The assignment is the race. Reading them would be a lesser problem."""
    offenders = [
        line.strip()
        for line in _source().splitlines()
        if line.strip().startswith(("self.use_multi_query", "self.use_hyde"))
        and not line.strip().startswith("#")
    ]
    assert not offenders, (
        "per-request expansion flags are written to instance state:\n  "
        + "\n  ".join(offenders)
    )


def test_expansion_flags_are_parameters_not_attributes():
    """Constructor arguments suggested the retriever was globally configurable."""
    init = inspect.signature(MasterHybridRetriever.__init__)
    for name in ("use_multi_query", "use_hyde"):
        assert name not in init.parameters, (
            f"{name} is a constructor argument but _run_search always overrides "
            f"it from the per-query plan, so the argument has no effect"
        )


def test_expansion_helper_takes_the_flags_as_arguments():
    sig = inspect.signature(MasterHybridRetriever._expand_and_hyde_parallel)
    params = list(sig.parameters)
    assert "use_multi_query" in params, (
        "_expand_and_hyde_parallel must receive the flags; reading them off self "
        "is what made the race possible"
    )
    assert "use_hyde" in params


def test_concurrent_searches_do_not_interfere(monkeypatch):
    """Two threads, opposite flags: each must get its own answer."""
    retriever = MasterHybridRetriever.__new__(MasterHybridRetriever)

    class _Plan:
        def __init__(self, qt):
            self.query_type = qt
            self.allow_references = False
            self.max_ref_ratio = 0.0

    def fake_expand(query, use_multi_query=False, use_hyde=False):
        # Yield so the two threads are guaranteed to interleave.
        threading.Event().wait(0.01)
        return [query] if not (use_multi_query or use_hyde) else [query, f"{query}|{use_multi_query}{use_hyde}"], None

    monkeypatch.setattr(retriever, "_expand_and_hyde_parallel", fake_expand)
    monkeypatch.setattr(retriever, "_choose_candidate_k", lambda top_k: 10)
    monkeypatch.setattr(retriever, "_dense_search", lambda query, k: [])
    monkeypatch.setattr(retriever, "_bm25_search", lambda query, k: [])
    monkeypatch.setattr(retriever, "_rrf_fuse", lambda *a, **k: [])
    retriever.reranker = None

    results: dict[str, bool] = {}
    errors: list[BaseException] = []

    def worker(name, plan, **flags):
        try:
            retriever._policy = type("P", (), {"initial_plan": staticmethod(lambda q, p=plan: p)})()
            out = retriever.search_with_trace(name, top_k=5, **flags)
            results[name] = out["trace"]["bm25_query_count"] > 1
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    # Both threads share one retriever and alternate plans.
    threads = [
        threading.Thread(target=worker,
                         args=("a", _Plan("general")),
                         kwargs={"use_multi_query": True, "use_hyde": True}),
        threading.Thread(target=worker,
                         args=("b", _Plan("factual")),
                         kwargs={"use_multi_query": False, "use_hyde": False}),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert not errors, f"concurrent search raised: {errors}"
    assert results.get("a") is True, "the expanding request lost its flags"
    assert results.get("b") is False, "the plain request picked up another request's flags"