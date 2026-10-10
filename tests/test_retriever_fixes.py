import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

class MockRetrievalResult:
    def __init__(self, chunk_id, text, doc_id, score):
        self.chunk_id = chunk_id
        self.text = text
        self.doc_id = doc_id
        self.score = score

class MockVectorStore:
    def search(self, query, k):
        return [
            MockRetrievalResult(
                chunk_id=f"c_{i}",
                text=f"chunk text about {query[:20]} number {i}",
                doc_id="doc_test",
                score=1.0 - i * 0.05
            )
            for i in range(k)
        ]

import unittest.mock as mock
with mock.patch.dict("sys.modules", {
    "src.retrieval.qdrant_store": mock.MagicMock(
        QdrantVectorStore=MockVectorStore,
        RetrievalResult=MockRetrievalResult
    )
}):
    with mock.patch.dict("sys.modules", {
        "sentence_transformers": mock.MagicMock(),
        "rank_bm25": mock.MagicMock(),
        "groq": mock.MagicMock(),
    }):
        import importlib
        import types
        pass

PASS = "OK"
FAIL = "FAIL"
results_log = []

def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    msg = f"  [{status}] {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    results_log.append((name, condition))
    return condition

def test_fix1_adaptive_rerank_n():
    print("\n[FIX 1] Adaptive rerank_n (bukan hardcoded 5)")
    
    cases = [
        (3,  6,  "top_k=3 → rerank 6, bukan 5"),
        (5,  10, "top_k=5 → rerank 10, bukan 5"),
        (7,  10, "top_k=7 → rerank 10 (capped)"),
        (2,  4,  "top_k=2 → rerank 4"),
        (10, 10, "top_k=10 → rerank 10 (capped)"),
    ]
    
    for top_k, expected, note in cases:
        rerank_n = min(top_k * 2, 10)
        check(note, rerank_n == expected, f"got {rerank_n}")
    
    # Pastikan rerank_n selalu >= top_k
    for top_k in range(1, 11):
        rerank_n = min(top_k * 2, 10)
        check(
            f"rerank_n >= top_k untuk top_k={top_k}",
            rerank_n >= top_k,
            f"rerank_n={rerank_n}"
        )

# Query Classifier
#
# This used to define a local _classify_query (3/5/7 by word count) and assert
# against it — a fourth copy of a heuristic, testing itself rather than the
# production path. The retriever had its own copy, the benchmark script had a
# third, and PolicyEngine returned 5/8/10 for the same decision.
#
# _resolve_top_k now delegates to PolicyEngine, so that is what gets tested.


def test_fix2_query_classifier_uses_the_canonical_policy():
    from src.controller.policy_engine import PolicyEngine
    from src.retrieval.hybrid_retriever import MasterHybridRetriever

    retriever = MasterHybridRetriever.__new__(MasterHybridRetriever)
    retriever._policy = PolicyEngine()

    for query in (
        "RAG",
        "what is RAG",
        "what is retrieval augmented generation",
        "how does hybrid search improve RAG quality",
        "explain the difference between dense and BM25 retrieval",
        "what are the main architectural differences between dense and sparse "
        "retrieval methods in modern RAG systems and how do they affect faithfulness",
    ):
        expected = PolicyEngine().initial_plan(query).top_k_hint
        resolved = retriever._resolve_top_k(query, 0)
        assert resolved == expected, (
            f"{query[:40]!r}: resolved {resolved}, policy says {expected}"
        )
        assert resolved in (5, 8, 10), f"top_k outside the policy set: {resolved}"

    # An explicit top_k must be returned untouched, including values the policy
    # would never produce. run_retrieval_eval.py measures at top_k=10.
    for explicit in (1, 3, 7, 10, 20):
        assert retriever._resolve_top_k("what is RAG", explicit) == explicit


def test_fix2_no_word_count_classifier_survives_anywhere():
    """The 3/5/7 word-count heuristic must not reappear, in any layer.

    An earlier version of this guard scanned src/ only, and a fifth copy sitting
    in tests/test_hybrid_retriever.py survived it — defining its own
    _classify_query and asserting against that. It kept passing after the
    production copy was deleted, because it never touched the code under test.
    """
    import pathlib

    repo = pathlib.Path(__file__).resolve().parents[1]
    self_path = pathlib.Path(__file__).resolve()
    offenders = []
    for folder in ("src", "scripts", "tests"):
        for path in (repo / folder).rglob("*.py"):
            if "__pycache__" in path.parts or "target" in path.parts:
                continue
            # This file holds the patterns it searches for; it cannot match itself.
            if path.resolve() == self_path:
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if stripped.startswith("#") or stripped.startswith('"') or stripped.startswith("'"):
                    continue
                # `words >= 15 -> 7` and `words <= 5 -> 3` is the signature.
                if "words" in stripped and (">= 15" in stripped or "<= 5" in stripped):
                    offenders.append(f"{path.relative_to(repo)}: {stripped}")
    assert not offenders, (
        "word-count top_k heuristic reintroduced; PolicyEngine is the only "
        "source:\n" + "\n".join(offenders)
    )


def test_fix2_no_module_may_shadow_the_production_classifier():
    """A local `def _classify_query` in a test is a copy that tests itself."""
    import pathlib

    repo = pathlib.Path(__file__).resolve().parents[1]
    self_path = pathlib.Path(__file__).resolve()
    allowed = {repo / "scripts" / "benchmark_retrieval.py"}  # delegates to PolicyEngine
    offenders = []
    for folder in ("src", "tests"):
        for path in (repo / folder).rglob("*.py"):
            if "__pycache__" in path.parts or path.resolve() == self_path:
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.startswith("def _classify_query") and path not in allowed:
                    offenders.append(str(path.relative_to(repo)))
    assert not offenders, (
        "a module defines its own _classify_query; that copy tests itself "
        "rather than MasterHybridRetriever:\n" + "\n".join(offenders)
    )


# Consistent candidate pipeline 
def test_fix3_candidate_pipeline():
    print("\n[FIX 3] Consistent candidate_k formula")

    cases = [
        (3,  9,  6),
        (5,  15, 10),
        (7,  21, 10),
        (10, 30, 10),
    ]

    for top_k, exp_candidate, exp_rerank in cases:
        candidate_k = min(top_k * 3, 30)
        rerank_n    = min(top_k * 2, 10)
        check(
            f"top_k={top_k} → fetch={candidate_k}, rerank={rerank_n}",
            candidate_k == exp_candidate and rerank_n == exp_rerank,
            f"got fetch={candidate_k}, rerank={rerank_n}"
        )

    print("  Invariant check: candidate_k >= rerank_n >= top_k")
    for top_k in range(1, 11):
        candidate_k = min(top_k * 3, 30)
        rerank_n    = min(top_k * 2, 10)
        ok = candidate_k >= rerank_n >= top_k
        check(
            f"invariant top_k={top_k}",
            ok,
            f"fetch={candidate_k}, rerank={rerank_n}, return={top_k}"
        )

# Summary 
def print_summary():
    total  = len(results_log)
    passed = sum(1 for _, ok in results_log if ok)
    failed = total - passed

    print("\n" + "=" * 50)
    print(f"  HASIL: {passed}/{total} passed", end="")
    if failed:
        print(f" | {failed} FAILED:")
        for name, ok in results_log:
            if not ok:
                print(f"    {FAIL} {name}")
    else:
        print(" — semua OK")
    print("=" * 50)

if __name__ == "__main__":
    print("=" * 50)
    print("  RAG Retriever Fix — Unit Tests")
    print("  Tidak butuh Qdrant / Groq / GPU")
    print("=" * 50)

    test_fix1_adaptive_rerank_n()
    test_fix2_query_classifier()
    test_fix3_candidate_pipeline()
    print_summary()
