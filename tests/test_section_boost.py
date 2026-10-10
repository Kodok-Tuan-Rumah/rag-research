"""The section boost is what keeps references out of the candidate pool.

`results/metrics/k3_reference_impact.md` measured this on the shipped corpus:
258 of 796 chunks (32.4%) are REFERENCES sections, dense retrieval returns
17.9% of them and BM25 16.7%, but RRF fusion returns **0** — 0 of 520. The
mechanism is `_section_boost` giving references 0.3 against 1.4 for methods,
applied as `w * boost / (k + rank + 1)`.

Two consequences worth pinning:

- The strict filter never fires for references (`blocked_by_strict_filter`
  was 0 across all 13 canonical queries). The boost does the work upstream.
- `PolicyEngine.reference_policy()` returns `allow_references=True` for
  citation-shaped queries, but if references never survive fusion that policy
  has nothing to permit.

These are pure-function checks, so they need no Qdrant index and no network.
The end-to-end numbers live in `scripts/measure_reference_impact.py`.
"""

from __future__ import annotations

import pytest

from src.retrieval.hybrid_retriever import MasterHybridRetriever

boost = MasterHybridRetriever._section_boost


@pytest.mark.parametrize("section", [
    "REFERENCES",
    "references",
    "Bibliography",
    "6. REFERENCES AND NOTES",
])
def test_references_are_penalised(section):
    assert boost({"section": section}) < 1.0


@pytest.mark.parametrize("section", [
    "3. METHOD",
    "4. EXPERIMENTS",
    "METHODOLOGY",
    "ARCHITECTURE",
    "ABSTRACT",
    "CONCLUSION",
    "1 INTRODUCTION",
])
def test_content_sections_are_never_penalised(section):
    assert boost({"section": section}) >= 1.0


def test_method_sections_outrank_references():
    assert boost({"section": "3 METHOD"}) > boost({"section": "REFERENCES"})


def test_missing_section_is_neutral():
    assert boost({}) == 1.0
    assert boost({"section": ""}) == 1.0
    assert boost({"section": None}) == 1.0


def test_the_penalty_is_large_enough_to_clear_the_candidate_window():
    """Why 0.3 and not 0.9.

    RRF contribution is w * boost / (k + rank + 1) with k = rrf_k (60). A
    boost below ~0.5 halves a chunk's contribution; 0.3 cuts it to roughly a
    third. That is what pushed every reference out of the top 40 measured in
    k3_reference_impact.json. Raising this value toward 1.0 would let
    references back into the reranker's input.
    """
    ref = boost({"section": "REFERENCES"})
    method = boost({"section": "3 METHOD"})
    assert ref <= 0.5 * method, (
        "references lost the measured containment; the reranker would start "
        "scoring bibliography again"
    )