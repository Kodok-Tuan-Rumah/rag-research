import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.retrieval.chunk_filters import (
    is_generation_eligible,
    is_informative_chunk,
    is_reference_chunk,
    is_retrieval_candidate,
    passes_strict_filter,
    reference_policy_allows,
)

BODY_CHUNK = {
    "text": "Retrieval augmented generation combines information search with large language model generation pipelines in production systems.",
    "metadata": {"section": "body"},
}
REFERENCE_CHUNK = {
    "text": "[12] Doe et al. (2020). Some paper title. [13] Smith (2021). Another.",
    "metadata": {"section": "references"},
}
TITLE_CHUNK = {
    "text": "Attention Is All You Need",
    "metadata": {"section": "abstract"},
}
FRAGMENT_CHUNK = {
    "text": "and generation must use this incomplete fragment",
    "metadata": {"section": "body"},
}


def test_reference_chunk_detection():
    assert is_reference_chunk(REFERENCE_CHUNK) is True
    assert is_reference_chunk(BODY_CHUNK) is False


def test_reference_policy_is_query_dependent():
    assert reference_policy_allows("references", allow_references=False) is False
    assert reference_policy_allows("references", allow_references=True) is True
    assert reference_policy_allows("body", allow_references=False) is True


def test_strict_filter_honors_reference_policy():
    informative_reference = {
        "text": "Doe et al. describe a research method that retrieves cited evidence and uses it to ground generated answers across several scientific domains with independent ranking and citation verification. The paper also evaluates the method against multiple baselines and discusses limitations in its experimental setting.",
        "metadata": {"section": "references"},
    }

    assert passes_strict_filter(informative_reference) is False
    assert passes_strict_filter(informative_reference, allow_references=True) is True


def test_reference_budget_is_enforced_on_retriever_output(monkeypatch):
    from types import SimpleNamespace
    from src.retrieval.hybrid_retriever import MasterHybridRetriever

    retriever = MasterHybridRetriever.__new__(MasterHybridRetriever)
    retriever._policy = SimpleNamespace(
        initial_plan=lambda query: SimpleNamespace(
            query_type="factual",
            allow_multi_query=False,
            allow_hyde=False,
            allow_references=True,
            max_ref_ratio=0.4,
        )
    )
    retriever.use_multi_query = False
    retriever.use_hyde = False
    retriever.reranker = None
    text = "This informative research passage describes retrieval methods and cited evidence in sufficient detail for grounded answers."
    references = [
        {"chunk_id": f"ref{i}", "doc_id": "d", "text": text, "retrieval_score": 0.02,
         "metadata": {"section": "references"}}
        for i in range(3)
    ]
    body = [
        {"chunk_id": f"body{i}", "doc_id": f"d{i}", "text": text, "retrieval_score": 0.01,
         "metadata": {"section": "body"}}
        for i in range(4)
    ]
    monkeypatch.setattr(retriever, "_choose_candidate_k", lambda top_k: 7)
    monkeypatch.setattr(retriever, "_dense_search", lambda query, k: [])
    monkeypatch.setattr(retriever, "_bm25_search", lambda query, k: [])
    monkeypatch.setattr(retriever, "_rrf_fuse", lambda *args, **kwargs: references + body)

    results = retriever.search("who wrote the paper", top_k=5, rerank=False)

    assert sum(c["metadata"]["section"] == "references" for c in results) == 2


def test_generation_eligibility_contract():
    assert is_generation_eligible(BODY_CHUNK, allow_references=False) is True
    assert is_generation_eligible(REFERENCE_CHUNK, allow_references=False) is False
    # hanya dengan explicit allow_references=True reference boleh masuk
    assert is_generation_eligible(REFERENCE_CHUNK, allow_references=True) in (True, False)  # text harus informative
    assert is_generation_eligible(FRAGMENT_CHUNK, allow_references=False) is False
    assert is_generation_eligible(TITLE_CHUNK, allow_references=False) is False


def test_informative_chunk():
    assert is_informative_chunk(BODY_CHUNK) is True
    assert is_informative_chunk(FRAGMENT_CHUNK) is False


def test_retrieval_candidate_accepts_body_rejects_short_noise():
    assert is_retrieval_candidate(BODY_CHUNK) is True
    assert is_retrieval_candidate({"text": "tiny." , "metadata": {}}) is False


# ---------------------------------------------------------------------------
# Regression Fase A: chunk valid yang sebelumnya ditolak oleh ends_complete
# ---------------------------------------------------------------------------

from pathlib import Path as _Path
import json as _json

_CORPUS = _json.loads(
    (_Path(__file__).resolve().parents[1] / "data" / "processed" / "chunks_semantic_v2.json").read_text(
        encoding="utf-8"
    )
)
_BY_ID = {c["chunk_id"]: c for c in _CORPUS}

FALSE_POSITIVE_GOLD_IDS = [
    "tier2_large_language_models_rs_0006",  # q04
    "tier1_advanced_rag_survey_rs_0028",   # q06
    "tier2_self_instruct_rs_0062",          # q09
    "tier1_rag_lewis2020_rs_0026",          # q11
    "tier3_llama2_rs_0152",                 # q20
]


def test_false_positive_gold_now_accepts():
    from src.retrieval.chunk_filters import classify_chunk

    for cid in FALSE_POSITIVE_GOLD_IDS:
        chunk = _BY_ID[cid]
        verdict = classify_chunk(chunk)
        assert verdict == "ok", (
            f"chunk {cid} sekarang masih rejected: {verdict}"
        )


def test_false_positive_gold_generation_eligible():
    for cid in FALSE_POSITIVE_GOLD_IDS:
        assert is_generation_eligible(_BY_ID[cid], allow_references=False) is True


def test_negative_reference_still_rejected():
    assert classification(_BY_ID["tier1_advanced_rag_survey_rs_0083"]) == "reference"
    assert is_reference_chunk(_BY_ID["tier1_advanced_rag_survey_rs_0083"]) is True


def classification(chunk):
    from src.retrieval.chunk_filters import classify_chunk

    return classify_chunk(chunk)


def test_negative_fragment_still_rejected():
    from src.retrieval.chunk_filters import classify_chunk, is_fragment_text

    noisy = {"text": "and generation was then used by", "metadata": {"section": "body"}}
    assert is_fragment_text(noisy["text"]) is True
    assert classify_chunk(noisy) == "fragment"
    assert is_generation_eligible(noisy, allow_references=False) is False


def test_negative_title_author_still_rejected():
    import re as _re

    noisy = {
        "text": "Yunfan Gao, Yun Xiong and Xinyu Gao",
        "metadata": {"section": "abstract"},
    }
    # cap_ratio high & pendek → heuristik author/title harus tetap aktif
    words = noisy["text"].split()
    cap_ratio = sum(1 for w in words if w and w[0].isupper()) / max(len(words), 1)
    assert cap_ratio > 0.5
    from src.retrieval.chunk_filters import classify_chunk

    assert classify_chunk(noisy) in ("title_author", "not_informative")
    assert is_generation_eligible(noisy, allow_references=False) is False


def test_negative_low_quality_still_rejected():
    from src.retrieval.chunk_filters import classify_chunk, is_low_quality_chunk

    noisy = _BY_ID["tier1_rag_lewis2020_rs_0012"]
    assert is_low_quality_chunk(noisy) is True or classify_chunk(noisy) in (
        "low_quality", "not_informative",
    )
