import json
from pathlib import Path

import pytest

from src.retrieval.hybrid_retriever import MasterHybridRetriever


class FakeCrossEncoder:
    max_seq_length = 512

    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def predict(self, pairs, **kwargs):
        self.calls.append((pairs, kwargs))
        return self.scores


def test_score_pairs_uses_model_limit_and_preserves_query_and_chunk_text():
    query = "What is the answer?"
    chunk_text = "A short answer-bearing chunk."
    pairs = [[query, chunk_text]]
    reranker = FakeCrossEncoder([0.25])
    retriever = MasterHybridRetriever.__new__(MasterHybridRetriever)
    retriever.reranker = reranker

    assert retriever.score_pairs(pairs) == [0.25]
    assert reranker.calls == [
        (pairs, {"batch_size": 32, "max_length": reranker.max_seq_length})
    ]
    assert reranker.calls[0][0][0] == [query, chunk_text]


def test_rerank_keeps_descending_score_order_deterministically():
    candidates = [
        {"chunk_id": "first", "text": "first text"},
        {"chunk_id": "second", "text": "second text"},
        {"chunk_id": "third", "text": "third text"},
    ]
    reranker = FakeCrossEncoder([0.2, 0.9, 0.9])
    retriever = MasterHybridRetriever.__new__(MasterHybridRetriever)
    retriever.reranker = reranker

    ranked = retriever.rerank("query", candidates)

    assert [item["chunk_id"] for item in ranked] == ["second", "third", "first"]
    assert [item["rerank_score"] for item in ranked] == [0.9, 0.9, 0.2]


def test_short_and_q20_like_inputs_fit_model_limit_without_losing_target_evidence():
    from transformers import AutoTokenizer

    model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    except OSError:
        pytest.skip("Reranker tokenizer is not cached locally.")

    records = json.loads(
        (Path(__file__).resolve().parents[1] / "data/evaluation/retrieval_eval_dataset_v2.json")
        .read_text(encoding="utf-8")
    )["records"]
    q20 = next(row for row in records if row["query_id"] == "q20")
    chunks = json.loads(
        (Path(__file__).resolve().parents[1] / "data/processed/chunks_semantic_v2.json")
        .read_text(encoding="utf-8")
    )
    gold_id = q20["gold_chunk_ids"][0]
    gold_text = next(chunk["text"] for chunk in chunks if chunk["chunk_id"] == gold_id)
    max_length = 512
    reranker = FakeCrossEncoder([0.5])
    retriever = MasterHybridRetriever.__new__(MasterHybridRetriever)
    retriever.reranker = reranker
    pair = [[q20["question"], gold_text]]
    assert retriever.score_pairs(pair) == [0.5]
    assert reranker.calls[0] == (
        pair,
        {"batch_size": 32, "max_length": max_length},
    )

    short_query = records[0]["question"]
    short_text = next(
        chunk["text"] for chunk in chunks if chunk["chunk_id"] == records[0]["gold_chunk_ids"][0]
    )
    short_full = tokenizer(short_query, short_text, add_special_tokens=True, truncation=False)
    short_limited = tokenizer(
        short_query,
        short_text,
        add_special_tokens=True,
        truncation="longest_first",
        max_length=max_length,
    )
    assert len(short_full["input_ids"]) < max_length
    assert short_limited["input_ids"] == short_full["input_ids"]
    assert tokenizer(
        short_query,
        short_text,
        add_special_tokens=True,
        truncation="longest_first",
        max_length=max_length,
    )["input_ids"] == short_limited["input_ids"]

    full = tokenizer(
        q20["question"],
        gold_text,
        add_special_tokens=True,
        truncation=False,
        return_offsets_mapping=True,
    )
    limited = tokenizer(
        q20["question"],
        gold_text,
        add_special_tokens=True,
        truncation="longest_first",
        max_length=max_length,
        return_offsets_mapping=True,
    )
    assert len(full["input_ids"]) > max_length
    assert len(limited["input_ids"]) == max_length
    assert tokenizer(
        q20["question"],
        gold_text,
        add_special_tokens=True,
        truncation="longest_first",
        max_length=max_length,
    )["input_ids"] == limited["input_ids"]

    sequence_ids = full.sequence_ids()
    target_start = gold_text.index("average IRR is 0.92")
    target_token_indices = [
        i
        for i, (sequence_id, (start, end)) in enumerate(
            zip(sequence_ids, full["offset_mapping"])
        )
        if sequence_id == 1 and start <= target_start < end
    ]
    assert target_token_indices
    assert target_token_indices[0] < max_length
    shorter = tokenizer(
        q20["question"],
        gold_text,
        add_special_tokens=True,
        truncation="longest_first",
        max_length=256,
        return_offsets_mapping=True,
    )
    shorter_sequence_ids = shorter.sequence_ids()
    assert not any(
        sequence_id == 1 and start <= target_start < end
        for sequence_id, (start, end) in zip(
            shorter_sequence_ids, shorter["offset_mapping"]
        )
    )

    limited_sequence_ids = limited.sequence_ids()
    assert sum(sequence_id == 0 for sequence_id in limited_sequence_ids) == sum(
        sequence_id == 0 for sequence_id in full.sequence_ids()
    )
