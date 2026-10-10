"""The CI contract gate must be able to fail.

`tests/test_ragas_importable.py` already covers the two import-time failures
this gate was created for. This file checks the gate itself: that it passes on
the current tree and that it reports problems when the data is wrong. A gate
that cannot fail is the defect that produced the previous `rag-eval` job, which
pointed `eval_ci.py` at data it could never parse and stayed green forever.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from scripts import check_ci_contracts as contracts

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_every_src_module_imports():
    """Regression guard: ragas used to fail to import, silently.

    Two separate failures hid here. ragas 0.2.15 imports
    langchain_community.chat_models.vertexai, removed in 0.4.2; and
    ragas/executor.py calls nest_asyncio.apply() at module level, which breaks
    asyncio.wait_for on Python 3.14 so every metric returns NaN.
    """
    assert contracts.check_imports() == []


def test_src_module_listing_covers_the_evaluation_surface():
    names = set(contracts.src_modules())
    assert names, "src/ yielded no modules"
    for expected in ("src.evaluation.ragas_evaluator",
                     "src.evaluation.retrieval_evaluator",
                     "src.retrieval.hybrid_retriever",
                     "src.generation.generator"):
        assert expected in names, f"{expected} missing from the import gate"


def test_adversarial_queries_are_not_a_labeled_dataset():
    """They have no reference answers, so they cannot be a quality gate.

    The old workflow fed this file to eval_ci.load_samples(), which requires
    answer/contexts/ground_truth and raised ValueError on every run.
    """
    path = contracts.REPO_ROOT / "data" / "adversarial" / "adversarial_queries.json"
    if not path.exists():
        pytest.skip("adversarial_queries.json not present")
    rows = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(rows, dict):
        rows = rows.get("records") or rows.get("samples") or [rows]

    for row in rows:
        assert "answer" not in row and "response" not in row
        assert "contexts" not in row
        assert "ground_truth" not in row


def test_check_datasets_reports_a_broken_gold_chunk(tmp_path, monkeypatch):
    chunks = [{"chunk_id": "c1", "doc_id": "d1"}]
    chunks_path = tmp_path / "chunks.json"
    chunks_path.write_text(json.dumps(chunks), encoding="utf-8")

    eval_dir = tmp_path / "evaluation"
    eval_dir.mkdir()
    (eval_dir / "retrieval_eval_dataset_broken.json").write_text(json.dumps({
        "records": [{
            "query_id": "q1",
            "question": "What is retrieval?",
            "ground_truth": "Retrieval finds evidence.",
            "gold_chunk_ids": ["chunk_that_does_not_exist"],
        }]
    }), encoding="utf-8")

    monkeypatch.setattr(contracts, "CHUNKS_PATH", chunks_path)
    monkeypatch.setattr(contracts, "EVAL_DIR", eval_dir)

    failures = contracts.check_datasets(chunks)
    assert failures, "a gold chunk id missing from the corpus must be reported"
    assert "MISSING_GOLD_CHUNK" in failures[0]


def test_check_datasets_accepts_a_consistent_dataset(tmp_path, monkeypatch):
    chunks = [{"chunk_id": "c1", "doc_id": "d1"}]
    chunks_path = tmp_path / "chunks.json"
    chunks_path.write_text(json.dumps(chunks), encoding="utf-8")

    eval_dir = tmp_path / "evaluation"
    eval_dir.mkdir()
    (eval_dir / "retrieval_eval_dataset_ok.json").write_text(json.dumps({
        "records": [{
            "query_id": "q1",
            "question": "What is retrieval?",
            "ground_truth": "Retrieval finds evidence.",
            "gold_chunk_ids": ["c1"],
            "gold_doc_id": "d1",
        }]
    }), encoding="utf-8")

    monkeypatch.setattr(contracts, "CHUNKS_PATH", chunks_path)
    monkeypatch.setattr(contracts, "EVAL_DIR", eval_dir)

    assert contracts.check_datasets(chunks) == []


def test_real_datasets_validate_against_the_real_corpus():
    """The shipped datasets must satisfy the gate CI runs."""
    assert contracts.check_datasets(
        json.loads(contracts.CHUNKS_PATH.read_text(encoding="utf-8"))
    ) == []


def test_gold_data_validator_cli_exits_non_zero_on_invalid_rows():
    """It used to always exit 0, so wiring it into CI could never fail."""
    from src.evaluation import gold_data_validator as v

    report = {"summary": {"TOTAL_QA": 3, "VALID_QA": 3}}
    assert v._exit_code(report) == 0

    report = {"summary": {"TOTAL_QA": 3, "VALID_QA": 1}}
    assert v._exit_code(report) == 1


# --- evaluator-schema gate -----------------------------------------------
#
# validate_dataset() accepts the legacy aliases source_chunk,
# source_chunk_id, chunk_id and gold_chunk so old datasets can still be
# audited. normalize_gold_chunk_ids() — the reader the evaluator actually
# uses — accepts only gold_chunk_id or gold_chunk_ids, and
# run_retrieval_eval.py calls it with no try/except.
#
# The first version of this gate only ran validate_dataset(), so a dataset
# using an alias passed CI and then aborted the whole evaluation run.


def _write_dataset(directory: pathlib.Path, records: list[dict], name: str = "retrieval_eval_dataset.json") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(json.dumps({"records": records}), encoding="utf-8")


def test_evaluable_schema_rejects_a_row_the_evaluator_cannot_read(tmp_path, monkeypatch):
    """The exact gap: passes validation, crashes the evaluation run."""
    eval_dir = tmp_path / "evaluation"
    _write_dataset(eval_dir, [{
        "query_id": "q1",
        "question": "Q?",
        "ground_truth": "A.",
        "source_chunk": "doc_rs_0001",  # legacy alias
    }])
    monkeypatch.setattr(contracts, "EVAL_DIR", eval_dir)

    failures = contracts.check_evaluable_schema()
    assert failures, "a row using a legacy alias must be reported"
    assert "q1" in failures[0]


def test_legacy_alias_still_passes_corpus_validation(tmp_path, monkeypatch):
    """Pins the divergence itself: the two checks genuinely disagree.

    If this ever starts failing, the validator tightened and the gap is gone.
    """
    from src.evaluation.gold_data_validator import validate_dataset

    records = [{
        "query_id": "q1",
        "question": "Q?",
        "ground_truth": "A.",
        "source_chunk": "c1",
    }]
    chunks = [{"chunk_id": "c1", "doc_id": "d1"}]
    report = validate_dataset(records, chunks, require_answer=True)
    assert report["summary"]["VALID_QA"] == 1

    eval_dir = tmp_path / "evaluation"
    _write_dataset(eval_dir, records)
    monkeypatch.setattr(contracts, "EVAL_DIR", eval_dir)
    assert contracts.check_evaluable_schema(), "divergence vanished; update this test"


def test_evaluable_schema_accepts_canonical_rows(tmp_path, monkeypatch):
    eval_dir = tmp_path / "evaluation"
    _write_dataset(eval_dir, [
        {"query_id": "q1", "question": "Q?", "ground_truth": "A.",
         "gold_chunk_ids": ["c1"]},
        {"query_id": "q2", "question": "Q?", "ground_truth": "A.",
         "gold_chunk_id": "c2"},
    ])
    monkeypatch.setattr(contracts, "EVAL_DIR", eval_dir)

    assert contracts.check_evaluable_schema() == []


def test_real_datasets_are_readable_by_the_evaluator():
    assert contracts.check_evaluable_schema() == []


# --- corpus fingerprint gate ----------------------------------------------
#
# PROJECT_GOALS.md §15 requires dataset and index versioning. The dataset
# envelopes already carried corpus_chunks_sha256 and corpus_documents_sha256,
# and both matched. Nothing read them, so any writer could replace the corpus
# and every baseline recorded against it would silently become meaningless.


def test_real_corpus_matches_the_recorded_fingerprint():
    assert contracts.check_corpus_fingerprint() == []


def test_fingerprint_detects_a_replaced_corpus(tmp_path, monkeypatch):
    corpus = tmp_path / "chunks.json"
    corpus.write_text('[{"chunk_id": "c1"}]', encoding="utf-8")
    docs = tmp_path / "documents.json"
    docs.write_text('[{"doc_id": "d1"}]', encoding="utf-8")

    monkeypatch.setattr(contracts, "CHUNKS_PATH", corpus)
    monkeypatch.setattr(contracts, "DOCUMENTS_PATH", docs)

    eval_dir = tmp_path / "evaluation"
    eval_dir.mkdir()
    (eval_dir / "retrieval_eval_dataset.json").write_text(json.dumps({
        "corpus_chunks_sha256": "0" * 64,
        "corpus_documents_sha256": "0" * 64,
        "records": [],
    }), encoding="utf-8")
    monkeypatch.setattr(contracts, "EVAL_DIR", eval_dir)

    failures = contracts.check_corpus_fingerprint()
    # Both recorded hashes are wrong, so both must be reported — not just the
    # first one to fail.
    assert len(failures) == 2
    assert any("corpus_chunks_sha256" in f for f in failures)
    assert any("corpus_documents_sha256" in f for f in failures)
    for failure in failures:
        assert "metric" in failure and "invalid" in failure, (
            "the message must tell the operator their recorded metrics no "
            "longer describe this corpus"
        )


def test_fingerprint_is_inactive_without_a_recorded_hash(tmp_path, monkeypatch):
    """A dataset built against another corpus is not corruption."""
    eval_dir = tmp_path / "evaluation"
    eval_dir.mkdir()
    (eval_dir / "retrieval_eval_dataset.json").write_text(
        json.dumps({"records": []}), encoding="utf-8")
    monkeypatch.setattr(contracts, "EVAL_DIR", eval_dir)

    failures = contracts.check_corpus_fingerprint()
    assert len(failures) == 1
    assert "fingerprint check is inactive" in failures[0]


def test_chunker_no_longer_writes_the_live_corpus():
    """src/ingestion/chunker.py:469 used to overwrite the indexed corpus.

    chunk_semantic emits 582 chunks with {doc}_s0000 ids against the live 796
    with {doc}_rs_0000 ids, zero overlap. One direct invocation replaced the
    whole index and invalidated every recorded baseline.
    """
    source = (REPO_ROOT / "src" / "ingestion" / "chunker.py").read_text(encoding="utf-8")
    main_block = source.split('if __name__ == "__main__":', 1)[1]

    assert 'save_chunks(chunks, "data/processed/chunks_semantic_v2.json")' not in main_block
    assert "chunks_legacy_chunker.json" in main_block