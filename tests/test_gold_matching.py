import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.gold_matching import (
    chunk_ids_match,
    doc_ids_match,
    gold_rank,
    is_gold_hit,
    is_subchunk_of,
)


def test_exact_match():
    assert chunk_ids_match("doc_rs_0001", "doc_rs_0001") is True


def test_subchunk_matches_parent():
    assert is_subchunk_of("doc_rs_0001_sub3", "doc_rs_0001") is True
    assert chunk_ids_match("doc_rs_0001", "doc_rs_0001_sub3") is True
    # sebaliknya: gold subchunk vs chunk parent
    assert chunk_ids_match("doc_rs_0001_sub3", "doc_rs_0001") is True


def test_non_subchunk_prefix_does_not_match():
    # "_rs_" adalah bagian nama chunk, bukan marker subchunk
    assert chunk_ids_match("doc_rs_0001", "doc_rs_00010") is False
    assert chunk_ids_match("doc_rs_0001", "doc_rs_0001x") is False


def test_empty_ids_do_not_match():
    assert chunk_ids_match("", "doc_rs_0001") is False
    assert chunk_ids_match(None, "doc_rs_0001") is False


def test_gold_hit_and_rank():
    retrieved = ["a", "b", "doc_rs_0001_sub2", "c"]
    assert is_gold_hit("doc_rs_0001", retrieved) is True
    assert gold_rank("doc_rs_0001", retrieved) == 3
    assert gold_rank("missing", retrieved) is None


def test_doc_ids_match_case_insensitive_and_missing():
    assert doc_ids_match("Doc_A", "doc_a") is True
    assert doc_ids_match("", "doc_a") is False
