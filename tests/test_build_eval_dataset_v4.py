"""Merge protections for scripts/build_eval_dataset_v4.py.

The fragment this guards is `llm_generated_questions.json`, which no CI glob
reads — DATASET_GLOBS matches retrieval_eval_dataset*.json, the train and
holdout sets, and nothing else. A fragment with a duplicate id, a gold chunk
absent from the corpus, or a query that restates its chunk would reach the
dataset unexamined and be discovered only by a later gate run.

The embedder is injected throughout. The real encoder costs seconds to load
and needs the model on disk; what is under test is the guard logic.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_eval_dataset_v4 import (
    _record_key,
    check_category_floor,
    merge_fragment,
)

CHUNKS = [
    {"chunk_id": "c1", "text": "Attention compares query and key vectors."},
    {"chunk_id": "c2", "text": "Recall is the share of gold chunks retrieved."},
]

UNIT_X = (1.0, 0.0)
UNIT_Y = (0.0, 1.0)


def embedder_for(table: dict[str, tuple[float, float]]):
    return lambda texts: [table.get(t, (0.0, 0.0)) for t in texts]


SAFE = embedder_for({})  # everything orthogonal -> no leaks, ever


def row(qid="x1", question="q?", gold="c1", **extra):
    base = {"query_id": qid, "question": question, "gold_chunk_id": gold}
    base.update(extra)
    return base


# --- 1. merge combines the right number --------------------------------------


def test_merge_appends_the_fragment():
    out = merge_fragment([row("b1")], [row("f1", question="other?")], CHUNKS, embedder=SAFE)

    assert len(out["records"]) == 2
    assert out["report"]["accepted"] == 1
    assert out["report"]["total_after_merge"] == 2


def test_merge_preserves_the_existing_rows_unchanged():
    base = [row("b1", question="base?")]
    snapshot = json.dumps(base, sort_keys=True)

    merge_fragment(base, [row("f1", question="new?")], CHUNKS, embedder=SAFE)

    assert json.dumps(base, sort_keys=True) == snapshot


# --- 2. duplicates are rejected ----------------------------------------------


def test_duplicate_question_and_gold_is_dropped():
    frag = [row("f1", question="same?", gold="c1")]
    base = [row("b1", question="same?", gold="c1")]

    out = merge_fragment(base, frag, CHUNKS, embedder=SAFE)

    assert out["report"]["duplicate"] == 1
    assert len(out["records"]) == 1


def test_same_question_with_different_gold_is_kept():
    base = [row("b1", question="same?", gold="c1")]
    frag = [row("f1", question="same?", gold="c2")]

    out = merge_fragment(base, frag, CHUNKS, embedder=SAFE)

    assert out["report"]["duplicate"] == 0
    assert len(out["records"]) == 2


def test_duplicate_detection_is_case_and_space_insensitive():
    base = [row("b1", question="Same Question?")]
    frag = [row("f1", question="  same question?  ")]

    out = merge_fragment(base, frag, CHUNKS, embedder=SAFE)

    assert out["report"]["duplicate"] == 1


# --- 3. query_id collisions are renamed --------------------------------------


def test_colliding_query_ids_are_renamed():
    base = [row("q01", question="a?")]
    frag = [row("q01", question="b?")]

    out = merge_fragment(base, frag, CHUNKS, embedder=SAFE)

    ids = [r["query_id"] for r in out["records"]]
    assert len(set(ids)) == len(ids), f"ids collided: {ids}"
    assert out["report"]["renamed_ids"] == 1


def test_a_fragment_id_colliding_with_a_base_id_is_also_fixed():
    out = merge_fragment(
        [row("shared", question="a?")],
        [row("shared", question="b?")],
        CHUNKS,
        embedder=SAFE,
    )
    ids = [r["query_id"] for r in out["records"]]
    assert len(set(ids)) == 2


def test_blank_query_id_is_filled_in():
    frag = [{"question": "no id here?", "gold_chunk_id": "c1"}]

    out = merge_fragment([], frag, CHUNKS, embedder=SAFE)

    assert out["records"][0]["query_id"].strip() != ""


# --- 4. malformed rows and absent gold are dropped ---------------------------


def test_gold_chunk_absent_from_corpus_is_dropped():
    frag = [row("f1", gold="ghost_rs_9999")]

    out = merge_fragment([], frag, CHUNKS, embedder=SAFE)

    assert out["report"]["dropped_bad_gold"] == 1
    assert out["records"] == []


def test_row_without_a_question_is_dropped():
    frag = [{"query_id": "f1", "question": "   ", "gold_chunk_id": "c1"}]

    out = merge_fragment([], frag, CHUNKS, embedder=SAFE)

    assert out["report"]["dropped_malformed"] == 1


def test_non_dict_entry_is_dropped_without_raising():
    frag = [row("f1"), "not a dict", None, 42]

    out = merge_fragment([], frag, CHUNKS, embedder=SAFE)

    assert out["report"]["dropped_malformed"] == 3
    assert len(out["records"]) == 1


def test_multihop_row_with_one_unknown_gold_is_dropped():
    frag = [{"query_id": "m1", "question": "m?", "gold_chunk_ids": ["c1", "ghost"]}]

    out = merge_fragment([], frag, CHUNKS, embedder=SAFE)

    assert out["report"]["dropped_bad_gold"] == 1


def test_abstain_row_without_gold_survives():
    frag = [{"query_id": "a1", "question": "capital of Peru?", "should_abstain": True,
             "gold_chunk_id": "", "gold_answer": ""}]

    out = merge_fragment([], frag, CHUNKS, embedder=SAFE)

    assert len(out["records"]) == 1
    assert out["records"][0]["should_abstain"] is True


# --- 5. §10.2 leakage is actually enforced ------------------------------------


def test_a_leaking_fragment_row_is_dropped():
    embed = embedder_for({
        "What does the passage say?": UNIT_X,
        "Attention compares query and key vectors.": UNIT_X,
    })
    frag = [row("leaky", question="What does the passage say?", gold="c1")]

    out = merge_fragment([], frag, CHUNKS, embedder=embed)

    assert out["report"]["dropped_leaking"] == 1
    assert out["records"] == []
    assert out["report"]["leaking_ids"] == ["leaky"]


def test_leakage_check_reports_how_many_rows_it_scored():
    """A guard that scores nothing must not read as a guard that passed."""
    frag = [row("f1", question="totally different text", gold="c1")]

    out = merge_fragment([], frag, CHUNKS, embedder=SAFE)

    assert out["report"]["leakage_scored"] == 1
    assert out["report"]["dropped_leaking"] == 0


def test_leakage_is_not_reported_as_checked_without_a_corpus():
    out = merge_fragment([], [row("f1")], [], embedder=SAFE)

    assert "leakage_scored" not in out["report"]


# --- 6. relabelling -----------------------------------------------------------


def test_merge_type_overrides_the_fragment_category():
    frag = [row("f1", question_type="lexical")]

    out = merge_fragment([], frag, CHUNKS, question_type="multihop", embedder=SAFE)

    assert out["records"][0]["question_type"] == "multihop"
    assert out["report"]["relabelled"] == 1


def test_without_merge_type_the_original_category_is_kept():
    frag = [row("f1", question_type="lexical")]

    out = merge_fragment([], frag, CHUNKS, embedder=SAFE)

    assert out["records"][0]["question_type"] == "lexical"
    assert out["report"]["relabelled"] == 0


# --- 7. §10.3 category floor is reported, not enforced -----------------------


def test_category_floor_flags_a_thin_category():
    records = [row(str(i), question_type="lexical") for i in range(9)]
    records.append(row("adv", question_type="adversarial"))

    short = check_category_floor(records)

    assert len(short) == 1
    assert "adversarial" in short[0]
    assert "lexical" not in short[0]


def test_category_floor_is_silent_when_balanced():
    records = [row(str(i), question_type="lexical") for i in range(5)]
    records += [row(f"a{i}", question_type="adversarial") for i in range(5)]

    assert check_category_floor(records) == []


def test_category_floor_does_not_delete_anything():
    """It reports. Trimming rows to satisfy a percentage would destroy the
    set to meet a number, which is the opposite of what §10.3 is for."""
    records = [row(str(i), question_type="lexical") for i in range(19)]
    records.append(row("adv", question_type="adversarial"))

    assert check_category_floor(records)  # flags it
    assert len(records) == 20  # and nothing was removed


def test_category_floor_ignores_an_empty_dataset():
    assert check_category_floor([]) == []


# --- record key ---------------------------------------------------------------


def test_record_key_handles_plural_gold():
    key = _record_key({"question": "Q?", "gold_chunk_ids": ["c9", "c8"]})
    assert key == ("q?", "c9")


def test_record_key_handles_a_missing_question():
    assert _record_key({"gold_chunk_id": "c1"}) == ("", "c1")