"""Label retrieval eval dataset items: question_type, difficulty, should_abstain.

Deterministic and evidence-aware: only the canonical derivation script
assigns these fields; this check ensures completeness and updates nothing
unless fields are missing.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.gold_data_validator import validate_dataset

DATA = ROOT / "data" / "evaluation" / "retrieval_eval_dataset.json"
CHUNKS = ROOT / "data" / "processed" / "chunks_semantic_v2.json"
OUT = ROOT / "results" / "metrics" / "eval_set_label_audit.json"
TRAIN = ROOT / "data" / "evaluation" / "retrieval_train_eval.json"
HOLDOUT = ROOT / "data" / "evaluation" / "retrieval_holdout_eval.json"


def question_type(q: str) -> str:
    words = q.lower().split()
    if any(w in words for w in ("what", "who", "when", "where", "which")):
        return "factual"
    if any(s in q.lower() for s in ("why", "how", "explain", "describe", "compare", "difference")):
        return "reasoning"
    return "general"


def difficulty(q: str) -> str:
    words = [w for w in re.sub(r"[?]", "", q).split() if w]
    if len(words) <= 8:
        return "easy"
    if len(words) <= 18:
        return "medium"
    return "hard"


def main() -> None:
    dataset = json.loads(DATA.read_text())
    issues = []
    for item in dataset:
        for key in (
            "query_id",
            "question",
            "gold_answer",
            "gold_chunk_id",
            "gold_doc_id",
            "match_type",
            "provenance",
            "question_type",
            "difficulty",
            "difficulty_method",
            "should_abstain",
        ):
            if key not in item:
                issues.append(f"{item.get('query_id')}: missing {key}")
        if item.get("question_type") not in {"factual", "reasoning", "general", "adversarial"}:
            issues.append(f"{item.get('query_id')}: bad question_type={item.get('question_type')}")
        if item.get("difficulty") not in {"easy", "medium", "hard"}:
            issues.append(f"{item.get('query_id')}: bad difficulty")

    qids = [i["query_id"] for i in dataset]
    questions = [i["question"].lower() for i in dataset]
    pairs = [(i["question"].lower(), i["gold_chunk_id"]) for i in dataset]

    if len(qids) != len(set(qids)):
        issues.append("duplicate query_id")
    if len(questions) != len(set(questions)):
        issues.append("duplicate question")
    if len(pairs) != len(set(pairs)):
        issues.append("duplicate (question, gold_chunk_id)")

    chunks = json.loads(CHUNKS.read_text())
    validation = validate_dataset(dataset, chunks, require_answer=True)
    if validation["summary"]["VALID_QA"] != len(dataset):
        issues.append("some items failed corpus validation")

    if TRAIN.exists() and HOLDOUT.exists():
        train = json.loads(TRAIN.read_text())
        holdout = json.loads(HOLDOUT.read_text())
        tq = {i["question"].lower() for i in train}
        hq = {i["question"].lower() for i in holdout}
        if tq & hq:
            issues.append(f"train/holdout question overlap={len(tq & hq)}")
        tp = {(i["question"].lower(), i["gold_chunk_id"]) for i in train}
        hp = {(i["question"].lower(), i["gold_chunk_id"]) for i in holdout}
        if tp & hp:
            issues.append(f"train/holdout pair overlap={len(tp & hp)}")

    report = {
        "total": len(dataset),
        "valid_via_corpus": validation["summary"]["VALID_QA"],
        "issues": issues,
        "summary_fields_missing": len([i for i in issues if "missing" in i]),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
