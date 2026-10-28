from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

FAILURE_ANALYSIS_PATH = REPO_ROOT / "data" / "debug" / "failure_analysis.json"
CHUNKS_PATH = REPO_ROOT / "data" / "processed" / "chunks_semantic_v2.json"
OUTPUT_DATASET = REPO_ROOT / "data" / "evaluation" / "retrieval_eval_dataset.json"
OUTPUT_AUDIT = REPO_ROOT / "results" / "metrics" / "gold_data_audit.json"

TOKEN_RECALL_THRESHOLD = 0.70


def question_type(question: str) -> str:
    """Heuristic-only labelling. Not treated as ground truth."""
    q = question.lower().strip()
    words = q.split()
    if any(word in words for word in ("what", "who", "when", "where", "which")) or q.endswith(
        "who?"
    ):
        return "factual"
    if any(
        signal in q for signal in ("why", "how", "explain", "describe", "compare", "difference")
    ):
        return "reasoning"
    return "general"


def difficulty_for(question: str) -> str:
    words = [w for w in question.replace("?", "").split() if w]
    if len(words) <= 8:
        return "easy"
    if len(words) <= 18:
        return "medium"
    return "hard"


def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def _tokens(s: str) -> set:
    return set(re.findall(r"\w+", s.lower()))


def derive_gold(question: str, ground_truth: str, chunks: list) -> dict | None:
    gt_norm = _norm_text(ground_truth)
    gt_tokens = _tokens(ground_truth)
    if not gt_norm or not gt_tokens:
        return None

    # 1) Strong: exact containment
    exact = [c for c in chunks if gt_norm in _norm_text(c.get("text", ""))]
    if exact:
        best = min(exact, key=lambda c: len(c.get("text", "")))
        return {
            "gold_chunk_id": best["chunk_id"],
            "gold_doc_id": best["doc_id"],
            "match_type": "exact_span",
        }

    # 2) Token recall fallback
    candidates = []
    for c in chunks:
        cw = _tokens(c.get("text", ""))
        if not cw:
            continue
        recall = len(gt_tokens & cw) / len(gt_tokens)
        if recall >= TOKEN_RECALL_THRESHOLD:
            overlap = len(gt_tokens & cw)
            precision = overlap / len(cw)
            f1 = 2 * recall * precision / max(recall + precision, 1e-9)
            candidates.append((recall, f1, c))
    if candidates:
        candidates.sort(key=lambda t: (-t[0], -t[1], t[2]["chunk_id"]))
        best = candidates[0][2]
        return {
            "gold_chunk_id": best["chunk_id"],
            "gold_doc_id": best["doc_id"],
            "match_type": "token_recall_ge_0.7",
            "match_recall": round(candidates[0][0], 4),
        }
    return None


def main() -> None:
    rows = json.loads(FAILURE_ANALYSIS_PATH.read_text(encoding="utf-8"))
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))

    dataset = []
    unmapped = []
    for i, row in enumerate(rows):
        question = str(row.get("query", "")).strip()
        ground_truth = str(row.get("ground_truth", "")).strip()
        derived = derive_gold(question, ground_truth, chunks)
        if derived is None:
            unmapped.append({"question": question, "reason": "no_chunk_matched_answer_span"})
            continue
        dataset.append(
            {
                "query_id": f"q{i + 1:02d}",
                "question": question,
                "gold_answer": ground_truth,
                "gold_chunk_id": derived["gold_chunk_id"],
                "gold_doc_id": derived["gold_doc_id"],
                "match_type": derived["match_type"],
                "provenance": "derived_from_data/debug/failure_analysis.json",
                "question_type": question_type(question),
                "difficulty": difficulty_for(question),
                "difficulty_method": "heuristic_v1",
                "should_abstain": False,
                "answer_span": ground_truth if derived["match_type"] == "exact_span" else "",
            }
        )

    OUTPUT_DATASET.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_DATASET.write_text(json.dumps(dataset, indent=2, ensure_ascii=False), encoding="utf-8")

    from src.evaluation.gold_data_validator import validate_dataset

    report = validate_dataset(dataset, chunks, require_answer=True)
    report["derivation"] = {
        "source": str(FAILURE_ANALYSIS_PATH.relative_to(REPO_ROOT)),
        "rows_in_source": len(rows),
        "mapped_rows": len(dataset),
        "unmapped_rows": unmapped,
    }
    OUTPUT_AUDIT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_AUDIT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[dataset] {len(dataset)} items -> {OUTPUT_DATASET}")
    print(f"[unmapped] {len(unmapped)} rows (evaluation_data_error):")
    for u in unmapped:
        print(f"  - {u['question'][:60]}")
    print(f"[audit] {OUTPUT_AUDIT}")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
