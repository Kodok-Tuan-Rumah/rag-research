"""Build retrieval evaluation dataset v3 from all known sources.

EXACT_ANSWER_SPAN > TOKEN_RECALL_0.8 only when demonstrable.
No manufactured gold.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.gold_matching import is_gold_hit, gold_rank
from src.evaluation.gold_data_validator import validate_dataset

FAIL = ROOT / "data" / "debug" / "failure_analysis.json"
ADV = ROOT / "data" / "adversarial" / "adversarial_queries.json"
RAGAS_DIR = ROOT / "results" / "metrics"
CHUNKS = ROOT / "data" / "processed" / "chunks_semantic_v2.json"
OUT = ROOT / "data" / "evaluation" / "retrieval_eval_dataset_v3.json"
AUDIT = ROOT / "results" / "metrics" / "gold_data_audit_v3.json"


def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def exact_span_match(question: str, answer: str, chunks: list[dict]) -> dict | None:
    n = norm_text(answer)
    if not n or len(n) < 20:
        return None
    candidates = [c for c in chunks if n in norm_text(c["text"])]
    if candidates:
        best = min(candidates, key=lambda c: len(c["text"]))
        return best
    return None


def token_recall_match(question: str, answer: str, chunks: list[dict]) -> tuple[dict | None, float]:
    import re as _re

    words = set(_re.findall(r"\w+", answer.lower()))
    if not words:
        return None, 0.0
    best, best_hit = None, 0.0
    for c in chunks:
        text_words = set(_re.findall(r"\w+", c["text"].lower()))
        if not text_words:
            continue
        rec = len(words & text_words) / len(words)
        if rec > best_hit:
            best_hit, best = rec, c
    return best, best_hit


def derive_fields(idx: int, row: dict, match_type: str, chunk: dict) -> dict:
    return {
        "query_id": f"q{idx:02d}",
        "question": row.get("question") or row.get("query", ""),
        "gold_answer": row.get("gold_answer") or row.get("ground_truth") or row.get("answer", ""),
        "gold_chunk_id": chunk["chunk_id"],
        "gold_doc_id": chunk["doc_id"],
        "match_type": match_type,
        "provenance": row.get("provenance", "build_v3"),
        "question_type": "factual" if "what" in row.get("question", "").lower() and "?" in row.get("question", "") else "general",
        "difficulty": "medium",
        "should_abstain": False,
        "expected_behavior": row.get("expected_behavior", "answer"),
    }


def main() -> None:
    chunks = json.loads(CHUNKS.read_text())
    candidates = []

    for row in json.loads(FAIL.read_text()):
        question = row.get("query") or row.get("question", "")
        answer = row.get("ground_truth") or row.get("answer", "")
        if not question or not answer:
            continue
        exact = exact_span_match(question, answer, chunks)
        if exact:
            candidates.append({**row, "matched_json": expand(exact, "EXACT_ANSWER_SPAN")})
        else:
            chunk, rec = token_recall_match(question, answer, chunks)
            if chunk and rec >= 0.8:
                candidates.append({**row, "matched_json": expand(chunk, "TOKEN_RECALL_0.8")})
            else:
                candidates.append({**row, "matched_json": None})

    for adv in json.loads(ADV.read_text()):
        question = adv.get("question")
        if question is None:
            continue
        answer = adv.get("expected_behavior") and "abstain" or ""
        if adv.get("should_abstain"):
            candidates.append({**adv, "is_adversarial_abstain": True})
        else:
            candidates.append({**adv, "is_adversarial": True})

    # For ragas question rows, only include as gold if question exactly in FA derivation later.
    for path in RAGAS_DIR.glob("ragas_phase2_postfix*.json"):
        try:
            d = json.loads(path.read_text())
        except Exception:
            continue
        samples = []
        for ex in d:
            if isinstance(ex, dict):
                if "samples" in ex:
                    samples.extend(list(ex.get("samples", [])))
                elif "question" in ex:
                    samples.append(ex)
        for s in samples:
            if isinstance(s, dict) and "question" in s:
                candidates.append({**s, "source": "ragas", "matched_json": None})

    # derive gold from matched rows
    valid = []
    row_only = []
    for row in candidates:
        if "matched_json" in row and row["matched_json"]:
            mj = row["matched_json"]
            valid.append({
                "query_id": row.get("query_id") or f"esti",
                "question": row.get("question") or row.get("query", ""),
                "gold_answer": row.get("ground_truth") or row.get("answer", ""),
                "gold_chunk_id": mj["chunk_id"],
                "gold_doc_id": mj["doc_id"],
                "match_type": mj["match_type"],
                "provenance": {"source": row.get("source", "failure_analysis.json"), "matched_chunk": mj["chunk_id"], "match_type": mj["match_type"]},
                "question_type": "factual" if "what" in row.get("question", "").lower() else "general",
                "difficulty": "medium",
                "should_abstain": bool(row.get("should_abstain", False)),
                "expected_behavior": row.get("expected_behavior", "answer"),
            })
        else:
            row_only.append({**row, "status": "evaluation_row_only"})

    clean_rows = []
    seen = set()
    for v in valid:
        key = (v["question"].lower(), v["gold_chunk_id"])
        if key not in seen:
            seen.add(key); clean_rows.append(v)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(clean_rows, indent=2, ensure_ascii=False))
    audit = validate_dataset(clean_rows, chunks, require_answer=True)
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text(json.dumps(audit, indent=2, ensure_ascii=False))

    print(f"rows inspected={len(candidates)}")
    print(f"exact span derived={sum(1 for r in candidates if isinstance(r.get('matched_json'), dict) and r['matched_json'].get('match_type') == 'EXACT_ANSWER_SPAN')}")
    print(f"token recall derived={sum(1 for r in candidates if isinstance(r.get('matched_json'), dict) and r['matched_json'].get('match_type') == 'TOKEN_RECALL_0.8')}")
    print(f"evaluation_row_only={len(row_only)}")
    print(f"final valid gold rows={len(clean_rows)}")
    print(f"gold audit -> {AUDIT}")


def expand(chunk: dict, match_type: str) -> dict:
    return {"chunk_id": chunk["chunk_id"], "doc_id": chunk["doc_id"], "match_type": match_type}


if __name__ == "__main__":
    main()
