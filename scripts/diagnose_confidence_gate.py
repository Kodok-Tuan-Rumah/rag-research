#!/usr/bin/env python3
"""Diagnose pre-generation confidence gate for Phase 1 answerable cases.

Runs the 15 answerable cases from the frozen Phase 1 smoke set through the
real retrieval + confidence path and writes a JSON summary of the confidence
decision, score, retrieval overlap, and top-k outcome. The script never calls
the LLM when the gate rejects; it records the rejection reason instead.

Usage (from rag-research/):
    QDRANT_URL=http://localhost:6333 \\
    QDRANT_COLLECTION=rag_research_phase1_v3 \\
    BM25_CHUNKS_PATH=data/processed/chunks_semantic_v3.json \\
    PYTHONPATH=. python scripts/diagnose_confidence_gate.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts import run_single_query as runner  # noqa: E402

DATASET = REPO_ROOT / "data/evaluation/phase1_smoke_set_v3.json"
OUTPUT = REPO_ROOT / "results/metrics/confidence_gate_diagnosis.json"


def main() -> int:
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    answerable = [r for r in dataset["records"] if r.get("should_abstain") is False]

    rows = []
    for case in answerable:
        try:
            result = runner.run_single_query(
                query=case["question"],
                mode="full",
                ground_truth=case.get("gold_answer"),
                save_output=False,
            )
            retrieval_confidence = result.get("retrieval_confidence") or {}
            rows.append(
                {
                    "query_id": case["query_id"],
                    "question": case["question"],
                    "status": result.get("status"),
                    "answer_status": result.get("answer_status"),
                    "failure_type": result.get("failure_type"),
                    "confidence_score": result.get("confidence_score"),
                    "confidence_decision": result.get("confidence_decision"),
                    "retrieval_confidence_score": retrieval_confidence.get("confidence_score"),
                    "retrieval_confidence_decision": retrieval_confidence.get("decision"),
                    "top_k": result.get("top_k"),
                    "chunks_used": len(result.get("chunks_used") or []),
                    "abstain_reason": result.get("abstain_reason"),
                    "provider_error": False,
                }
            )
        except Exception as exc:  # noqa: BLE001 - record, don't abort the batch
            rows.append(
                {
                    "query_id": case["query_id"],
                    "question": case["question"],
                    "status": "FAILED",
                    "answer_status": "failed",
                    "failure_type": "pipeline_error",
                    "confidence_score": None,
                    "confidence_decision": None,
                    "retrieval_confidence_score": None,
                    "retrieval_confidence_decision": None,
                    "top_k": None,
                    "chunks_used": 0,
                    "abstain_reason": f"{type(exc).__name__}: {exc}",
                    "provider_error": True,
                }
            )

    summary = {
        "total": len(rows),
        "answered": sum(1 for r in rows if r["status"] == "ANSWERED"),
        "abstained": sum(1 for r in rows if r["status"] == "INSUFFICIENT_CONTEXT"),
        "failed": sum(1 for r in rows if r["status"] == "FAILED"),
        "min_retrieval_confidence": min(
            (
                r["retrieval_confidence_score"]
                for r in rows
                if isinstance(r["retrieval_confidence_score"], (int, float))
            ),
            default=None,
        ),
        "max_retrieval_confidence": max(
            (
                r["retrieval_confidence_score"]
                for r in rows
                if isinstance(r["retrieval_confidence_score"], (int, float))
            ),
            default=None,
        ),
        "rows": rows,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    print(f"Report saved: {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
EOF