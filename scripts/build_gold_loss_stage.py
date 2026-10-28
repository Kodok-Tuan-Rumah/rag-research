from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIAG = ROOT / "results" / "debug" / "filter_gold_diagnosis.json"
GOLD_OUT = ROOT / "results" / "metrics" / "gold_loss_by_stage.json"
GOLD_MD = ROOT / "results" / "metrics" / "gold_loss_by_stage.md"
FILTER_OUT = ROOT / "results" / "metrics" / "filter_precision_audit.json"
FILTER_MD = ROOT / "results" / "metrics" / "filter_precision_audit.md"


def main() -> None:
    diag = json.loads(DIAG.read_text())
    rows = diag["rows"]

    # Build per-query lifecycle view
    lifecycle = []
    stage_gold_counts = {"retrieval": 0, "rrf": 0, "reranked": 0, "final": 0}
    for r in rows:
        in_rrf = bool(r.get("gold_in_rrf"))
        in_reranked = bool(r.get("gold_in_reranked"))
        in_final = bool(r.get("gold_in_final"))

        if in_rrf:
            stage_gold_counts["retrieval"] += 1
            stage_gold_counts["rrf"] += 1
        if in_reranked:
            stage_gold_counts["reranked"] += 1
        if in_final:
            stage_gold_counts["final"] += 1

        lost_stage = None
        if not in_rrf:
            lost_stage = "retrieval"
        elif not in_reranked:
            lost_stage = "reranker"
        elif not in_final:
            lost_stage = "filter"

        if r.get("classification") == "UNKNOWN" and r.get("filter_verdict") == "ok":
            adjusted_class = "NOT_FILTER_FAILURE"
        else:
            adjusted_class = r.get("classification")

        lifecycle.append(
            {
                "query_id": r["query_id"],
                "gold_chunk_id": r["gold_chunk_id"],
                "retrieved": in_rrf,
                "in_rrf": in_rrf,
                "in_reranked": in_reranked,
                "in_final": in_final,
                "lost_stage": lost_stage,
                "loss_reason": r.get("first_reason"),
                "filter_rule": r.get("filter_verdict"),
                "classification": adjusted_class,
            }
        )

    recall = {
        "retrieval_rrf": stage_gold_counts["rrf"] / len(rows),
        "reranker": stage_gold_counts["reranked"] / len(rows),
        "final": stage_gold_counts["final"] / len(rows),
    }

    payload = {"summary": {**stage_gold_counts, "total": len(rows)}, "lifecycle": lifecycle}
    GOLD_OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    lines = [
        "# Gold Loss by Stage",
        "",
        "| query_id | gold_chunk_id | in_rrf | in_reranked | in_final | lost_stage | loss_reason | filter_rule |",
        "|---|---|---:|---:|---:|---|---|---|",
    ]
    for r in lifecycle:
        lines.append(
            f"| {r['query_id']} | {r['gold_chunk_id']} | {r['in_rrf']} | {r['in_reranked']} | {r['in_final']} | {r['lost_stage'] or '-'} | {r['loss_reason'] or '-'} | {r['filter_rule'] or '-'} |"
        )
    lines.append("")
    lines.append(
        f"- retrieval/rrf gold present: {stage_gold_counts['rrf']}/{len(rows)} ({recall['retrieval_rrf']:.4f})"
    )
    lines.append(
        f"- reranked gold present: {stage_gold_counts['reranked']}/{len(rows)} ({recall['reranker']:.4f})"
    )
    lines.append(
        f"- final gold present: {stage_gold_counts['final']}/{len(rows)} ({recall['final']:.4f})"
    )
    GOLD_MD.write_text("\n".join(lines) + "\n")

    # simple filter audit view from classification
    from collections import Counter

    verdict_counts = Counter(r.get("filter_verdict") for r in rows)

    for r in rows:
        if r.get("classification") == "UNKNOWN" and r.get("filter_verdict") == "ok":
            r["classification"] = "NOT_FILTER_FAILURE"

    classification_counts = Counter(r.get("classification") for r in rows)
    # postprocess missing rows with verdict ok: treat as potential rank-cutoff, not filter failure
    overrides: dict[str, str] = {}
    for r in rows:
        if r.get("classification") == "UNKNOWN" and r.get("filter_verdict") == "ok":
            overrides[r["query_id"]] = "NOT_FILTER_FAILURE"
    for r in rows:
        if r["query_id"] in overrides:
            r["classification"] = overrides[r["query_id"]]
    filter_payload = {
        "verdict_counts": dict(verdict_counts),
        "classification_counts": dict(classification_counts),
        "rows": rows,
    }
    FILTER_OUT.write_text(json.dumps(filter_payload, indent=2, ensure_ascii=False))

    f_lines = ["# Filter Precision Audit", "", "| filter_rule | count |", "|---|---:|"]
    for verdict, count in sorted(verdict_counts.items(), key=lambda x: str(x[0])):
        f_lines.append(f"| {verdict} | {count} |")
    f_lines.append("")
    f_lines.append("| classification | count |")
    f_lines.append("|---|---:|")
    for c, count in sorted(classification_counts.items()):
        f_lines.append(f"| {c} | {count} |")
    FILTER_MD.write_text("\n".join(f_lines) + "\n")

    print("[saved] gold_loss_by_stage + filter_precision_audit")


if __name__ == "__main__":
    main()
