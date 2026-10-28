from __future__ import annotations

import glob
import json
from pathlib import Path

INV = Path("results/metrics/evidence_inventory.json")
INV_MD = Path("results/metrics/evidence_inventory.md")

TARGET_GLOBS = [
    ("results/metrics/**/*.json",),
    ("data/debug/**/*.json",),
    ("data/adversarial/**/*.json",),
    ("tests/fixtures/**/*.json",),
    ("data/processed/**/*.json",),
]


def _iter_json():
    for (pat,) in TARGET_GLOBS:
        yield from glob.glob(pat, recursive=True)


def _has_key(obj, keys) -> bool:
    return any(k in obj for k in keys)


def _inspect(path: str) -> dict:
    p = Path(path)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"path": str(p), "error": f"unparseable: {exc}"}

    row = {
        "path": str(p),
        "num_items": None,
        "has_question": False,
        "has_answer": False,
        "has_ground_truth": False,
        "has_gold_chunk_id": False,
        "has_gold_doc_id": False,
        "has_answer_span": False,
        "recoverable": False,
        "category": "unknown",
        "notes": "",
    }

    def _scan_obj(obj):
        if isinstance(obj, dict):
            if _has_key(obj, ("question", "query")):
                row["has_question"] = True
            if _has_key(obj, ("answer", "gold_answer", "ground_truth")):
                row["has_answer"] = True
            if "answer_span" in obj:
                row["has_answer_span"] = True
            if any(k in obj for k in ("gold_chunk_id", "source_chunk", "source_chunk_id")):
                row["has_gold_chunk_id"] = True
            if any(k in obj for k in ("gold_doc_id", "doc_id")):
                row["has_gold_doc_id"] = True
            for v in obj.values():
                _scan_obj(v)
        elif isinstance(obj, list):
            if obj:
                row["num_items"] = (row["num_items"] or 0) + len(obj)
            for item in obj:
                _scan_obj(item)

    _scan_obj(data)

    if row["has_gold_chunk_id"]:
        row["category"] = "direct_gold"
        row["recoverable"] = True
    elif row["has_answer"] and row["has_question"]:
        row["category"] = "answer_span_gold"
        row["recoverable"] = True
    elif row["has_question"] and "adversarial" in str(p):
        row["category"] = "adversarial_only"
        row["recoverable"] = False
    elif row["has_question"]:
        row["category"] = "question_only"
        row["recoverable"] = False
    else:
        row["category"] = "unknown"

    if "stale" in str(p):
        row["category"] = "stale_gold"
    note = []
    if row["has_gold_chunk_id"]:
        note.append("has direct gold id; must validate against corpus")
    if row["has_answer"] and not row["has_gold_chunk_id"]:
        note.append("answer present but no gold id; needs span-level derivation")
    if "ragas" in str(p):
        note.append("ragas style eval; question + metric only, no chunk evidence")
    if "ablation" in str(p):
        note.append("ablation artifact; source_chunk ids may be stale")
    row["notes"] = "; ".join(note)
    return row


def main() -> None:
    rows = []
    for pat_row in TARGET_GLOBS:
        pat = pat_row[0]
        for path in sorted(glob.glob(pat, recursive=True)):
            rows.append(_inspect(path))

    # filter huge py scripts duplicated entries (TARGET includes json only)
    INV.parent.mkdir(parents=True, exist_ok=True)
    INV.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Evidence Inventory",
        "",
        "| path | num_items | category | has_gold_chunk_id | has_answer | has_answer_span | recoverable |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['path']}` | {row.get('num_items', '')} | {row['category']} | {row['has_gold_chunk_id']} | {row['has_answer']} | {row['has_answer_span']} | {row['recoverable']} |"
        )
    INV_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[saved] {INV} and {INV_MD}")


if __name__ == "__main__":
    main()
