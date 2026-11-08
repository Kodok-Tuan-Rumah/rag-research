"""Corpus coverage metrics + "what's not in corpus" report.

Computes coverage/recall metrics for the rag-research retrieval corpus and
emits a human-readable gap report listing domains/topics that are NOT covered
by the indexed corpus.

Design (P1-4):
  * Quantitative coverage: doc count, chunk count, char count, per-doc coverage.
  * Domain gap report: explicitly lists out-of-scope domains (biology, medicine,
    physics, etc.) that the system must NOT answer questions about.
  * Output is machine-readable (JSON) and human-readable (Markdown) so it can
    be checked into CI and reviewed.
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(REPO_ROOT))

CORPUS_PATH = REPO_ROOT / "data" / "processed" / "corpus_unified_v1.json"
DOCS_PATH = REPO_ROOT / "data" / "processed" / "documents.json"
CHUNKS_PATH = REPO_ROOT / "data" / "processed" / "chunks_semantic_v3.json"
OUT_JSON = REPO_ROOT / "results" / "metrics" / "corpus_coverage.json"
OUT_MD = REPO_ROOT / "results" / "metrics" / "corpus_coverage.md"

# Domains explicitly OUT of scope — the system must abstain on these.
OUT_OF_SCOPE_DOMAINS = [
    "Biology / life sciences",
    "Medicine / clinical research",
    "Chemistry",
    "Physics / materials science",
    "Engineering (non-LLM/non-RAG)",
    "Economics / finance",
    "Law / legal",
    "Social science",
    "General-world knowledge (non-RAG)",
]

# In-scope topic keywords — presence of these suggests a question IS in scope.
IN_SCOPE_KEYWORDS = [
    "retrieval-augmented generation", "rag", "large language model", "llm",
    "retrieval", "generation", "augmentation", "self-instruct", "llama",
    "attention", "transformer", "vector database", "reranker",
]


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_coverage(corpus: dict[str, Any]) -> dict[str, Any]:
    docs = corpus.get("documents", [])
    total_chunks = corpus.get("chunk_count", 0)
    total_docs = corpus.get("document_count", 0)

    # Per-doc chunk distribution
    per_doc = []
    for d in docs:
        chunks = d.get("chunks", [])
        chars = sum(len(c.get("text", "")) for c in chunks)
        per_doc.append(
            {
                "doc_id": d.get("doc_id"),
                "title": d.get("metadata", {}).get("title", d.get("doc_id")),
                "domain": d.get("metadata", {}).get("source_domain", "unknown"),
                "category": d.get("metadata", {}).get("category", "unknown"),
                "year": d.get("metadata", {}).get("year"),
                "arxiv_id": d.get("metadata", {}).get("arxiv_id", ""),
                "chunk_count": len(chunks),
                "char_count": chars,
            }
        )
    per_doc.sort(key=lambda x: x["chunk_count"], reverse=True)

    total_chars = sum(d["char_count"] for d in per_doc)

    # In-scope vs out-of-scope split
    in_scope_docs = [d for d in per_doc if d["domain"] == "rag_llm"]
    out_scope_docs = [d for d in per_doc if d["domain"] != "rag_llm"]

    coverage = {
        "index_version": corpus.get("index_version"),
        "generated_at": _dt.datetime.now().isoformat(),
        "document_count": total_docs,
        "chunk_count": total_chunks,
        "total_characters": total_chars,
        "avg_chunks_per_doc": round(total_chunks / total_docs, 2) if total_docs else 0,
        "avg_chars_per_doc": round(total_chars / total_docs, 2) if total_docs else 0,
        "in_scope_doc_count": len(in_scope_docs),
        "in_scope_chunk_count": sum(d["chunk_count"] for d in in_scope_docs),
        "out_of_scope_doc_count": len(out_scope_docs),
        "out_of_scope_chunk_count": sum(d["chunk_count"] for d in out_scope_docs),
        "in_scope_share_of_chunks": round(
            sum(d["chunk_count"] for d in in_scope_docs) / total_chunks, 4
        )
        if total_chunks
        else 0.0,
        "per_document": per_doc,
        "domain_coverage": {
            "in_scope": "rag_llm",
            "out_of_scope_domains": OUT_OF_SCOPE_DOMAINS,
        },
    }
    return coverage


def gap_report(corpus: dict[str, Any], coverage: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Corpus Coverage & Gap Report")
    lines.append("")
    lines.append(f"Index version: `{coverage['index_version']}`")
    lines.append(f"Generated: {coverage['generated_at']}")
    lines.append("")
    lines.append("## Coverage summary")
    lines.append("")
    lines.append(f"- Documents: {coverage['document_count']}")
    lines.append(f"- Chunks: {coverage['chunk_count']}")
    lines.append(f"- Total characters: {coverage['total_characters']:,}")
    lines.append(f"- Avg chunks/doc: {coverage['avg_chunks_per_doc']}")
    lines.append(f"- In-scope (RAG/LLM) docs: {coverage['in_scope_doc_count']}")
    lines.append(f"- Out-of-scope docs: {coverage['out_of_scope_doc_count']}")
    lines.append(f"- In-scope share of chunks: {coverage['in_scope_share_of_chunks']*100:.1f}%")
    lines.append("")
    lines.append("## Per-document coverage")
    lines.append("")
    lines.append("| doc_id | domain | category | year | arxiv_id | chunks | chars |")
    lines.append("|---|---|---|---|---|---|---|")
    for d in coverage["per_document"]:
        lines.append(
            f"| {d['doc_id']} | {d['domain']} | {d['category']} | "
            f"{d['year'] or '-'} | {d['arxiv_id'] or '-'} | "
            f"{d['chunk_count']} | {d['char_count']:,} |"
        )
    lines.append("")
    lines.append("## Out-of-scope domains (must abstain)")
    lines.append("")
    lines.append(
        "The corpus does NOT cover these domains. Questions about them must "
        "result in hard abstention, not generated answers."
    )
    lines.append("")
    for domain in OUT_OF_SCOPE_DOMAINS:
        lines.append(f"- {domain}")
    lines.append("")
    lines.append("## In-scope topic keywords")
    lines.append("")
    lines.append("Questions containing these terms are considered in-scope:")
    lines.append("")
    for kw in IN_SCOPE_KEYWORDS:
        lines.append(f"- {kw}")
    lines.append("")
    lines.append("## Recommendation")
    lines.append("")
    if coverage["out_of_scope_doc_count"] > 0:
        lines.append(
            "- The corpus contains out-of-scope documents. Review whether these "
            "belong in the RAG/LLM index or a separate collection."
        )
    if coverage["chunk_count"] < 1000:
        lines.append(
            "- The corpus is small (< 1000 chunks). Expect low recall for "
            "narrow or technical questions; abstain when retrieval is weak."
        )
    lines.append(
        "- Expand the corpus only through the unified ingestion pipeline "
        "(`scripts/unify_corpus.py`) with explicit metadata and a new index "
        "version."
    )
    return "\n".join(lines)


def main() -> int:
    if not CORPUS_PATH.exists():
        print(f"[ERROR] Unified corpus not found at {CORPUS_PATH}")
        print("       Run: python scripts/unify_corpus.py")
        return 1

    corpus = _load_json(CORPUS_PATH)
    coverage = compute_coverage(corpus)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(coverage, f, indent=2, ensure_ascii=False)
    OUT_MD.write_text(gap_report(corpus, coverage), encoding="utf-8")

    print(f"[coverage] {coverage['document_count']} docs, {coverage['chunk_count']} chunks")
    print(f"[coverage] in-scope share: {coverage['in_scope_share_of_chunks']*100:.1f}%")
    print(f"  -> {OUT_JSON.relative_to(REPO_ROOT)}")
    print(f"  -> {OUT_MD.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
