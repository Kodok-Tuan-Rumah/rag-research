from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
INDEX_VERSION = "v1.0.0-unified-20261031"

RAG_CHUNKS_PATH = REPO_ROOT / "data" / "processed" / "chunks_semantic_v3.json"
RAG_DOCS_PATH = REPO_ROOT / "data" / "processed" / "documents.json"
ARXIV_CHUNKS_PATH = (
    REPO_ROOT.parent
    / "arxiv-research-assistant"
    / "python-ml"
    / "data"
    / "processed"
    / "arxiv_chunks.json"
)
OUT_PATH = REPO_ROOT / "data" / "processed" / "corpus_unified_v1.json"

# Source-domain labels — the rag-research corpus is RAG/LLM only.
SOURCE_DOMAIN_RAG_LLM = "rag_llm"
SOURCE_DOMAIN_ARXIV = "arxiv_general"

# ArXiv primary category → human-readable domain.
ARXIV_CATEGORY_MAP = {
    "cs.CL": "Computation and Language",
    "cs.AI": "Artificial Intelligence",
    "cs.LG": "Machine Learning",
    "cs.IR": "Information Retrieval",
    "cs.HC": "Human-Computer Interaction",
    "stat.ML": "Machine Learning (Statistics)",
}


def _sha12(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:12]


def _norm_arxiv_id(raw: str) -> str:
    if not raw:
        return ""
    s = str(raw).strip()
    s = re.sub(r"^arXiv:", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\.\d+$", "", s)  # strip version (e.g. v1)
    return s


def _extract_year(text: str) -> str | None:
    m = re.search(r"\b(19|20)\d{2}\b", text[:2000])
    return m.group(0) if m else None


def _doc_title(doc: dict) -> str:
    md = doc.get("metadata", {}) or {}
    for key in ("title", "paper_title", "arxiv_title"):
        if md.get(key):
            return str(md[key])
    # Fallback: derive a readable title from the doc_id.
    return doc.get("doc_id", "unknown").replace("_", " ").title()


def _doc_category(doc: dict) -> str:
    md = doc.get("metadata", {}) or {}
    if md.get("category"):
        return str(md["category"])
    tier = md.get("tier")
    if tier is not None:
        return f"RAG/LLM tier{tier}"
    return "RAG/LLM"


def _doc_arxiv_id(doc: dict) -> str:
    md = doc.get("metadata", {}) or {}
    raw = md.get("arxiv_id") or md.get("paper_id") or doc.get("arxiv_id") or ""
    return _norm_arxiv_id(raw)


def _doc_source_domain(doc: dict) -> str:
    md = doc.get("metadata", {}) or {}
    if md.get("source_domain"):
        return str(md["source_domain"])
    return SOURCE_DOMAIN_RAG_LLM


def _enrich_rag_document(doc: dict, index_version: str) -> dict:
    md = dict(doc.get("metadata", {}) or {})
    md.update(
        {
            "index_version": index_version,
            "source_domain": _doc_source_domain(doc),
            "category": _doc_category(doc),
            "title": _doc_title(doc),
            "year": _extract_year(doc.get("text", "")) or md.get("year"),
            "arxiv_id": _doc_arxiv_id(doc),
        }
    )
    return {
        "doc_id": doc.get("doc_id"),
        "text": doc.get("text", ""),
        "source": doc.get("source", ""),
        "tier": md.get("tier"),
        "metadata": md,
    }


def load_rag_corpus(index_version: str) -> list[dict]:
    if not RAG_DOCS_PATH.exists():
        print(f"[WARN] {RAG_DOCS_PATH} not found; skipping rag corpus")
        return []

    with open(RAG_DOCS_PATH, "r", encoding="utf-8") as f:
        docs = json.load(f)

    enriched_docs = [_enrich_rag_document(d, index_version) for d in docs]
    chunks_by_doc: dict[str, list[dict]] = {}
    if RAG_CHUNKS_PATH.exists():
        with open(RAG_CHUNKS_PATH, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        for c in chunks:
            chunks_by_doc.setdefault(c.get("doc_id"), []).append(c)

    unified_docs = []
    for doc in enriched_docs:
        did = doc["doc_id"]
        doc["chunks"] = chunks_by_doc.get(did, [])
        doc["chunk_count"] = len(doc["chunks"])
        unified_docs.append(doc)

    total_chunks = sum(d["chunk_count"] for d in unified_docs)
    print(
        f"[unify] rag-research: {len(unified_docs)} docs, {total_chunks} chunks "
        f"(domain={SOURCE_DOMAIN_RAG_LLM})"
    )
    return unified_docs


def load_arxiv_corpus(index_version: str) -> list[dict]:
    if not ARXIV_CHUNKS_PATH.exists():
        print(f"[unify] arxiv corpus not present at {ARXIV_CHUNKS_PATH} — skipped")
        return []

    with open(ARXIV_CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # Group chunks by arxiv_id into document-like records.
    by_arxiv: dict[str, dict] = {}
    for c in chunks:
        aid = _norm_arxiv_id(c.get("arxiv_id", ""))
        if aid not in by_arxiv:
            by_arxiv[aid] = {
                "doc_id": f"arxiv:{aid}",
                "arxiv_id": aid,
                "title": c.get("paper_title") or c.get("title") or aid,
                "year": c.get("year") or "",
                "category": c.get("category") or "",
                "source_domain": SOURCE_DOMAIN_ARXIV,
                "index_version": index_version,
                "chunks": [],
            }
        by_arxiv[aid]["chunks"].append(c)

    unified = list(by_arxiv.values())
    total_chunks = sum(len(d["chunks"]) for d in unified)
    print(
        f"[unify] arxiv: {len(unified)} papers, {total_chunks} chunks "
        f"(domain={SOURCE_DOMAIN_ARXIV})"
    )
    return unified


def build_unified_corpus(index_version: str = INDEX_VERSION) -> dict:
    rag_docs = load_rag_corpus(index_version)
    arxiv_docs = load_arxiv_corpus(index_version)

    all_docs = rag_docs + arxiv_docs
    total_chunks = sum(len(d.get("chunks", [])) for d in all_docs)

    corpus = {
        "index_version": index_version,
        "generated_at": __import__("datetime").datetime.now().isoformat(),
        "domain_scope": "RAG/LLM papers (rag-research) + optional arxiv general",
        "document_count": len(all_docs),
        "chunk_count": total_chunks,
        "documents": all_docs,
    }
    return corpus


def save_corpus(corpus: dict, out_path: Path = OUT_PATH) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=2, ensure_ascii=False)
    print(
        f"[unify] Wrote {corpus['chunk_count']} chunks / {corpus['document_count']} docs → {out_path}"
    )


def main() -> int:
    corpus = build_unified_corpus()
    save_corpus(corpus)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
