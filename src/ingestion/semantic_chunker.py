from __future__ import annotations

import json
import re
import time
import numpy as np
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

from src.ingestion.document_loader import extract_sections
from src.api.config import settings
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- IMPORT RUST ENGINE WITH PYTHON FALLBACK ---
try:
    import semantic_chunker_rust
except ImportError:
    class SemanticChunkerRustMock:
        @staticmethod
        def split_sentences_rs(text: str) -> List[str]:
            raw = re.split(r'(?<=[.?!])\s+', text)
            return [s.strip() for s in raw if s.strip()]

        @staticmethod
        def find_breakpoints_rs(similarities, threshold: float) -> List[int]:
            breakpoints = []
            # similarities is a 1D or 2D array of adjacent similarities
            sim_list = np.diag(similarities, k=1) if isinstance(similarities, np.ndarray) and similarities.ndim == 2 else similarities
            for i, sim in enumerate(sim_list):
                if sim < threshold:
                    breakpoints.append(i + 1)
            return breakpoints

        @staticmethod
        def assemble_chunks_rs(sentences: List[str], breakpoints: List[int], overlap: int) -> List[List[str]]:
            if not sentences:
                return []
            chunks = []
            start = 0
            for bp in breakpoints:
                if bp > start and bp <= len(sentences):
                    chunks.append(sentences[start:bp])
                    start = max(0, bp - overlap)
            if start < len(sentences):
                chunks.append(sentences[start:])
            return chunks

    semantic_chunker_rust = SemanticChunkerRustMock()

# Config
# Single source of truth: src/api/config.py. This constant previously duplicated
# the model name as a literal, so switching embedders required three edits and
# a missed one would index the corpus at a different dimension than the query
# path reads — the same failure class as the stale vector_dim == 768 assertion.
EMBED_MODEL      = settings.embed_model
BREAKPOINT_PCT   = 85
MAX_CHUNK_TOKENS = 512
MIN_CHUNK_TOKENS = 50
OVERLAP_SENTENCES = 1


# ---------------------------------------------------------------------------
# Noise filter — dipakai di chunking (ingestion) layer
# ---------------------------------------------------------------------------

def _is_noise_chunk(text: str) -> bool:
    """
    Return True jika chunk tidak layak di-index.
    Aturan noise di-sentralisasi di src/retrieval/chunk_filters.py
    (sebelumnya diduplikasi di sini & qdrant_store.py).
    """
    from src.retrieval.chunk_filters import is_noise_chunk_text
    return is_noise_chunk_text(text) 

# Dataclass 
@dataclass
class SemanticChunk:
    chunk_id:    str
    doc_id:      str
    text:        str
    token_count: int
    sentence_count: int
    start_sentence_idx: int
    end_sentence_idx:   int
    metadata: dict

# Utilities 
def _estimate_tokens(text: str) -> int:
    return len(text) // 4

def _compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    sims = []
    for i in range(len(embeddings) - 1):
        sim = cosine_similarity(
            embeddings[i].reshape(1, -1),
            embeddings[i + 1].reshape(1, -1)
        )[0][0]
        sims.append(float(sim)) 
    return sims

# Main Chunker 
class SemanticChunker:
    def __init__(
        self,
        model_name: str = EMBED_MODEL,
        breakpoint_percentile: int = BREAKPOINT_PCT,
        max_chunk_tokens: int = MAX_CHUNK_TOKENS,
        min_chunk_tokens: int = MIN_CHUNK_TOKENS,
        overlap_sentences: int = OVERLAP_SENTENCES,
    ):
        print(f"[SemanticChunker] Engine: RUST + PYTHON (Hybrid)")
        print(f"  - Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.breakpoint_percentile = breakpoint_percentile
        self.max_chunk_tokens = max_chunk_tokens
        self.min_chunk_tokens = min_chunk_tokens
        self.overlap_sentences = overlap_sentences

    def chunk_document(self, doc: dict) -> List[SemanticChunk]:
        doc_id = doc.get("doc_id", "unknown")
        full_text = doc.get("text", "")
        
        # 1. Gunakan fungsi extractor kita
        # Pastikan fungsi ini sudah di-import di atas
        sections = extract_sections(full_text)
        
        final_chunks = []
        chunk_idx = 0

        for sec in sections:
            sec_name = sec["section_name"]
            sec_text = sec["text"]
            
            # Split per section supaya konteks bab tetap terjaga
            sentences = semantic_chunker_rust.split_sentences_rs(sec_text)
            
            if len(sentences) < 1: continue
            
            # Jika cuma 1 kalimat, langsung buat chunk
            if len(sentences) < 2:
                meta = {**doc.get("metadata", {}), "section": sec_name}
                chunk = self._make_chunk(doc_id, sentences, 0, 0, chunk_idx, meta)
                # Filter noise — single-sentence chunk sangat rentan jadi judul/author
                if _is_noise_chunk(chunk.text):
                    print(f"  [NOISE_SKIP] single-sentence: '{chunk.text[:80]}'")
                    continue
                final_chunks.append(chunk)
                chunk_idx += 1
                continue

            # Logic Semantic (Rust + Python) tetap sama
            embeddings = self.model.encode(sentences, batch_size=64, show_progress_bar=False)
            similarities = _compute_similarity_matrix(embeddings)
            
            threshold = np.percentile(similarities, 100 - self.breakpoint_percentile)
            breakpoints = semantic_chunker_rust.find_breakpoints_rs(similarities, float(threshold))
            
            chunk_groups = semantic_chunker_rust.assemble_chunks_rs(
                sentences, breakpoints, self.overlap_sentences
            )
            chunk_groups = self._merge_short_groups(chunk_groups)

            for group_sentences in chunk_groups:
                sub_chunks = self._enforce_max_tokens(group_sentences)
                for sub_sentences in sub_chunks:
                    # SIMPAN DATA SECTION DI SINI
                    meta = {**doc.get("metadata", {}), "section": sec_name}
                    chunk = self._make_chunk(doc_id, sub_sentences, 0, 0, chunk_idx, meta)
                    
                    # Keep every non-noise semantic fragment. Dropping short
                    # tails silently discarded text between adjacent chunks.
                    if _is_noise_chunk(chunk.text):
                        print(f"  [NOISE_SKIP] '{chunk.text[:80]}'")
                        continue
                    final_chunks.append(chunk)
                    chunk_idx += 1
                        
        return final_chunks

    def _enforce_max_tokens(self, sentences: List[str]) -> List[List[str]]:
        result = []
        current = []
        current_tokens = 0

        for sent in sentences:
            sent_tokens = _estimate_tokens(sent)
            if current_tokens + sent_tokens > self.max_chunk_tokens and current:
                result.append(current)
                overlap = current[-self.overlap_sentences:] if self.overlap_sentences else []
                overlap_tokens = sum(_estimate_tokens(item) for item in overlap)
                # Keep overlap only when it leaves room for the next sentence.
                if overlap_tokens + sent_tokens > self.max_chunk_tokens:
                    overlap = []
                    overlap_tokens = 0
                current = [*overlap, sent]
                current_tokens = overlap_tokens + sent_tokens
            else:
                current.append(sent)
                current_tokens += sent_tokens
        if current:
            result.append(current)
        return result

    def _merge_short_groups(self, groups: List[List[str]]) -> List[List[str]]:
        """Keep short semantic fragments attached to a neighboring passage."""
        merged = [list(group) for group in groups if group]
        index = 0
        while index < len(merged) and len(merged) > 1:
            group_tokens = _estimate_tokens(" ".join(merged[index]))
            if group_tokens >= self.min_chunk_tokens:
                index += 1
                continue

            if index > 0:
                candidate = merged[index - 1] + merged[index]
                if _estimate_tokens(" ".join(candidate)) <= self.max_chunk_tokens:
                    merged[index - 1] = candidate
                    del merged[index]
                    index = max(index - 1, 0)
                    continue

            if index + 1 < len(merged):
                candidate = merged[index] + merged[index + 1]
                if _estimate_tokens(" ".join(candidate)) <= self.max_chunk_tokens:
                    merged[index] = candidate
                    del merged[index + 1]
                    continue
            # Keep a short group rather than silently dropping its text.
            index += 1
        return merged

    def _make_chunk(
        self, doc_id: str, sentences: List[str],
        start_idx: int, end_idx: int, chunk_idx: int, metadata: dict
    ) -> SemanticChunk:
        text = " ".join(sentences)
        return SemanticChunk(
            chunk_id=f"{doc_id}_rs_{chunk_idx:04d}", 
            doc_id=doc_id,
            text=text,
            token_count=_estimate_tokens(text),
            sentence_count=len(sentences),
            start_sentence_idx=start_idx,
            end_sentence_idx=end_idx,
            metadata={**metadata, "engine": "rust_semantic", "chunk_idx": chunk_idx},
        )

    def chunk_all_documents(self, documents: List[dict]) -> List[SemanticChunk]:
        all_chunks = []
        t0 = time.time()
        total_noise_skipped = 0

        for doc in documents:
            before = 0  # chunk_idx sebelum dokumen ini
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)

        elapsed = time.time() - t0
        print(f"\n[SemanticChunker] Total: {len(all_chunks)} chunks dari {len(documents)} dokumen ({elapsed:.2f}s)")
        return all_chunks

# Save 
def save_chunks(chunks: List[SemanticChunk], output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    data = [asdict(c) for c in chunks]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[Saved] {len(chunks)} chunks → {output_path}")

# CLI 
if __name__ == "__main__":
    import argparse
    docs_path = Path("data/processed/documents.json")
    if not docs_path.exists():
        print(f"[ERROR] {docs_path} tidak ditemukan.")
        raise SystemExit(1)

    parser = argparse.ArgumentParser(description="Build a versioned semantic chunk corpus.")
    parser.add_argument(
        "--output",
        default="data/processed/chunks_semantic_v3.json",
        help="Output path. Defaults to a new v3 file and never overwrites the frozen v2 corpus.",
    )
    args = parser.parse_args()

    with open(docs_path, "r", encoding="utf-8") as f:
        documents = json.load(f)

    chunker = SemanticChunker()
    chunks = chunker.chunk_all_documents(documents)
    save_chunks(chunks, args.output)
