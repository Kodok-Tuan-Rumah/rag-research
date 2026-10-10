import math
import re

import numpy as np
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger("rag.confidence_engine")

class ConfidenceEngine:
    def __init__(self):
        self.reject_threshold = 0.35
        self.partial_threshold = 0.65

    def _normalize_scores(self, scores: List[float], score_type: str = "cosine") -> np.ndarray:
        arr = np.array(scores, dtype=float)
        if score_type == "cross_encoder":
            return 1 / (1 + np.exp(-arr))
        elif score_type == "rrf":
            if arr.max() > 0:
                return np.clip(arr / (arr.max() + 1e-9), 0.0, 1.0)
            return np.ones_like(arr) * 0.5
        else:
            if arr.max() == arr.min():
                return np.ones_like(arr) * 0.5
            return (arr - arr.min()) / (arr.max() - arr.min() + 1e-9)

    def _sigmoid(self, x: float) -> float:
        import math
        return 1.0 / (1.0 + math.exp(-x))

    def _calculate_entropy(self, scores: List[float]) -> float:
        if scores is None or len(scores) < 2:
            return 1.0
        arr = np.array(scores)
        exp_scores = np.exp(arr - np.max(arr))
        probs = exp_scores / exp_scores.sum()
        entropy = -np.sum(probs * np.log(probs + 1e-9))
        return float(entropy / np.log(len(probs)))

    def _select_chunk_score(self, chunk: Any) -> tuple[Optional[float], str]:
        """Eksplisit: pilih skor mana dari chunk dan jenis score-space-nya.

        Precedence (canonical):
          2. ``rerank_score``    → score-space cross-encoder (sigmoid-normalize)
          3. ``retrieval_score`` → score-space RRF (min-max / max normalize)
          4. ``score``           → legacy; diperlakukan sebagai RRF

        ``final_score`` memang menyimpan nilai yang sama dengan rerank_score
        (bila ada) atau retrieval_score — kami sengaja tidak membacanya langsung
        agar semantic "score-space" tetap dari sumber asalnya.
        """
        if isinstance(chunk, dict):
            get = chunk.get
        else:
            def get(key, default=None):
                return getattr(chunk, key, default)

        rerank = get("rerank_score")
        if rerank is not None:
            try:
                return float(rerank), "cross_encoder"
            except (TypeError, ValueError):
                return None, "cross_encoder"
        retrieval = get("retrieval_score")
        if retrieval is not None:
            try:
                return float(retrieval), "rrf"
            except (TypeError, ValueError):
                return None, "rrf"
        legacy = get("score")
        if legacy is not None:
            try:
                return float(legacy), "rrf"
            except (TypeError, ValueError):
                return None, "rrf"
        return None, "unknown"

    def _get_chunk_score(self, chunk: Any) -> float:
        score, _kind = self._select_chunk_score(chunk)
        return score

    def _source_agreement(self, chunks) -> float:
        """
        Fix 2: Logika dibalik dari versi sebelumnya.
        Agreement tinggi = top chunks berasal dari sedikit sumber (fokus).
        Agreement rendah = top chunks tersebar dari banyak sumber (tidak fokus).

        Sebelumnya: min(len(unique) / 3.0, 1.0) → makin banyak doc unik = makin tinggi (SALAH)
        Sekarang:   1.0 - min((unique - 1) / 4.0, 1.0) → 1 sumber unik = 1.0, 5+ sumber = 0.0
        """
        doc_ids = []
        for c in chunks[:5]:
            meta = c.get("metadata", {}) if isinstance(c, dict) else getattr(c, "metadata", {})
            doc_id = (
                (c.get("doc_id") if isinstance(c, dict) else getattr(c, "doc_id", None))
                or meta.get("doc_id")
                or meta.get("source")
                or meta.get("file_name")
            )
            if doc_id:
                doc_ids.append(doc_id)
        if not doc_ids:
            return 0.2
        unique_count = len(set(doc_ids))
        # 1 sumber unik → 1.0 (sangat fokus)
        # 2 sumber unik → 0.75
        # 3 sumber unik → 0.5
        # 5+ sumber unik → 0.0
        return max(0.0, 1.0 - (unique_count - 1) / 4.0)

    def _detect_patterns(self, top_norm: float, mean_top3_norm: float, gap: float) -> dict:
        return {
            "spurious_match": bool(top_norm > 0.85 and mean_top3_norm < 0.45),
            "unstable_retrieval": bool(gap < 0.05 and mean_top3_norm < 0.4),
        }

    def _decision_for_score(self, confidence_score: float, query_type: Optional[str] = None) -> str:
        reject, partial = self.reject_threshold, self.partial_threshold
        if query_type == "factual":
            reject += 0.05
            partial = min(1.0, partial + 0.05)
        elif query_type == "reasoning":
            reject = max(0.0, reject - 0.05)
            partial = max(reject, partial - 0.05)
        if confidence_score < reject:
            return "REJECT"
        if confidence_score < partial:
            return "PARTIAL_TRUST"
        return "GENERATE"

    def _context_overlap(self, answer: Optional[str], contexts: Optional[List[str]]) -> Optional[float]:
        if not answer or not contexts:
            return None
        answer_words = set(re.sub(r"[^\w\s]", "", answer.lower()).split())
        context_words = set(
            re.sub(r"[^\w\s]", "", " ".join(contexts).lower()).split()
        )
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to",
            "for", "of", "and", "or", "but", "it", "this", "that", "with", "as",
            "by", "from", "be", "has", "have",
        }
        answer_words -= stopwords
        if not answer_words:
            return None
        return round(len(answer_words & context_words) / len(answer_words), 4)

    def calculate_confidence(
        self,
        chunks: list[Any],
        answer: Optional[str] = None,
        contexts: Optional[List[str]] = None,
        query_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not chunks:
            return {
                "confidence_score": 0.0,
                "decision": "REJECT",
                "signals": {
                    "top_score": 0.0,
                    "gap": 0.0,
                    "mean_top3": 0.0,
                    "score_level": 0.0,
                    "entropy": 1.0,
                    "agreement": 0.0,
                    "spurious_match": False,
                    "unstable_retrieval": False,
                }
            }

        # Fix 3: skip chunk invalid, jangan reject semua batch
        # Sebelumnya: jika has_invalid_score → langsung return REJECT
        # Sekarang: log warning, tapi tetap lanjut dengan chunk yang valid
        scored_chunks = []
        invalid_count = 0
        for chunk in chunks:
            score = self._get_chunk_score(chunk)
            if isinstance(score, (int, float)) and np.isfinite(score):
                scored_chunks.append((chunk, float(score)))
            else:
                invalid_count += 1

        if invalid_count > 0:
            logger.warning(
                f"[CONFIDENCE] {invalid_count} chunk(s) dengan score invalid di-skip, "
                f"{len(scored_chunks)} chunk valid tersisa."
            )

        if not scored_chunks:
            return {
                "confidence_score": 0.0,
                "decision": "REJECT",
                "signals": {"reason": "no_valid_scores"}
            }

        valid_chunks = [chunk for chunk, _score in scored_chunks]
        raw_scores = [score for _chunk, score in scored_chunks]
        logger.info(f"[RAW_SCORES] {raw_scores[:10]}")

        selected_kinds = [
            self._select_chunk_score(chunk)[1]
            for chunk in valid_chunks
        ]

        score_type = "cross_encoder" if "cross_encoder" in selected_kinds else "rrf"
        normalized_scores = list(self._normalize_scores(raw_scores, score_type=score_type))
        top_score = raw_scores[0]
        mean_top3 = float(np.mean(raw_scores[:3])) if len(raw_scores) >= 3 else top_score
        top_score_norm = normalized_scores[0]

        mean_top3_norm = float(np.mean(normalized_scores[:3])) if len(normalized_scores) >= 3 else normalized_scores[0]
        if len(normalized_scores) > 1:
            gap = normalized_scores[0] - normalized_scores[1]
        else:
            gap = 0.0
        score_level = float(np.mean(normalized_scores[:3])) if len(normalized_scores) >= 3 else normalized_scores[0]
        entropy = self._calculate_entropy(normalized_scores)
        agreement = self._source_agreement(valid_chunks)

        base_confidence = (
            0.45 * top_score_norm +
            0.25 * mean_top3_norm +
            0.15 * agreement +
            0.15 * min(gap * 2, 1.0)
        )

        # Fix 1: turunkan penalty mean_top3_norm dari 0.40 → 0.15
        # Penalty 0.40 terlalu agresif — cross-encoder sering output negatif
        # sehingga setelah sigmoid normalisasi mean_top3_norm < 0.4 hampir selalu,
        # menyebabkan query valid pun di-REJECT
        penalty = 0.0
        if mean_top3_norm < 0.4:
            penalty += 0.15  # was: 0.40
        if gap < 0.05:
            penalty += 0.08
        if entropy > 0.8:
            penalty += 0.10
        if agreement < 0.2:
            penalty += 0.25

        confidence_score = max(0.0, base_confidence - penalty)

        # --- Context overlap + conflicting signal detection (Goals 5.3 / 12.2) ---
        context_overlap = self._context_overlap(answer, contexts)
        conflicts = []
        if context_overlap is not None:
            confidence_score = 0.6 * confidence_score + 0.4 * context_overlap
            if top_score_norm > 0.7 and context_overlap < 0.3:
                confidence_score -= 0.15
                conflicts.append("high_retrieval_low_overlap")
            if top_score_norm < 0.4 and context_overlap > 0.8:
                confidence_score -= 0.10
                conflicts.append("low_retrieval_high_overlap")

        confidence_score = max(0.0, min(1.0, confidence_score))

        logger.info(
            f"[CONFIDENCE_BREAKDOWN] top_score={top_score:.4f} | mean_top3={mean_top3:.4f} | "
            f"gap={gap:.4f} | entropy={entropy:.4f} | agreement={agreement:.4f} | score_level={score_level:.4f}"
        )
        logger.info(
            f"[DEBUG_CONFIDENCE] raw top_score={top_score} | mean_top3={mean_top3} | "
            f"confidence_score={confidence_score} | reject_threshold={self.reject_threshold}"
        )
        logger.info(
            f"[DEBUG_NORM] top_norm={top_score_norm:.4f} | mean_norm={mean_top3_norm:.4f}"
        )
        patterns = self._detect_patterns(top_score_norm, mean_top3_norm, gap)

        decision = self._decision_for_score(confidence_score, query_type=query_type)

        logger.info(f"[CONFIDENCE_RESULT] confidence_score={confidence_score:.4f} | decision={decision} | query_type={query_type} | conflicts={conflicts}")
        assert 0.0 <= confidence_score <= 1.0, f"Confidence out of range: {confidence_score}"

        return {
            "confidence_score": round(float(confidence_score), 4),
            "decision": decision,
            "signals": {
                "top_score": round(top_score, 4),
                "gap": round(gap, 4),
                "mean_top3": round(mean_top3, 4),
                "score_level": round(score_level, 4),
                "entropy": round(entropy, 4),
                "agreement": round(agreement, 4),
                "context_overlap": context_overlap,
                "conflicts": conflicts,
                "query_type": query_type,
                "spurious_match": patterns["spurious_match"],
                "unstable_retrieval": patterns["unstable_retrieval"],
                "score_type": score_type,
            }
        }
