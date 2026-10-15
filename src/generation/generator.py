from __future__ import annotations

import logging
import json
import os
import re
import time
from src.api.config import settings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from src.controller.confidence_engine import ConfidenceEngine
from src.controller.policy_engine import PolicyEngine
from groq import Groq, AuthenticationError, BadRequestError, GroqError

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logger = logging.getLogger("rag.generator")
load_dotenv()

DEFAULT_MODEL     = "openai/gpt-oss-20b"
DEFAULT_MAX_CHARS = settings.generation_max_chars
DEFAULT_TEMP      = settings.generation_temperature
DEFAULT_TOP_K     = 10

ABSTAIN_ANSWER   = "INSUFFICIENT_CONTEXT"

# PARTIAL_TRUST presentation rule. Both values used to be inlined at four
# sites; see apply_partial_trust_policy below.
PARTIAL_TRUST_CHUNK_LIMIT = 3
PARTIAL_TRUST_PREFIX = "[Low confidence — verify independently] "


def apply_partial_trust_policy(
    chunks: list,
    answer: str,
    retrieval_decision: Optional[str],
) -> tuple[list, str]:
    """Return (chunks, answer) after the PARTIAL_TRUST presentation rule.

    §4 requires a low-confidence answer to be visibly qualified and grounded on
    fewer chunks. This existed in four copies — agent.py truncated, agent.py
    prefixed, api/main.py did both, generator.py truncated again — and the copies
    had drifted:

    - the API prefixed unconditionally, so a generator abstain inside the
      PARTIAL_TRUST branch returned the literal string
      "[Low confidence — verify independently] INSUFFICIENT_CONTEXT" as the
      answer. check_retrieval_confidence() can still abstain after the confidence
      engine said PARTIAL_TRUST, so that path was reachable.
    - the Agent skipped the prefix when the answer was empty, so the two
      adapters disagreed on the same condition.

    An abstention is not a low-confidence answer. It must pass through untouched.
    """
    if retrieval_decision != "PARTIAL_TRUST":
        return chunks, answer

    limited = list(chunks[:PARTIAL_TRUST_CHUNK_LIMIT])
    if not answer or answer == ABSTAIN_ANSWER:
        return limited, answer
    return limited, f"{PARTIAL_TRUST_PREFIX}{answer}"

SYSTEM_PROMPT = (
    "You are a Technical Research Assistant. Answer the query using ONLY the provided CONTEXT.\n\n"
    "STRICT RULES:\n"
    "1. Grounding: Every claim must be directly supported by at least one [Source N].\n"
    "2. Bounded Reasoning: You may infer only if multiple sources support the logic. Do NOT introduce external facts.\n"
    "3. Format: Respond ONLY in valid JSON. No preamble, no markdown, no backticks.\n\n"
    "JSON SCHEMA:\n"
    "{\n"
    "  \"answer\": \"your technical answer here\",\n"
    "  \"status\": \"ANSWERED\" | \"INSUFFICIENT_CONTEXT\",\n"
    "  \"confidence_score\": 0.0 to 1.0,\n"
    "  \"supporting_sources\": [\"[Source 1]\", \"[Source 2]\"]\n"
    "}"
)

@dataclass
class RAGResponse:
    query:                    str
    answer:                   str
    retrieved_chunks:         list
    retrieval_method:         str
    context_used:             str
    latency_context_build_ms: float
    latency_generation_ms:    float
    model:                    str
    num_chunks_retrieved:     int
    num_chunks_used:          int
    status:                   str   = "UNKNOWN"
    confidence_score:         float = 0.0
    llm_confidence_score:      Optional[float] = None
    supporting_sources:       list  = None
    context_chunk_ids:        list  = None
    retrieval_confidence_score: Optional[float] = None
    retrieval_decision:        Optional[str] = None
    post_generation_confidence_score: Optional[float] = None
    post_generation_decision: Optional[str] = None
    # Why the pipeline abstained. PROJECT_GOALS.md §4 requires structured
    # failure classification; without this on the response, callers cannot tell a
    # provider failure from a model refusal without parsing logs.
    reason: Optional[str] = None

    def __post_init__(self):
        if self.supporting_sources is None:
            self.supporting_sources = []
        if self.context_chunk_ids is None:
            self.context_chunk_ids = []

# Helper: safe attribute access for RetrievalResult OR dict
def _get_attr(chunk, key: str, default=None):
    if isinstance(chunk, dict):
        return chunk.get(key, default)
    return getattr(chunk, key, default)

def _get_score(chunk) -> float:
    if isinstance(chunk, dict):
        for key in ("rerank_score", "retrieval_score", "final_score", "score"):
            value = chunk.get(key)
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
        return 0.0
    for key in ("rerank_score", "retrieval_score", "final_score", "score"):
        value = getattr(chunk, key, None)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return 0.0

# Context helpers
def filter_chunks_by_diversity(chunks: list, max_per_doc: int = 2) -> list:
    if not chunks:
        return chunks

    if len(chunks) <= 2:
        logger.info(
            f"[DIVERSITY_FILTER] Skipped (only {len(chunks)} chunks — below guard threshold)"
        )
        return chunks

    seen_docs: dict[str, int] = {}
    filtered = []

    for chunk in chunks:
        doc_id = _get_attr(chunk, "doc_id", "unknown")

        if doc_id not in seen_docs:
            seen_docs[doc_id] = 0

        if seen_docs[doc_id] < max_per_doc:
            filtered.append(chunk)
            seen_docs[doc_id] += 1

    logger.info(
        f"[DIVERSITY_FILTER] docs_count={len(seen_docs)} | "
        f"chunks_before={len(chunks)} | chunks_after={len(filtered)}"
    )
    return filtered

def build_context_with_ids(chunks: list, max_chars: int = DEFAULT_MAX_CHARS) -> tuple[str, list[str]]:
    context_parts = []
    included_ids = []
    total_chars = 0

    for i, chunk in enumerate(chunks):
        if isinstance(chunk, str):
            text     = chunk.strip()
            chunk_id = f"idx_{i}"
            doc_id   = "manual"
            section  = "General"
        else:
            text     = (_get_attr(chunk, "text", "") or "").strip()
            chunk_id = _get_attr(chunk, "chunk_id", f"c_{i}")
            doc_id   = _get_attr(chunk, "doc_id", "unknown")
            metadata = _get_attr(chunk, "metadata", {}) or {}
            section  = metadata.get("section", "General")

        header = f"[Source {i+1} | {doc_id} | {chunk_id} | Section: {section}]"
        block  = f"{header}\n{text}"

        if total_chars + len(block) > max_chars:
            logger.info(f"[CONTEXT_LIMIT] Reached max_chars={max_chars} at chunk {i+1}")
            break
        context_parts.append(block)
        if not isinstance(chunk, str):
            included_ids.append(_get_attr(chunk, "chunk_id", f"c_{i}"))
        total_chars += len(block)
    return "\n\n".join(context_parts), included_ids


def build_context(chunks: list, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    context, _included_ids = build_context_with_ids(chunks, max_chars=max_chars)
    return context

def filter_chunks_by_score(chunks: list, min_score: Optional[float] = None) -> list:
    if not chunks:
        return chunks
    has_rerank = any(
        (c.get("rerank_score") if isinstance(c, dict) else getattr(c, "rerank_score", None)) is not None
        for c in chunks
    )
    effective_min_score = -3.0 if has_rerank else (min_score if min_score is not None else settings.min_score_threshold)

    scores = [_get_score(c) for c in chunks]
    if scores[0] < effective_min_score:
        return []
    GAP_THRESHOLD = settings.score_gap_threshold
    selected = [chunks[0]]
    for i in range(1, len(chunks)):
        gap = scores[i - 1] - scores[i]
        if scores[i] < effective_min_score or gap > GAP_THRESHOLD:
            break
        selected.append(chunks[i])
    return selected

# LLM output parser
def _parse_llm_output(raw: str) -> tuple[str, str, float, list]:
    try:
        parsed = json.loads(raw)
        answer   = parsed.get("answer", "").strip()
        status   = parsed.get("status", "UNKNOWN")
        conf     = float(parsed.get("confidence_score", 0.0))
        sources  = parsed.get("supporting_sources", [])
        if not answer:
            return "", "INSUFFICIENT_CONTEXT", conf, sources
        return answer, status, conf, sources
    
    except Exception as e:
        logger.warning(f"[PARSE_ERROR] LLM output is not valid JSON: {e} | raw='{raw[:200]}'")
        return raw, "INSUFFICIENT_CONTEXT", 0.0, []

def should_abort_before_generation(confidence_score: float, decision: str, has_chunks: bool, min_confidence: float = 0.05) -> bool:
    if not has_chunks:
        return True
    if decision == "REJECT" and confidence_score <= min_confidence:
        return True
    return False


def sanity_check_answer(answer: str, chunks: list, verbatim_threshold: float = 0.95) -> tuple[bool, str]:
    if not answer or not chunks:
        return True, "ok"
    answer_clean = answer.strip().lower()
    answer_words = set(answer_clean.split())

    if len(answer_words) < 20:
        return True, "ok"

    for i, chunk in enumerate(chunks):
        chunk_text = (_get_attr(chunk, "text", "") or "").strip().lower()
        if not chunk_text:
            continue
        if len(answer_clean) > 200 and answer_clean in chunk_text:
            return False, f"reference_dump: verbatim substring match di chunk {i+1}"
        overlap = len(answer_words & set(chunk_text.split())) / len(answer_words)
        is_verbatim_copy = answer_clean in chunk_text
        chunk_coverage = len(answer_words & set(chunk_text.split())) / max(len(set(chunk_text.split())), 1)
        is_near_copy = overlap >= verbatim_threshold and chunk_coverage > 0.5
        if is_verbatim_copy or is_near_copy:
            return False, f"reference_dump: word_overlap={overlap:.2f} di chunk {i+1}"
    return True, "ok"


def check_answer_evidence(
    answer: str,
    chunks: list,
    min_sentence_coverage: float = 0.35,
) -> tuple[bool, str]:
    """Reject generated statements with too little lexical support in any chunk.

    Each sentence is compared with each retrieved passage independently. This
    avoids treating a collection of loosely related passages as if one passage
    supported the whole answer. It is a conservative lexical guard, not an NLI
    entailment model; callers should keep the threshold configurable.
    """
    if not answer or not chunks:
        return False, "insufficient_evidence: answer or evidence is empty"

    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "do", "does", "did", "has", "have", "had", "what", "which", "who",
        "when", "where", "why", "how", "and", "or", "but", "if", "then",
        "this", "that", "these", "those", "it", "its", "they", "them", "their",
        "with", "from", "for", "to", "of", "in", "on", "at", "by", "as",
        "can", "could", "would", "should", "will", "may", "might", "must",
        "also", "than", "into", "using", "use", "used", "based", "according",
    }

    def content_terms(text: str) -> set[str]:
        return {
            token for token in re.findall(r"[a-z0-9]+", text.lower())
            if len(token) > 1 and token not in stopwords
        }

    evidence_terms = []
    for chunk in chunks:
        text = _get_attr(chunk, "text", "") or ""
        terms = content_terms(text)
        if terms:
            evidence_terms.append(terms)
    if not evidence_terms:
        return False, "insufficient_evidence: retrieved chunks contain no usable text"

    sentences = [part.strip() for part in re.split(r"(?<=[.!?;])\s+", answer) if part.strip()]
    if not sentences:
        sentences = [answer.strip()]

    weak_sentences = []
    for sentence in sentences:
        terms = content_terms(sentence)
        if not terms:
            weak_sentences.append((sentence, 0.0))
            continue
        best_coverage = max(len(terms & passage) / len(terms) for passage in evidence_terms)
        if best_coverage < min_sentence_coverage:
            weak_sentences.append((sentence, best_coverage))

    if weak_sentences:
        examples = "; ".join(
            f"'{sentence[:80]}' ({coverage:.2f})"
            for sentence, coverage in weak_sentences[:2]
        )
        return False, f"insufficient_evidence: answer statement(s) lack chunk support: {examples}"
    return True, "ok"

# Retrieval Hit Guard
def _make_abstain_response(
    query: str,
    chunks: list,
    model: str,
    reason: str,
    context_used: str = "",
    context_chunk_ids: Optional[list[str]] = None,
    llm_confidence_score: Optional[float] = None,
    retrieval_confidence_score: Optional[float] = None,
    retrieval_decision: Optional[str] = None,
    post_generation_confidence_score: Optional[float] = None,
    post_generation_decision: Optional[str] = None,
) -> RAGResponse:
    logger.warning(f"[ABSTAIN] {reason} | query='{query[:60]}'")
    return RAGResponse(
        query=query,
        answer="INSUFFICIENT_CONTEXT",
        retrieved_chunks=chunks,
        retrieval_method="abstained",
        context_used=context_used,
        latency_context_build_ms=0.0,
        latency_generation_ms=0.0,
        model=model,
        num_chunks_retrieved=len(chunks),
        num_chunks_used=0,
        status="INSUFFICIENT_CONTEXT",
        confidence_score=0.0,
        llm_confidence_score=llm_confidence_score,
        supporting_sources=[],
        context_chunk_ids=context_chunk_ids or [],
        retrieval_confidence_score=retrieval_confidence_score,
        retrieval_decision=retrieval_decision,
        post_generation_confidence_score=post_generation_confidence_score,
        post_generation_decision=post_generation_decision,
        reason=reason,
    )


# Groq reports this as HTTP 400 with code `json_validate_failed` when the model
# emits output that does not satisfy response_format={"type": "json_object"}.
# openai/gpt-oss-20b is a reasoning model, and at production context size it does
# this often enough to be visible (measured 6/10 with a production-sized context,
# 0/10 with a short one). It is a provider-side output-format rejection, not a
# decision the confidence engine made, so it must degrade to a structured
# abstain rather than propagate as an operational error.
_JSON_VALIDATION_MARKERS = ("json_validate_failed", "validate json", "generate json")


def _is_json_validation_failure(exc: BadRequestError) -> bool:
    code = ""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        code = str((body.get("error") or {}).get("code") or "")
    haystack = f"{code} {exc}".lower()
    return any(marker in haystack for marker in _JSON_VALIDATION_MARKERS)

def check_retrieval_confidence(
    chunks: list,
    source_chunk_id: Optional[str],
    top_k_scores: list[float],
    min_top1_score: float = 0.0,
) -> tuple[bool, str]:
    if not chunks:
        return False, "chunks kosong — tidak ada kandidat untuk generation"
    if source_chunk_id:
        retrieved_ids = [_get_attr(c, "chunk_id", "") for c in chunks]
        is_hit = any(
            rid == source_chunk_id or rid.startswith(source_chunk_id + "_sub")
            for rid in retrieved_ids
        )
        if not is_hit:
            return False, (
                f"retrieval_miss: source_chunk_id='{source_chunk_id}' "
                f"tidak ditemukan di {len(retrieved_ids)} retrieved chunks — "
                f"generation akan hallucinate, pipeline abstain"
            )

    if top_k_scores and top_k_scores[0] <= min_top1_score:
        return False, (
            f"top1_score={top_k_scores[0]:.4f} ≤ threshold={min_top1_score} — "
            f"retrieval tidak yakin, pipeline abstain"
        )
    return True, "ok"

# Generator
class GroqGenerator:
    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_MODEL):
        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise ValueError("GROQ_API_KEY tidak ditemukan.")
        self.client = Groq(api_key=key)
        self.model  = model
        logger.info(f"[GroqGenerator] Ready. Model: {model}")
        self._confidence = ConfidenceEngine()
        
    def generate(
        self,
        query:           str,
        chunks:          list,
        max_tokens:      Optional[int] = None,
        temperature:     Optional[float] = None,
        source_chunk_id: Optional[str] = None,
        min_top1_score:  Optional[float] = None,
        query_type:      Optional[str]  = None,
        retrieval_decision: Optional[str] = None,
        retrieval_confidence: Optional[float] = None,
    ) -> RAGResponse:
        t0_total = time.time()
        query_type = query_type or PolicyEngine().decide_query_type(query)
        temperature = settings.generation_temperature if temperature is None else temperature
        # This defaulted to -3.0 while check_retrieval_confidence() defaults to
        # 0.0 and every explicit caller passes 0.0. The API never passes the
        # argument, so it alone ran the permissive gate and generated answers from
        # chunks the CLI and Agent rejected. Now sourced from settings so all
        # paths agree by default.
        min_top1_score = (
            settings.retrieval_min_top1_score if min_top1_score is None else min_top1_score
        )
        num_chunks_retrieved = len(chunks)

        if retrieval_decision not in (None, "REJECT", "PARTIAL_TRUST", "GENERATE"):
            raise ValueError(f"Unknown retrieval decision: {retrieval_decision}")
        if retrieval_decision == "REJECT":
            return _make_abstain_response(
                query=query,
                chunks=chunks,
                model=self.model,
                reason="retrieval decision REJECT",
                retrieval_confidence_score=retrieval_confidence,
                retrieval_decision=retrieval_decision,
            )
        if retrieval_decision == "PARTIAL_TRUST":
            chunks = chunks[:PARTIAL_TRUST_CHUNK_LIMIT]

        top_k_scores = [_get_score(c) for c in chunks[:5]]

        should_proceed, reason = check_retrieval_confidence(
            chunks=chunks,
            source_chunk_id=source_chunk_id,
            top_k_scores=top_k_scores,
            min_top1_score=min_top1_score,
        )

        if not should_proceed:
            return _make_abstain_response(
                query=query,
                chunks=chunks,
                model=self.model,
                reason=reason,
                retrieval_confidence_score=retrieval_confidence,
                retrieval_decision=retrieval_decision,
            )

        if not chunks:
            return _make_abstain_response(
                query=query, chunks=[], model=self.model,
                reason="chunks kosong setelah guard (defensive)",
                retrieval_confidence_score=retrieval_confidence,
                retrieval_decision=retrieval_decision,
            )
        score_filtered = filter_chunks_by_score(chunks)
        filtered_chunks = filter_chunks_by_diversity(score_filtered, max_per_doc=2)
        num_chunks_used = len(filtered_chunks)
        logger.info(
            f"[RETRIEVAL_QUALITY] query='{query[:50]}...' | "
            f"num_retrieved={num_chunks_retrieved} | num_after_diversity={num_chunks_used} | "
            f"top_5_scores={[round(s, 4) for s in top_k_scores]}"
        )

        context, context_chunk_ids = build_context_with_ids(
            filtered_chunks, max_chars=DEFAULT_MAX_CHARS
        )
        t1 = time.time()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"CONTEXT:\n{context}\n\nQUESTION:\n{query}"}
        ]
        effective_max_tokens = max_tokens if max_tokens is not None else settings.generation_max_tokens

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=effective_max_tokens,
                response_format={"type": "json_object"},
            )
        except AuthenticationError as exc:
            raise ValueError(
                "Invalid GROQ_API_KEY atau token tidak valid. "
                "Set environment variable GROQ_API_KEY dengan API key yang benar."
            ) from exc
        except BadRequestError as exc:
            if _is_json_validation_failure(exc):
                logger.warning(
                    "[GENERATION_JSON_REJECTED] provider rejected model output as invalid "
                    "JSON (code=json_validate_failed); treating as abstain | model=%s",
                    self.model,
                )
                return _make_abstain_response(
                    query=query,
                    chunks=chunks,
                    model=self.model,
                    reason=(
                        "provider rejected model output: response_format=json_object "
                        "validation failed"
                    ),
                    context_used=context,
                    context_chunk_ids=context_chunk_ids,
                    retrieval_confidence_score=retrieval_confidence,
                    retrieval_decision=retrieval_decision,
                )
            raise RuntimeError(
                f"Groq generation failed: {exc}. "
                "Periksa koneksi API dan kunci Anda."
            ) from exc
        except GroqError as exc:
            raise RuntimeError(
                f"Groq generation failed: {exc}. "
                "Periksa koneksi API dan kunci Anda."
            ) from exc

        raw_content = response.choices[0].message.content or ""
        if not raw_content.strip():
            logger.warning(
                "[GENERATION_EMPTY] model returned empty output | model=%s",
                self.model,
            )
            return _make_abstain_response(
                query=query,
                chunks=chunks,
                model=self.model,
                reason="model returned empty output",
                context_used=context,
                context_chunk_ids=context_chunk_ids,
                retrieval_confidence_score=retrieval_confidence,
                retrieval_decision=retrieval_decision,
            )

        raw_answer = raw_content.strip()
        t2 = time.time()


        answer, status, confidence, sources = _parse_llm_output(raw_answer)
        is_ok, sanity_reason = sanity_check_answer(answer, filtered_chunks)
        if not is_ok:
            logger.warning(f"[SANITY_CHECK] REJECTED | reason={sanity_reason}")
            return _make_abstain_response(
                query=query, chunks=chunks, model=self.model, reason=sanity_reason,
                context_used=context, context_chunk_ids=context_chunk_ids,
                llm_confidence_score=confidence,
                retrieval_confidence_score=retrieval_confidence,
                retrieval_decision=retrieval_decision,
            )
        if status == "INSUFFICIENT_CONTEXT":
            answer = "I'm sorry, I couldn't find the answer in the provided context."
        elif status == "ANSWERED":
            evidence_ok, evidence_reason = check_answer_evidence(answer, filtered_chunks)
            if not evidence_ok:
                return _make_abstain_response(
                    query=query,
                    chunks=chunks,
                    model=self.model,
                    reason=evidence_reason,
                    context_used=context,
                    context_chunk_ids=context_chunk_ids,
                    llm_confidence_score=confidence,
                    retrieval_confidence_score=retrieval_confidence,
                    retrieval_decision=retrieval_decision,
                )

        latency_context_ms = round((t1 - t0_total) * 1000, 2)
        latency_gen_ms     = round((t2 - t1) * 1000, 2)
        latency_total_ms   = round((t2 - t0_total) * 1000, 2)

        logger.info(
            f"[GENERATION_LATENCY] query='{query[:50]}...' | "
            f"context_build_ms={latency_context_ms} | "
            f"generation_ms={latency_gen_ms} | "
            f"total_ms={latency_total_ms} | "
            f"status={status} | confidence={confidence:.2f}"
        )

        retrieval_method = _get_attr(chunks[0], "retrieval_method", "hybrid_rrf_baseline")
        confidence_result = self._confidence.calculate_confidence(
            filtered_chunks,
            answer=answer,
            contexts=[(_get_attr(c, "text", "") or "") for c in filtered_chunks],
            query_type=query_type,
            query=query,
        )
        final_confidence = float(confidence_result.get("confidence_score", 0.0))
        final_confidence = max(0.0, min(1.0, final_confidence))
        confidence_result["confidence_score"] = final_confidence

        if confidence_result.get("signals", {}).get("broad_topic_with_mixed_sources"):
            return _make_abstain_response(
                query=query,
                chunks=chunks,
                model=self.model,
                reason=(
                    "insufficient_evidence: broad topic question retrieved mixed-source "
                    "context without a focused passage"
                ),
                context_used=context,
                context_chunk_ids=context_chunk_ids,
                llm_confidence_score=confidence,
                retrieval_confidence_score=retrieval_confidence,
                retrieval_decision=retrieval_decision,
                post_generation_confidence_score=final_confidence,
                post_generation_decision=confidence_result.get("decision"),
            )

        if should_abort_before_generation(
            confidence_score=final_confidence,
            decision=confidence_result.get("decision", "GENERATE"),
            has_chunks=bool(filtered_chunks),
        ):
            return _make_abstain_response(
                query=query,
                chunks=chunks,
                model=self.model,
                reason=f"confidence_engine REJECT | score={final_confidence:.3f}",
                context_used=context,
                context_chunk_ids=context_chunk_ids,
                llm_confidence_score=confidence,
                retrieval_confidence_score=retrieval_confidence,
                retrieval_decision=retrieval_decision,
                post_generation_confidence_score=final_confidence,
                post_generation_decision=confidence_result.get("decision"),
            )
        
        logger.info(
            f"[CONFIDENCE_CALC] score={final_confidence:.2f} | "
            f"decision={confidence_result['decision']}"
            f"signals={confidence_result['signals']}"
        )
        return RAGResponse(
            query=query,
            answer=answer,
            retrieved_chunks=filtered_chunks,
            retrieval_method=retrieval_method,
            context_used=context,
            latency_context_build_ms=latency_context_ms,
            latency_generation_ms=latency_gen_ms,
            model=self.model,
            num_chunks_retrieved=num_chunks_retrieved,
            num_chunks_used=num_chunks_used,
            status=status,
            confidence_score=final_confidence,  
            llm_confidence_score=confidence,
            supporting_sources=sources,
            context_chunk_ids=context_chunk_ids,
            retrieval_confidence_score=retrieval_confidence,
            retrieval_decision=retrieval_decision,
            post_generation_confidence_score=final_confidence,
            post_generation_decision=confidence_result.get("decision"),
        )


# Module-level convenience
def generate(query: str, chunks: list, **kwargs) -> RAGResponse:
    gen = GroqGenerator(model=kwargs.get("model_name", DEFAULT_MODEL))
    return gen.generate(query, chunks)


# Self-contained sanity-check tests
def _run_sanity_check_answer_tests() -> bool:
    long_chunk = (
        "Attention mechanisms let a model compare query representations with key representations, "
        "produce normalized weights, and use those weights to combine value representations into a "
        "context vector. This helps the model focus on relevant tokens when generating each output "
        "rather than compressing an entire sequence into one fixed representation. The mechanism is "
        "especially useful in encoder decoder and transformer architectures because it creates a "
        "direct path between distant positions in the input sequence."
    )
    high_overlap_answer = (
        "model Attention mechanisms let a model compare query representations with key representations, "
        "produce normalized weights, and use those weights to combine value representations into a "
        "context vector. This helps the model focus on relevant tokens when generating each output"
    )
    normal_answer = (
        "An attention mechanism scores which input tokens matter for the current output step and "
        "uses those scores to build a context representation from the source sequence."
    )
    cases = [
        ("short answer <20 words", "Attention weights identify relevant input tokens.", [{"text": long_chunk}], True),
        ("verbatim chunk copy >200 chars", long_chunk, [{"text": long_chunk}], False),
        ("word overlap >=0.95", high_overlap_answer, [{"text": long_chunk}], False),
        ("normal grounded answer", normal_answer, [{"text": long_chunk}], True),
        ("empty answer", "", [{"text": long_chunk}], True),
    ]

    print("\n" + "=" * 80)
    print("[SANITY_CHECK_ANSWER TESTS]")
    print("=" * 80)
    all_passed = True
    for name, answer, chunks, expected in cases:
        actual, reason = sanity_check_answer(answer, chunks)
        passed = actual is expected
        all_passed = all_passed and passed
        status = "PASS" if passed else "FAIL"
        print(f"{status} | {name} | expected={expected} actual={actual} | reason={reason}")
    return all_passed

# Optional smoke test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sanity_ok = _run_sanity_check_answer_tests()

    if os.getenv("RUN_GENERATOR_SMOKE") != "1":
        raise SystemExit(0 if sanity_ok else 1)

    from src.retrieval.qdrant_store import QdrantVectorStore
    from src.retrieval.hybrid_retriever import MasterHybridRetriever

    store = QdrantVectorStore()
    try:
        retriever = MasterHybridRetriever(vector_store=store)
        gen = GroqGenerator()

        queries = [
            "What is Retrieval-Augmented Generation?",
            "How does hybrid search improve RAG performance?",
            "What are the limitations of large language models?",
        ]

        print("\n" + "=" * 80)
        print("[GENERATOR TEST] Object-safe + JSON-parsed generation")
        print("=" * 80)

        for q in queries:
            chunks = retriever.search(q, top_k=10)
            resp = gen.generate(q, chunks)
            print(f"\nQ: {q}")
            print(f"Chunks Retrieved: {resp.num_chunks_retrieved} | Used: {resp.num_chunks_used}")
            print(f"Status: {resp.status} | Confidence: {resp.confidence_score:.2f}")
            print(f"A: {resp.answer[:200]}...")
            print(f"Sources: {resp.supporting_sources}")
            print(f"Context Build: {resp.latency_context_build_ms}ms | Generation: {resp.latency_generation_ms}ms")
            print("-" * 80)
    finally:
        store.close()
