"""R1: taksonomi provider failure eksplisit.

Provider error (429, koneksi, auth, JSON invalid, GroqError lain) tidak boleh
dilaporkan sebagai `insufficient_context` — abstention sah kontekstual harus
tetap terpisah dari kegagalan operasional.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from groq import AuthenticationError, APIConnectionError, BadRequestError, GroqError, RateLimitError

from src.generation.generator import (
    FAILURE_EMPTY_OUTPUT,
    FAILURE_INSUFFICIENT_CONTEXT,
    FAILURE_PROVIDER_ERROR,
    FAILURE_SANITY_CHECK,
    FAILURE_UNSUPPORTED_CLAIMS,
    GroqGenerator,
    PROVIDER_AUTH,
    PROVIDER_CONNECTION,
    PROVIDER_INVALID_JSON,
    PROVIDER_RATE_LIMIT,
    PROVIDER_UNKNOWN,
)

CHUNKS = [
    {
        "chunk_id": "c1",
        "doc_id": "d1",
        "text": "Hybrid search combines dense retrieval with BM25 and fuses the lists.",
        "retrieval_score": 0.8,
        "metadata": {},
    }
]


def json_answer(answer: str, status: str = "ANSWERED") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=json.dumps({
                        "answer": answer,
                        "status": status,
                        "confidence_score": 0.9,
                        "supporting_sources": ["[Source 1]"],
                    })
                )
            )
        ]
    )


def _httpx_request() -> "object":
    import httpx
    return httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")


def scripted_generator(script):
    def _create(**kwargs):
        item = script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    gen = GroqGenerator.__new__(GroqGenerator)
    gen.model = "openai/gpt-oss-20b"
    gen.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=_create))
    )
    gen._confidence = SimpleNamespace(
        calculate_confidence=lambda *args, **kwargs: {
            "confidence_score": 0.85,
            "decision": "GENERATE",
            "signals": {},
        }
    )
    return gen


def generate_with_provider_error(exc: BaseException):
    return scripted_generator([exc, exc]).generate("What is hybrid search?", CHUNKS)


# --- mapping ------------------------------------------------------------------

def test_rate_limit_after_retry_maps_to_provider_error():
    request = _httpx_request()

    class Wait429(RateLimitError):
        pass

    def build():
        import httpx
        return Wait429(
            "Error code: 429 - rate_limit_exceeded",
            response=httpx.Response(429, request=request, headers={"retry-after": "0.01"}),
            body={"error": {"code": "rate_limit_exceeded"}},
        )

    result = generate_with_provider_error(build())
    assert result.failure_type == FAILURE_PROVIDER_ERROR
    assert result.provider_error_kind == PROVIDER_RATE_LIMIT


def test_connection_after_retry_maps_to_provider_error():
    result = generate_with_provider_error(APIConnectionError(request=_httpx_request()))
    assert result.failure_type == FAILURE_PROVIDER_ERROR
    assert result.provider_error_kind == PROVIDER_CONNECTION


def test_auth_error_is_structured_provider_error():
    def build():
        import httpx
        return AuthenticationError(
            "Error code: 401 - invalid api key",
            response=httpx.Response(401, request=_httpx_request()),
            body={"error": {"message": "invalid api key"}},
        )

    result = generate_with_provider_error(build())
    assert result.status == "INSUFFICIENT_CONTEXT"
    assert result.failure_type == FAILURE_PROVIDER_ERROR
    assert result.provider_error_kind == PROVIDER_AUTH


def test_json_fallback_failure_maps_to_invalid_json_provider_error():
    def build():
        import httpx
        return BadRequestError(
            "Error code: 400 - json_validate_failed",
            response=httpx.Response(400, request=_httpx_request()),
            body={"error": {"code": "json_validate_failed"}},
        )

    result = scripted_generator([build(), build()]).generate("What is hybrid search?", CHUNKS)
    assert result.failure_type == FAILURE_PROVIDER_ERROR
    assert result.provider_error_kind == PROVIDER_INVALID_JSON


def test_generic_groq_during_fallback_maps_to_provider_kind():
    class Boom(GroqError):
        pass

    result = scripted_generator([Boom("boom"), Boom("boom")]).generate("What is hybrid search?", CHUNKS)
    assert result.failure_type == FAILURE_PROVIDER_ERROR
    assert result.provider_error_kind in (PROVIDER_UNKNOWN, "provider")


def test_empty_output_has_own_failure_type():
    from types import SimpleNamespace as NS

    empty = NS(choices=[NS(message=NS(content=""))])
    result = scripted_generator([empty, empty]).generate("What is hybrid search?", CHUNKS)
    assert result.failure_type == FAILURE_EMPTY_OUTPUT
    assert result.provider_error_kind is None


def test_sanity_check_failure_has_own_failure_type():
    long_text = (
        "Hybrid search combines dense retrieval with bm25 term matching and fuses both ranking lists "
        "using reciprocal rank fusion so that a document appearing high in either list receives a strong "
        "combined score that improves overall retrieval quality and robustness across corpora."
    )
    chunks = [{
        "chunk_id": "c1",
        "doc_id": "d1",
        "text": long_text,
        "retrieval_score": 0.8,
        "metadata": {},
    }]

    result = scripted_generator([json_answer(long_text)]).generate("What is hybrid search?", chunks)
    assert result.failure_type == FAILURE_SANITY_CHECK
    assert result.provider_error_kind is None


def test_plain_generator_abstention_is_not_a_provider_error():
    gen = scripted_generator([json_answer("irrelevant")])
    result = gen.generate("What is hybrid search?", CHUNKS, retrieval_decision="REJECT")
    assert result.failure_type == FAILURE_INSUFFICIENT_CONTEXT
    assert result.provider_error_kind is None


def test_constants_are_stable():
    assert FAILURE_PROVIDER_ERROR == "provider_error"
    assert FAILURE_INSUFFICIENT_CONTEXT == "insufficient_context"
    assert FAILURE_EMPTY_OUTPUT == "empty_output"
    assert FAILURE_SANITY_CHECK == "sanity_check"
    assert FAILURE_UNSUPPORTED_CLAIMS == "unsupported_claims"
    assert {PROVIDER_RATE_LIMIT, PROVIDER_CONNECTION, PROVIDER_AUTH,
            PROVIDER_INVALID_JSON, PROVIDER_UNKNOWN}
