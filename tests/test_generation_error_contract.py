"""Generation error contract and orchestrator resilience.

PROJECT_GOALS.md §2.3 treats abstention as a first-class outcome and §4 requires
structured failure observability with no silent failures. A provider-side
failure is neither an answer nor a decision the confidence engine made — it is an
operational error, and surfacing it as an unhandled exception breaks both.

These tests encode three requirements:

  1. A provider JSON-validation failure degrades to a structured abstain
     response, not a raised RuntimeError.
  2. The abstain response carries a machine-readable reason.
  3. Agent.run does not propagate an exception raised by generation.

All three fail against the previous behaviour.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from groq import BadRequestError

from src.controller.agent import Agent
from src.controller.agent_state import AgentStatus
from src.generation.generator import GroqGenerator

CHUNKS = [
    {
        "chunk_id": "c1",
        "doc_id": "d1",
        "text": "Hybrid search combines dense retrieval with BM25 and fuses the lists.",
        "retrieval_score": 0.8,
        "metadata": {},
    }
]


def json_validate_error() -> BadRequestError:
    """Build the exact error Groq raises for response_format validation."""
    import httpx

    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    return BadRequestError(
        "Error code: 400 - {'error': {'message': "
        '"Failed to validate JSON. Please adjust your prompt.", '
        "'type': 'invalid_request_error', 'code': 'json_validate_failed'}}",
        response=httpx.Response(400, request=request),
        body={
            "error": {
                "message": "Failed to validate JSON. Please adjust your prompt.",
                "type": "invalid_request_error",
                "code": "json_validate_failed",
            }
        },
    )


def build_generator(error: BaseException) -> GroqGenerator:
    gen = GroqGenerator.__new__(GroqGenerator)
    gen.model = "openai/gpt-oss-20b"

    def _create(**kwargs):
        raise error

    gen.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=_create))
    )
    gen._confidence = SimpleNamespace()
    return gen


# --- 1. provider validation failure degrades to abstain -----------------------

def test_json_validation_failure_returns_abstain_not_exception():
    gen = build_generator(json_validate_error())

    result = gen.generate("What is hybrid search?", CHUNKS)

    assert result.status == "INSUFFICIENT_CONTEXT"
    assert result.answer == "INSUFFICIENT_CONTEXT"
    assert result.confidence_score == 0.0


def test_authentication_error_still_raises():
    """A bad API key is an operator problem and must not be hidden as an abstain."""
    from groq import AuthenticationError
    import httpx

    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    auth_error = AuthenticationError(
        "Error code: 401 - invalid api key",
        response=httpx.Response(401, request=request),
        body={"error": {"message": "invalid api key"}},
    )
    gen = build_generator(auth_error)

    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        gen.generate("What is hybrid search?", CHUNKS)


# --- 2. the abstain carries a machine-readable reason -----------------------

def test_abstain_response_exposes_reason():
    """§4 requires structured failure classification.

    _make_abstain_response receives a reason and logs it, but the returned object
    exposes only status, so a caller cannot distinguish a provider failure from
    a refusal. This test requires the reason on the response.
    """
    gen = build_generator(json_validate_error())

    result = gen.generate("What is hybrid search?", CHUNKS)

    assert hasattr(result, "reason"), (
        "RAGResponse must expose the abstain reason so callers can classify "
        "the failure without parsing logs"
    )
    reason = getattr(result, "reason", "")
    assert reason, "abstain reason must be non-empty"
    assert "json" in reason.lower() or "validation" in reason.lower(), (
        f"reason should identify the provider validation failure, got {reason!r}"
    )


# --- 3. the orchestrator does not leak generation errors --------------------

class RaisingTools:
    def __init__(self):
        self.search_calls = 0

    async def call_tool(self, name, params):
        if name == "search_hybrid":
            self.search_calls += 1
            return {
                "chunks": [
                    {
                        "chunk_id": "c1",
                        "doc_id": "d1",
                        "text": "grounded context.",
                        "retrieval_score": 0.8,
                        "metadata": {},
                    }
                ]
            }
        if name == "generate_answer":
            raise RuntimeError("Groq generation failed: json_validate_failed")
        raise AssertionError(f"unexpected tool: {name}")

    async def assess_retrieval_quality(self, chunks, query, query_type=None):
        return {
            "confidence": {
                "confidence_score": 0.8,
                "decision": "GENERATE",
                "signals": {},
            }
        }


def test_agent_run_does_not_propagate_generation_exception():
    """Agent.run had no try/except, so a generation error escaped to the caller.

    run_single_query.py and api/main.py both already contain such an error and
    convert it to an abstain. The orchestrator should not be the weak link.
    """
    tools = RaisingTools()
    agent = Agent.__new__(Agent)
    agent.max_steps = 8
    agent.policy_engine = SimpleNamespace(
        initial_plan=lambda query: SimpleNamespace(
            top_k_hint=10,
            query_type="general",
            allow_multi_query=False,
            allow_hyde=False,
            generation_hint={"max_tokens": 100, "temperature": 0.0},
        )
    )
    agent.confidence_engine = SimpleNamespace()
    agent.tools = tools

    result = asyncio.run(agent.run("test query"))

    assert result.status == AgentStatus.ABSTAINED
    assert result.answer is None or "insufficient" in str(result.answer).lower()