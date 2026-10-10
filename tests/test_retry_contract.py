"""Retry contract — PROJECT_GOALS.md §3.4.

Contract:

    Attempt 1: standard retrieval → generate
      └── IF (confidence < threshold OR grounding < 0.5):
            Attempt 2 (ONE retry only)
              └── IF still low confidence → ABSTAIN

    Hard constraints on retry:
      - Max iterations: 2 (initial + 1 retry)
      - Max additional LLM calls from retry: 1

These tests pin the iteration bound. They are written against the contract, not
against current behaviour, so a regression shows up as a failure rather than a
quietly updated expectation.

`test_agent_loop_stops_after_two_attempts` currently FAILS: the RETRIEVE_AGAIN
branch checks only `is_budget_exhausted()` and never `retry_attempted`, so a
persistently unstable signal drives the loop to MAX_STEPS/2 attempts instead of
2. See Task 7.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from src.controller.agent import Agent
from src.controller.agent_state import AgentStatus

MAX_ATTEMPTS_PER_CONTRACT = 2


class RecordingTools:
    """Tool stub that counts retrieval attempts and always reports instability."""

    def __init__(self, *, unstable: bool = True, decision: str = "GENERATE",
                 confidence: float = 0.8):
        self.search_calls = 0
        self.search_params: list[dict] = []
        self.generate_calls = 0
        self.unstable = unstable
        self.decision = decision
        self.confidence = confidence

    async def call_tool(self, name, params):
        if name == "search_hybrid":
            self.search_calls += 1
            self.search_params.append(dict(params))
            return {
                "chunks": [
                    {
                        "chunk_id": f"c{i}",
                        "doc_id": "d1",
                        "text": f"grounded context chunk {i}.",
                        # Vary the top score so stagnation detection cannot fire
                        # and mask the iteration bound under test.
                        "retrieval_score": 0.9 - self.search_calls * 0.2,
                        "metadata": {},
                    }
                    for i in range(1, 4)
                ]
            }
        if name == "generate_answer":
            self.generate_calls += 1
            return {
                "response": SimpleNamespace(status="ANSWERED", answer="grounded answer")
            }
        raise AssertionError(f"unexpected tool: {name}")

    async def assess_retrieval_quality(self, chunks, query, query_type=None):
        return {
            "confidence": {
                "confidence_score": self.confidence,
                "decision": self.decision,
                "signals": {"unstable_retrieval": self.unstable},
            }
        }


def build_agent(tools: RecordingTools, *, max_steps: int = 8) -> Agent:
    agent = Agent.__new__(Agent)
    agent.max_steps = max_steps
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
    return agent


def test_agent_loop_stops_after_two_attempts():
    """A persistently unstable signal must not extend the loop beyond 2 attempts.

    §3.4 allows one retry, so at most 2 retrieval attempts may happen.
    """
    tools = RecordingTools(unstable=True, decision="GENERATE", confidence=0.8)
    agent = build_agent(tools, max_steps=8)

    result = asyncio.run(agent.run("test query"))

    assert tools.search_calls <= MAX_ATTEMPTS_PER_CONTRACT, (
        f"retry contract §3.4 allows {MAX_ATTEMPTS_PER_CONTRACT} retrieval attempts; "
        f"loop performed {tools.search_calls} (step_count={result.state.step_count})"
    )


def test_agent_retry_uses_bm25_heavy_weights_once():
    """The single permitted retry must shift weight toward BM25, per §3.4."""
    tools = RecordingTools(unstable=True, decision="GENERATE", confidence=0.8)
    agent = build_agent(tools, max_steps=8)

    asyncio.run(agent.run("test query"))

    assert len(tools.search_params) >= 2, "expected at least one retry attempt"
    first, second = tools.search_params[0], tools.search_params[1]
    assert (first["dense_weight"], first["bm25_weight"]) == (0.5, 0.5)
    assert (second["dense_weight"], second["bm25_weight"]) == (0.3, 0.7)
    assert second["min_score_threshold"] == pytest.approx(-0.1)


def test_agent_retry_is_bounded_even_when_step_budget_is_large():
    """The bound must come from the retry contract, not from MAX_STEPS.

    A generous step budget must not buy extra retrieval attempts.
    """
    tools = RecordingTools(unstable=True, decision="GENERATE", confidence=0.8)
    agent = build_agent(tools, max_steps=64)

    asyncio.run(agent.run("test query"))

    assert tools.search_calls <= MAX_ATTEMPTS_PER_CONTRACT, (
        f"step budget must not relax the §3.4 bound; got {tools.search_calls} attempts"
    )


def test_agent_stable_signal_generates_without_retry():
    """Control case: a stable, confident signal must not trigger any retry."""
    tools = RecordingTools(unstable=False, decision="GENERATE", confidence=0.8)
    agent = build_agent(tools, max_steps=8)

    result = asyncio.run(agent.run("test query"))

    assert tools.search_calls == 1
    assert tools.generate_calls == 1
    assert result.status == AgentStatus.ANSWERED


def test_agent_reject_abstains_after_bounded_retry():
    """REJECT must retry once, then abstain — and never call the generator."""
    tools = RecordingTools(unstable=False, decision="REJECT", confidence=0.1)
    agent = build_agent(tools, max_steps=8)

    result = asyncio.run(agent.run("test query"))

    assert tools.search_calls <= MAX_ATTEMPTS_PER_CONTRACT
    assert tools.generate_calls == 0
    assert result.status == AgentStatus.ABSTAINED
    assert result.state.retry_attempted is True

# --- the retry must match §3.4 exactly, and must not re-enable §9 features ---


def test_retry_does_not_enable_multi_query():
    """§9 lists multi-query expansion as a non-goal: "Do not re-enable".

    The retry path used to force use_multi_query=True, which contradicted the
    non-goal and added an LLM call to every retry. §3.4 specifies three retry
    adjustments and query expansion is not among them.
    """
    tools = RecordingTools(unstable=True, decision="GENERATE", confidence=0.8)
    agent = build_agent(tools, max_steps=8)

    asyncio.run(agent.run("test query"))

    assert len(tools.search_params) >= 2
    first, second = tools.search_params[0], tools.search_params[1]
    assert first["use_multi_query"] is False
    assert second["use_multi_query"] is False, (
        "retry must not enable multi-query expansion; §9 forbids re-enabling it "
        "and §3.4 does not list it as a retry adjustment"
    )
    assert second["use_hyde"] is False, "§9 also forbids HyDE"


def test_retry_adjustments_match_the_section_3_4_contract():
    """§3.4 lists exactly three retry adjustments. No more, no fewer."""
    tools = RecordingTools(unstable=True, decision="GENERATE", confidence=0.8)
    agent = build_agent(tools, max_steps=8)

    asyncio.run(agent.run("test query"))

    base, retry = tools.search_params[0], tools.search_params[1]

    # 1. widen the candidate pool
    assert retry["k"] > base["k"]
    # 2. shift weight toward BM25 by 0.2
    assert retry["bm25_weight"] - base["bm25_weight"] == pytest.approx(0.2)
    assert retry["dense_weight"] - base["dense_weight"] == pytest.approx(-0.2)
    # 3. relax the score floor 0.0 -> -0.1
    assert base["min_score_threshold"] == pytest.approx(0.0)
    assert retry["min_score_threshold"] == pytest.approx(-0.1)

    # nothing else changed
    changed = {k for k in base if base[k] != retry[k]}
    assert changed == {"k", "dense_weight", "bm25_weight", "min_score_threshold"}, (
        f"retry changed unexpected fields: {changed - {'k','dense_weight','bm25_weight','min_score_threshold'}}"
    )
