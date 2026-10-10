import pytest
from unittest.mock import AsyncMock, MagicMock
from src.controller.agent import Agent
from src.controller.agent_state import AgentStatus


async def _run_agent_abstain_case():
    """
    Regression test: kalau generator balikin status INSUFFICIENT_CONTEXT,
    agent.status HARUS jadi ABSTAINED, bukan tetap RUNNING.
    """
    agent = Agent.__new__(Agent)  # skip __init__ berat (no real Qdrant/Groq connection)

    fake_response = MagicMock()
    fake_response.status = "INSUFFICIENT_CONTEXT"
    fake_response.answer = "INSUFFICIENT_CONTEXT"

    agent.tools = MagicMock()
    agent.tools.call_tool = AsyncMock(return_value={"response": fake_response})

    from src.controller.agent_state import AgentState
    state = AgentState(query="dummy query")

    from src.controller.policy_engine import PlanHints
    plan = MagicMock()
    plan.generation_hint = {"max_tokens": 100, "temperature": 0.0}

    result = await agent._execute_generation(state, plan)

    assert result.status == AgentStatus.ABSTAINED, (
        f"Expected ABSTAINED when response.status=INSUFFICIENT_CONTEXT, "
        f"got {result.status}"
    )


def test_agent_abstains_on_insufficient_context():
    import asyncio

    asyncio.run(_run_agent_abstain_case())


def test_agent_honors_configured_step_budget():
    import asyncio
    from src.controller.agent import Agent
    from src.controller.agent_state import AgentStatus

    class Tools:
        def __init__(self):
            self.search_calls = 0
            self.search_params = []

        async def call_tool(self, name, params):
            if name == "search_hybrid":
                self.search_calls += 1
                self.search_params.append(params)
                return {"chunks": []}

        async def assess_retrieval_quality(self, chunks, query):
            return {"confidence": {
                "confidence_score": 0.1,
                "signals": {"unstable_retrieval": True},
            }}

    agent = Agent.__new__(Agent)
    agent.max_steps = 3
    agent.policy_engine = type(
        "Policy", (), {"initial_plan": lambda self, query: type(
            "Plan", (), {
                "top_k_hint": 5,
                "allow_multi_query": False,
                "allow_hyde": False,
                "generation_hint": {"max_tokens": 10, "temperature": 0.0},
            }
        )()}
    )()
    agent.tools = Tools()
    agent.confidence_engine = type(
        "Confidence", (), {"reject_threshold": 0.35, "partial_threshold": 0.65}
    )()
    agent._decide_next_action = lambda signals, score: "RETRIEVE_AGAIN"

    result = asyncio.run(agent.run("test query"))

    assert result.status == AgentStatus.ABSTAINED
    assert result.state.MAX_STEPS == 3
    assert result.state.step_count == 3
    assert agent.tools.search_calls == 2
    assert agent.tools.search_params[1]["use_multi_query"] is True