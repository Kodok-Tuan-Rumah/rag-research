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

        async def assess_retrieval_quality(self, chunks, query, query_type=None):
            return {"confidence": {
                "confidence_score": 0.8,
                "decision": "GENERATE",
                "signals": {"unstable_retrieval": True},
            }}

    agent = Agent.__new__(Agent)
    agent.max_steps = 3
    agent.policy_engine = type(
        "Policy", (), {"initial_plan": lambda self, query: type(
            "Plan", (), {
                "top_k_hint": 5,
                "query_type": "general",
                "allow_multi_query": False,
                "allow_hyde": False,
                "generation_hint": {"max_tokens": 10, "temperature": 0.0},
            }
        )()}
    )()
    agent.tools = Tools()
    agent.confidence_engine = type("Confidence", (), {})()

    result = asyncio.run(agent.run("test query"))

    assert result.status == AgentStatus.ABSTAINED
    assert result.state.MAX_STEPS == 3
    assert agent.tools.search_calls <= 2
    # The retry must not enable multi-query expansion. This assertion previously
    # required True, locking in behaviour that contradicted PROJECT_GOALS.md §9
    # ("Do not re-enable") and was not among the three adjustments §3.4 specifies
    # for the retry. See tests/test_retry_contract.py for the full contract.
    assert agent.tools.search_params[1]["use_multi_query"] is False
    assert agent.tools.search_params[1]["use_hyde"] is False


@pytest.mark.parametrize(
    "signals",
    [
        {"unstable_retrieval": True},
        {"spurious_match": True},
        {"unstable_retrieval": True, "spurious_match": True},
    ],
)
def test_agent_reject_decision_overrides_heuristic_signals(signals):
    agent = Agent.__new__(Agent)

    assert agent._decide_next_action(signals, "REJECT") == "REJECT"


def _run_agent_decision_case(confidence_score):
    import asyncio
    from types import SimpleNamespace

    from src.controller.agent import Agent
    from src.controller.agent_state import AgentStatus

    class Tools:
        def __init__(self):
            self.generate_calls = []
            self.assessment_query_types = []

        async def call_tool(self, name, params):
            if name == "search_hybrid":
                return {
                    "chunks": [
                        {
                            "chunk_id": f"c{i}",
                            "doc_id": f"d{i}",
                            "text": f"grounded context {i}.",
                            "retrieval_score": 0.8 - i * 0.01,
                        }
                        for i in range(1, 5)
                    ]
                }
            if name == "generate_answer":
                self.generate_calls.append(params)
                response = SimpleNamespace(
                    status="ANSWERED",
                    answer="grounded answer",
                )
                return {"response": response}
            raise AssertionError(f"Unexpected tool call: {name}")

        async def assess_retrieval_quality(self, chunks, query, query_type=None):
            self.assessment_query_types.append(query_type)
            decision = (
                "REJECT" if confidence_score < 0.35
                else "PARTIAL_TRUST" if confidence_score < 0.65
                else "GENERATE"
            )
            return {
                "confidence": {
                    "confidence_score": confidence_score,
                    "decision": decision,
                    "signals": {},
                }
            }

    tools = Tools()
    agent = Agent.__new__(Agent)
    agent.max_steps = 8
    agent.policy_engine = SimpleNamespace(
        initial_plan=lambda query: SimpleNamespace(
            top_k_hint=10,
            query_type="factual",
            allow_multi_query=False,
            allow_hyde=False,
            generation_hint={"max_tokens": 100, "temperature": 0.0},
        )
    )
    agent.confidence_engine = SimpleNamespace()
    agent.tools = tools

    result = asyncio.run(agent.run("test query"))
    return result, tools.generate_calls, tools.assessment_query_types, AgentStatus


@pytest.mark.parametrize(
    ("confidence_score", "expected_status", "expected_generation_calls"),
    [
        (0.2, "abstained", 0),  # REJECT
        (0.5, "answered", 1),   # PARTIAL_TRUST
        (0.8, "answered", 1),   # GENERATE
    ],
)
def test_agent_decision_contract(confidence_score, expected_status, expected_generation_calls):
    result, generation_calls, query_types, agent_status = _run_agent_decision_case(confidence_score)

    expected_enum = (
        agent_status.ABSTAINED if expected_status == "abstained" else agent_status.ANSWERED
    )
    assert result.status == expected_enum
    assert len(generation_calls) == expected_generation_calls
    expected_query_types = ["factual", "factual"] if confidence_score < 0.35 else ["factual"]
    assert query_types == expected_query_types

    if confidence_score == 0.5:
        assert generation_calls[0]["query_type"] == "factual"
        assert generation_calls[0]["retrieval_decision"] == "PARTIAL_TRUST"
        assert generation_calls[0]["retrieval_confidence"] == confidence_score
        assert len(generation_calls[0]["chunks"]) == 3
        assert result.answer.startswith("[Low confidence — verify independently] ")
    elif confidence_score == 0.8:
        assert generation_calls[0]["retrieval_decision"] == "GENERATE"
        assert len(generation_calls[0]["chunks"]) == 4
        assert result.answer == "grounded answer"


def test_agent_retry_replaces_previous_retrieval_candidates():
    import asyncio
    from types import SimpleNamespace

    from src.controller.agent import Agent
    from src.controller.agent_state import AgentStatus

    first_chunks = [{
        "chunk_id": "initial",
        "doc_id": "doc-old",
        "text": "initial candidate evidence.",
        "retrieval_score": 0.8,
    }]
    retry_chunks = [
        {
            "chunk_id": "retry-low",
            "doc_id": "doc-new",
            "text": "retry candidate with a lower score.",
            "retrieval_score": 0.3,
        },
        {
            "chunk_id": "retry-high",
            "doc_id": "doc-new",
            "text": "retry candidate with a higher score.",
            "retrieval_score": 0.7,
        },
    ]

    class Tools:
        def __init__(self):
            self.search_results = [first_chunks, retry_chunks]
            self.assessed_chunks = []
            self.generation_chunks = None
            self.assessment_count = 0

        async def call_tool(self, name, params):
            if name == "search_hybrid":
                return {"chunks": self.search_results.pop(0)}
            if name == "generate_answer":
                self.generation_chunks = params["chunks"]
                return {
                    "response": SimpleNamespace(
                        status="ANSWERED",
                        answer="grounded answer",
                    )
                }
            raise AssertionError(f"Unexpected tool call: {name}")

        async def assess_retrieval_quality(self, chunks, query, query_type=None):
            self.assessed_chunks.append([chunk["chunk_id"] for chunk in chunks])
            self.assessment_count += 1
            if self.assessment_count == 1:
                return {"confidence": {
                    "confidence_score": 0.5,
                    "decision": "PARTIAL_TRUST",
                    "signals": {"unstable_retrieval": True},
                }}
            return {"confidence": {
                "confidence_score": 0.8,
                "decision": "GENERATE",
                "signals": {},
            }}

    tools = Tools()
    agent = Agent.__new__(Agent)
    agent.max_steps = 8
    agent.policy_engine = SimpleNamespace(
        initial_plan=lambda query: SimpleNamespace(
            top_k_hint=5,
            query_type="general",
            allow_multi_query=False,
            allow_hyde=False,
            generation_hint={"max_tokens": 100, "temperature": 0.0},
        )
    )
    agent.confidence_engine = SimpleNamespace()
    agent.tools = tools

    result = asyncio.run(agent.run("test query"))

    assert result.status == AgentStatus.ANSWERED
    assert tools.assessed_chunks == [["initial"], ["retry-high", "retry-low"]]
    assert [chunk["chunk_id"] for chunk in tools.generation_chunks] == [
        "retry-high",
        "retry-low",
    ]
    assert result.retrieved_ids == ["retry-high", "retry-low"]