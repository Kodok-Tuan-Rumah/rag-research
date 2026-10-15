from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# One retrieval retry is the shared Phase 1 contract for API and Agent paths.
MAX_RETRIEVAL_RETRIES = 1


class AgentStatus(Enum):
    RUNNING = "running"
    ANSWERED = "answered"
    ABSTAINED = "abstained"
    FAILED = "failed"


@dataclass
class AgentState:
    query: str
    step_count: int = 0
    MAX_STEPS: int = 8
    scratchpad: list[dict] = field(default_factory=list)
    retrieved_chunks: list = field(default_factory=list)
    confidence_score: float = 0.0
    prev_top_score: float = 0.0
    stagnation_count: int = 0
    force_expand: bool = False
    retrieval_retries: int = 0
    status: AgentStatus = AgentStatus.RUNNING
    source_chunk_id: str | None = None
    last_tool: str | None = None
    last_result: Any = None
    # Machine-readable cause when the loop ends in a non-answer outcome, so
    # PROJECT_GOALS.md §4 failure classification is available to callers.
    failure_reason: str | None = None

    @property
    def retry_attempted(self) -> bool:
        """Compatibility signal for callers that only need retry/no-retry."""
        return self.retrieval_retries > 0

    def add_observation(self, tool: str, result: Any):
        self.scratchpad.append({
            "step": self.step_count,
            "tool": tool,
            "result": result,
        })
        self.step_count += 1
        self.last_tool = tool
        self.last_result = result

    def is_budget_exhausted(self) -> bool:
        return self.step_count >= self.MAX_STEPS
