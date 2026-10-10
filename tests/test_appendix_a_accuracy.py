"""Appendix A of PROJECT_GOALS.md must match the runtime.

Appendix A is the project's systematical description of its own runtime, which
is why §5 defines the confidence thresholds as calibration parameters and Phase 3
depends on being able to change them.

It was made false on 2026-10-11 and nothing noticed. `confidence_reject_threshold`
and `confidence_partial_threshold` were moved out of `ConfidenceEngine.__init__`
into `Settings` so Phase 3 could calibrate them without editing source, and the
Appendix A row still read `| ConfidenceEngine.__init__ | **no** |` — asserting
the opposite of what the change was for.

That is the same failure `test_readme_accuracy.py` was written for, in a
document nobody guarded. This is the guard.

Two things are checked:

1. the JSON block's values equal the live runtime values
2. the "Env-overridable" column matches reality for the keys it lists, by
   asking the code rather than by hardcoding an expectation
"""

from __future__ import annotations

import inspect
import json
import pathlib
import re

import pytest

import src._compat  # noqa: F401  must precede qdrant_client on 3.14

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
GOALS = REPO_ROOT / "PROJECT_GOALS.md"

START = "## Appendix A — Current Runtime Snapshot"
END = "## Appendix B — Known Issues"


def _appendix_a() -> str:
    text = GOALS.read_text(encoding="utf-8")
    start = text.index(START)
    end = text.index(END)
    return text[start:end]


def _snapshot() -> dict:
    block = re.search(r"```json\n(\{.*?\})\n```", _appendix_a(), re.S)
    assert block, "Appendix A no longer contains a json snapshot block"
    return json.loads(block.group(1))


def test_snapshot_values_match_the_runtime():
    from src.api.config import settings
    from src.controller.confidence_engine import ConfidenceEngine
    from src.controller.policy_engine import PolicyEngine

    snap = _snapshot()
    engine = ConfidenceEngine()
    policy = PolicyEngine()

    assert snap["embedding_model"] == settings.embed_model
    assert snap["reranker_model"] == settings.reranker_model
    assert snap["candidate_k_setting"] == settings.candidate_k
    assert snap["rrf_k"] == settings.hybrid_rrf_k
    assert snap["hyde_enabled"] is settings.use_hyde
    assert snap["multi_query_enabled"] is settings.use_multi_query
    assert snap["llm_model"] == settings.groq_model
    assert snap["context_max_chars"] == settings.generation_max_chars
    assert snap["confidence_reject_threshold"] == engine.reject_threshold
    assert snap["confidence_partial_threshold"] == engine.partial_threshold
    assert snap["api_top_k_policy_defaults"] == {
        kind: policy.retrieval_strategy(kind)["top_k"]
        for kind in ("factual", "general", "reasoning")
    }


def test_confidence_thresholds_are_documented_as_configurable():
    """The row that went stale on 2026-10-11."""
    appendix = _appendix_a()
    for key in ("confidence_reject_threshold", "confidence_partial_threshold"):
        row = next(
            line for line in appendix.splitlines()
            if line.startswith(f"| `{key}`")
        )
        assert "| **yes**" in row or "| yes" in row, (
            f"Appendix A says `{key}` is not env-overridable, but it now comes "
            f"from Settings. Row: {row}"
        )
        assert "settings.confidence_" in row, (
            f"Appendix A should name Settings as the source. Row: {row}"
        )


def test_the_stale_hardcoded_note_is_gone():
    appendix = _appendix_a()
    assert "cannot be changed without editing source" not in appendix, (
        "Appendix A still claims the confidence thresholds require editing "
        "source; they have been configurable since 2026-10-11"
    )


def test_retriever_takes_expansion_flags_per_call_not_per_instance():
    """The second Appendix A claim invalidated on 2026-10-11."""
    from src.retrieval.hybrid_retriever import MasterHybridRetriever

    init = inspect.signature(MasterHybridRetriever.__init__).parameters
    search = inspect.signature(MasterHybridRetriever.search).parameters
    for name in ("use_multi_query", "use_hyde"):
        assert name not in init, (
            f"{name} is a constructor argument but _run_search always overrides "
            f"it, so the argument has no effect"
        )
        assert name in search, f"{name} must be a per-call argument"


def test_appendix_a_carries_a_verification_date():
    """An unverified snapshot is indistinguishable from a stale one."""
    appendix = _appendix_a()
    assert re.search(r"re-verified against source on \d{4}-\d{2}-\d{2}", appendix, re.I), (
        "Appendix A must record when it was last checked against source"
    )


def test_a5_note_matches_the_measured_retry_behaviour():
    """§3.4's third action is a no-op; the document must not imply otherwise."""
    appendix = _appendix_a()
    assert "min_score_threshold" in appendix, (
        "Appendix A should record that min_score_threshold compares against "
        "RRF scores and therefore filters nothing"
    )