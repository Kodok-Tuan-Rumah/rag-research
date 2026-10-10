"""Confidence thresholds must be configurable without changing behaviour.

PROJECT_GOALS.md §5 defines the reject and partial cuts as calibration
parameters. They were literals in `ConfidenceEngine.__init__`, so Phase 3
calibration required editing source, and Appendix A recorded both as
non-overridable. They now come from `Settings`, with the same defaults.

This test is the reason that change is safe: it re-implements the original
inline branch verbatim and compares it against the refactored version across
every query type and a dense sweep of scores. A refactor that silently moved a
clamp would show up here as a nonzero difference.

The clamps matter more than they look. `reject` is floored at 0.0 for
`reasoning` and `partial` is floored at `reject` — so with a reject threshold
below 0.05, reasoning queries cannot be rejected at all. That is pre-existing
behaviour, preserved deliberately, not an oversight introduced here.
"""

from __future__ import annotations

import pytest

from src.api.config import settings
from src.controller.confidence_engine import TYPE_ADJUSTMENTS, ConfidenceEngine

QUERY_TYPES = (None, "factual", "reasoning", "general", "unknown", "something_else")
SCORES = [i / 400.0 for i in range(400)]


def _original(score: float, query_type, reject: float, partial: float) -> str:
    """The branch as it was before the thresholds became configurable."""
    if query_type == "factual":
        reject += 0.05
        partial = min(1.0, partial + 0.05)
    elif query_type == "reasoning":
        reject = max(0.0, reject - 0.05)
        partial = max(reject, partial - 0.05)
    if score < reject:
        return "REJECT"
    if score < partial:
        return "PARTIAL_TRUST"
    return "GENERATE"


@pytest.mark.parametrize("reject,partial", [
    (0.35, 0.65),   # the shipped defaults
    (0.02, 0.04),   # below the adjustment, so both clamps engage
    (0.00, 0.10),
    (0.90, 0.95),
    (0.30, 0.31),   # partial barely above reject
    (0.50, 0.00),   # partial below reject
])
def test_decisions_are_identical_to_the_original_branch(reject, partial):
    engine = ConfidenceEngine(reject_threshold=reject, partial_threshold=partial)
    for query_type in QUERY_TYPES:
        for score in SCORES:
            assert engine._decision_for_score(score, query_type) == _original(
                score, query_type, reject, partial
            ), f"regressed at score={score} type={query_type} reject={reject}"


def test_defaults_are_unchanged():
    engine = ConfidenceEngine()
    assert engine.reject_threshold == 0.35
    assert engine.partial_threshold == 0.65
    assert settings.confidence_reject_threshold == 0.35
    assert settings.confidence_partial_threshold == 0.65


def test_thresholds_come_from_settings(monkeypatch):
    """Calibration must not require editing source."""
    monkeypatch.setattr(settings, "confidence_reject_threshold", 0.42)
    monkeypatch.setattr(settings, "confidence_partial_threshold", 0.77)
    engine = ConfidenceEngine()
    assert engine.reject_threshold == 0.42
    assert engine.partial_threshold == 0.77


def test_explicit_argument_overrides_settings(monkeypatch):
    monkeypatch.setattr(settings, "confidence_reject_threshold", 0.42)
    engine = ConfidenceEngine(reject_threshold=0.11)
    assert engine.reject_threshold == 0.11
    assert engine.partial_threshold == settings.confidence_partial_threshold


def test_boundaries_are_strict_less_than():
    """0.35 is PARTIAL_TRUST, not REJECT; 0.65 is GENERATE, not PARTIAL_TRUST."""
    engine = ConfidenceEngine()
    assert engine._decision_for_score(0.35, "general") == "PARTIAL_TRUST"
    assert engine._decision_for_score(0.65, "general") == "GENERATE"
    assert engine._decision_for_score(0.34, "general") == "REJECT"


def test_type_adjustments_are_the_documented_offsets():
    assert TYPE_ADJUSTMENTS["factual"] == (0.05, 0.05)
    assert TYPE_ADJUSTMENTS["reasoning"] == (-0.05, -0.05)
    assert TYPE_ADJUSTMENTS["general"] == (0.0, 0.0)


def test_no_literal_thresholds_survive_in_the_constructor():
    """The point of the change: no magic numbers left to edit."""
    import inspect
    source = inspect.getsource(ConfidenceEngine.__init__)
    code = "\n".join(
        line for line in source.splitlines() if not line.strip().startswith("#")
    )
    assert "= 0.35" not in code, "reject threshold is still a literal"
    assert "= 0.65" not in code, "partial threshold is still a literal"