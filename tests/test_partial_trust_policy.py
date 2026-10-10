"""The PARTIAL_TRUST presentation rule lives in exactly one place.

PROJECT_GOALS.md §4 requires a low-confidence answer to be visibly qualified
and grounded on fewer chunks. It was implemented four times — agent.py
truncated, agent.py prefixed, api/main.py did both, generator.py truncated
again — and the copies had drifted in two ways that mattered:

- `api/main.py` prefixed unconditionally, so a generator abstain inside the
  PARTIAL_TRUST branch returned the literal string
  "[Low confidence — verify independently] INSUFFICIENT_CONTEXT" as the answer.
  `check_retrieval_confidence()` can still abstain after the confidence engine
  has said PARTIAL_TRUST, so that path was reachable.
- the Agent skipped the prefix when the answer was empty, so the two adapters
  disagreed on the same condition.

These tests pin the shared behaviour so the copies cannot reappear.
"""

from __future__ import annotations

import pathlib
import re

import pytest

from src.generation.generator import (
    ABSTAIN_ANSWER,
    PARTIAL_TRUST_CHUNK_LIMIT,
    PARTIAL_TRUST_PREFIX,
    apply_partial_trust_policy,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def chunks(n: int = 10) -> list[str]:
    return [f"c{i}" for i in range(n)]


@pytest.mark.parametrize("decision", [None, "GENERATE", "REJECT", "something_else"])
def test_other_decisions_are_untouched(decision):
    c, a = apply_partial_trust_policy(chunks(), "answer", decision)
    assert c == chunks()
    assert a == "answer"


def test_partial_trust_limits_chunks_and_prefixes():
    c, a = apply_partial_trust_policy(chunks(), "answer", "PARTIAL_TRUST")
    assert len(c) == PARTIAL_TRUST_CHUNK_LIMIT == 3
    assert a == PARTIAL_TRUST_PREFIX + "answer"


def test_an_abstention_is_not_a_low_confidence_answer():
    """The bug: the API prefixed abstentions.

    An abstain means the pipeline declined to answer. Qualifying it with
    "verify independently" invites the reader to verify an answer that was
    never produced.
    """
    c, a = apply_partial_trust_policy(chunks(), ABSTAIN_ANSWER, "PARTIAL_TRUST")
    assert a == ABSTAIN_ANSWER
    assert PARTIAL_TRUST_PREFIX not in a
    # The chunk limit still applies; only the presentation changes.
    assert len(c) == PARTIAL_TRUST_CHUNK_LIMIT


def test_an_empty_answer_gets_no_prefix():
    _, a = apply_partial_trust_policy(chunks(), "", "PARTIAL_TRUST")
    assert a == ""


def test_the_rule_is_not_reimplemented_elsewhere():
    """Guard against a fifth copy appearing."""
    offenders = []
    for path in (REPO_ROOT / "src").rglob("*.py"):
        if path.name == "generator.py":
            continue
        text = path.read_text(encoding="utf-8")
        # code lines only, not comments or the helper's own definition
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "Low confidence" in stripped:
                offenders.append(f"{path.relative_to(REPO_ROOT)}: {stripped}")
    assert not offenders, (
        "PARTIAL_TRUST prefix written inline again; call "
        "apply_partial_trust_policy instead:\n" + "\n".join(offenders)
    )


def test_adapters_import_the_shared_helper():
    for rel in ("src/api/main.py", "src/controller/agent.py"):
        text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        assert "apply_partial_trust_policy" in text, f"{rel} does not use the helper"
        # and must not carry its own truncation literal for the same rule
        code = "\n".join(
            line for line in text.splitlines()
            if not line.strip().startswith("#")
        )
        assert not re.search(r"(?<!_LIMIT)\bchunks\[:3\]", code), (
            f"{rel} still truncates to 3 inline; use PARTIAL_TRUST_CHUNK_LIMIT"
        )