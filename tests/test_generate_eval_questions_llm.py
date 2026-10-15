"""Offline tests for the LLM question generator.

None of these call the API. They pin the parts that decide whether a generated
row is trustworthy — pairing, parsing, truncation handling and backoff — so a
change there fails here instead of quietly shrinking the dataset.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_eval_questions_llm import (  # noqa: E402
    MAX_ESCALATION,
    MAX_TOKENS,
    MIN_INTERVAL,
    MULTIHOP_PROMPT,
    _complete,
    _jaccard,
    _sleep_for_api_error,
    balanced_pairs,
    parse_multihop_reply,
    parse_reply,
)


def _chunk(doc_id: str, idx: int, text: str) -> dict:
    return {
        "chunk_id": f"{doc_id}_c{idx}",
        "doc_id": doc_id,
        "text": text,
    }


def test_jaccard_identical_is_one():
    assert _jaccard("alpha beta gamma", "alpha beta gamma") == pytest.approx(1.0)


def test_jaccard_disjoint_is_zero():
    assert _jaccard("alpha beta", "gamma delta") == 0.0


def test_jaccard_empty_is_zero():
    # An empty side must not raise; a divide-by-zero here would abort a whole run.
    assert _jaccard("", "alpha") == 0.0


def test_balanced_pairs_stay_within_one_document():
    chunks = [_chunk(f"d{i}", j, f"content number {i}-{j} unique{i}{j}")
              for i in range(3) for j in range(6)]
    pairs = balanced_pairs(chunks, 2, random.Random(42))
    assert pairs, "expected pairs from a 6-chunk-per-doc corpus"
    for a, b in pairs:
        assert a["doc_id"] == b["doc_id"]
        assert a["chunk_id"] != b["chunk_id"]


def test_balanced_pairs_never_pair_near_duplicate_text():
    # A pair that restates the same text cannot force a two-hop answer: one
    # half already holds everything. Checked as an invariant over several
    # seeds, because whether the duplicate pair is even formed depends on the
    # shuffle — the guarantee is that no *emitted* pair is a near-duplicate.
    chunks = [
        _chunk("d0", 0, "the filter removes documents below the threshold"),
        _chunk("d0", 1, "the filter removes documents below the threshold"),
        _chunk("d0", 2, "an unrelated paragraph about evaluation metrics"),
        _chunk("d0", 3, "a distinct paragraph concerning index construction"),
        _chunk("d0", 4, "index construction uses a sparse posting list"),
        _chunk("d0", 5, "evaluation metrics recall and mrr are reported"),
    ]
    assert _jaccard(chunks[0]["text"], chunks[1]["text"]) > 0.5, "fixture is not a duplicate"

    for seed in range(8):
        pairs = balanced_pairs(chunks, 10, random.Random(seed))
        for a, b in pairs:
            assert _jaccard(a["text"], b["text"]) <= 0.5, (
                f"near-duplicate pair emitted at seed {seed}: "
                f"{a['chunk_id']} + {b['chunk_id']}"
            )


def test_balanced_pairs_are_deterministic_for_a_seed():
    chunks = [_chunk(f"d{i}", j, f"prose {i} {j} distinct{i}{j}word")
              for i in range(2) for j in range(8)]
    first = balanced_pairs(chunks, 3, random.Random(7))
    second = balanced_pairs(chunks, 3, random.Random(7))
    assert [(a["chunk_id"], b["chunk_id"]) for a, b in first] == \
           [(a["chunk_id"], b["chunk_id"]) for a, b in second]


def test_balanced_pairs_respect_per_doc_quota():
    chunks = [_chunk(f"d{i}", j, f"body {i}-{j} token{i}{j}x")
              for i in range(3) for j in range(6)]
    pairs = balanced_pairs(chunks, 1, random.Random(3))
    per_doc: dict[str, int] = {}
    for a, _ in pairs:
        per_doc[a["doc_id"]] = per_doc.get(a["doc_id"], 0) + 1
    assert set(per_doc.values()) == {1}, f"quota not respected: {per_doc}"


def test_parse_multihop_reply_extracts_json_from_surrounding_text():
    raw = 'thinking...\n{"question": "Which stages combine?", "answer": "filter and rank"}\n'
    assert parse_multihop_reply(raw) == ("Which stages combine?", "filter and rank")


def test_parse_multihop_reply_rejects_unusable_output():
    assert parse_multihop_reply("") is None
    assert parse_multihop_reply("no json at all") is None
    assert parse_multihop_reply('{"question": "Q?"}') is None
    assert parse_multihop_reply('{"answer": "A."}') is None


def test_parse_reply_rejects_malformed_json():
    assert parse_reply('{"question": "Q", "answer": ') is None


def test_multihop_prompt_carries_both_passages():
    prompt = MULTIHOP_PROMPT.format(chunk_a="TEXT-ALPHA", chunk_b="TEXT-BETA")
    assert "TEXT-ALPHA" in prompt and "TEXT-BETA" in prompt
    assert '<passage id="A">' in prompt and '<passage id="B">' in prompt
    assert "BOTH" in prompt


def test_multihop_prompt_forbids_meta_questions():
    prompt = MULTIHOP_PROMPT.format(chunk_a="x", chunk_b="y")
    assert "the passage" in prompt
    assert "CANNOT be answered from either passage alone" in prompt


class _FakeChoice:
    def __init__(self, content: str, finish_reason: str) -> None:
        self.message = type("M", (), {"content": content})()
        self.finish_reason = finish_reason


class _FakeCompletions:
    def __init__(self, script: list[tuple[str, str]]) -> None:
        self.script = list(script)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        content, finish = self.script.pop(0)
        return type("R", (), {"choices": [_FakeChoice(content, finish)]})()


class _FakeClient:
    def __init__(self, script: list[tuple[str, str]]) -> None:
        self.chat = type("C", (), {"completions": _FakeCompletions(script)})()


def test_complete_escalates_budget_after_truncation():
    # A fixed max_tokens cannot serve both the easy and the hard prompt.
    client = _FakeClient([
        ("", "length"),
        ('{"question": "Q?", "answer": "A."}', "stop"),
    ])
    content, finish = _complete(client, "m", "prompt")
    assert finish == "stop"
    assert parse_reply(content) == ("Q?", "A.")
    budgets = [c["max_tokens"] for c in client.chat.completions.calls]
    assert budgets[1] == budgets[0] * 2, f"budget must escalate: {budgets}"


def test_complete_gives_up_as_length_at_the_ceiling():
    # One escalation to the ceiling, then stop. Retrying past it would ask for
    # 8192 and make the rate limiter more likely to fire.
    client = _FakeClient([("", "length")] * 4)
    content, finish = _complete(client, "m", "prompt", retries=2)
    assert finish == "length"
    assert content == ""
    budgets = [c["max_tokens"] for c in client.chat.completions.calls]
    assert budgets == [MAX_TOKENS, MAX_ESCALATION], f"unexpected budgets: {budgets}"


def test_max_tokens_clears_the_single_chunk_budget():
    assert MAX_TOKENS >= 1024


def test_rate_limit_backs_off_far_longer_than_a_connection_fault():
    class RateLimitError(Exception):
        pass

    assert _sleep_for_api_error(RateLimitError(), 0) == 15.0
    assert _sleep_for_api_error(RateLimitError(), 1) == 30.0
    assert _sleep_for_api_error(RateLimitError(), 2) == 60.0
    assert _sleep_for_api_error(ConnectionError(), 0) == 2.0

def test_complete_throttles_between_attempts(monkeypatch):
    # Pacing, not backoff, is what keeps the run under Groq's token-based limit.
    slept: list[float] = []
    monkeypatch.setattr("scripts.generate_eval_questions_llm.time.sleep", slept.append)
    client = _FakeClient([('{"question": "Q?", "answer": "A."}', "stop")])
    _complete(client, "m", "prompt")
    assert slept == [MIN_INTERVAL], f"expected one throttle before the call, got {slept}"


def test_escalation_stops_at_the_ceiling():
    # Unbounded doubling would request 8192, which raises the odds of tripping
    # the very limiter the escalation is trying to dodge.
    client = _FakeClient([("", "length")] * 4)
    _complete(client, "m", "prompt", retries=3)
    budgets = [c["max_tokens"] for c in client.chat.completions.calls]
    assert max(budgets) <= MAX_ESCALATION, f"escalation exceeded ceiling: {budgets}"
    assert budgets[0] == MAX_TOKENS


def test_min_interval_is_set():
    # A zero interval reintroduces the failure this pacing was added to fix.
    assert MIN_INTERVAL > 0
