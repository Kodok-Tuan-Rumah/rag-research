from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_eval_dataset_v4 import match_gold_chunk  # noqa: E402

CHUNKS_PATH = ROOT / "data" / "processed" / "chunks_semantic_v2.json"
OUT_PATH = ROOT / "data" / "evaluation" / "llm_generated_questions.json"

DEFAULT_MODEL = "openai/gpt-oss-20b"
SEED = 42

# Measured 2026-10-13 against openai/gpt-oss-20b on a real passage:
#   max_tokens=  32 -> empty string, finish_reason="length", no error
#   max_tokens= 768 -> truncated on one chunk, valid JSON on another
#   max_tokens=1024 -> finish_reason="stop", parseable JSON
# The reasoning channel consumes the budget silently and the amount varies per
# prompt, so the budget has to clear the worst case rather than the average.
MAX_TOKENS = 1536

# Bibliography chunks are retrievable-looking but contain no answer. Asking a
# model to write a question about "PMLR, 2023, pp." yields nothing usable.
REFERENCE_SECTIONS = {"references", "bibliography"}

PROMPT = """You are writing evaluation questions for a retrieval system.

Here is one passage from a research paper:

<passage>
{chunk_text}
</passage>

Write ONE question that can be answered using ONLY the passage above.

Rules:
- The answer must be a short fact, number or name stated in the passage.
- Do NOT use outside knowledge. Do NOT invent facts.

The question must be about the SUBJECT MATTER, never about the text itself.
Banned in the question: "the passage", "the text", "this document", "this paper",
"this article", "the excerpt", "mentioned in", "referred to in", "described in",
"according to the text". A reader who has never seen the paper must still be
able to tell what is being asked.

Examples of BANNED: "What does the passage call the parametric memory?"
Examples of ALLOWED: "What is used as parametric memory in the RAG models?"

The answer must be a self-contained fact from the subject matter:
- a short phrase, a number with its unit, or a named entity ("42.7%", "a dense
  vector index", "BART"), OR
- BANNED: a pointer into the document's furniture — "Fig. 1", "Table 2",
  "Section 3", "Eq. 4", "(a)". Referring to a figure is not an answer.

Reply with JSON only, exactly these keys:
{{"question": "...", "answer": "..."}}"""

# A question that points at the carrier rather than the content tests string
# extraction, not retrieval: the answer is in the wording of the question, so
# any retriever that returns the chunk wins regardless of ranking quality.
# Measured 2026-10-13: 6 of 28 generated questions (21%) were of this shape.
_META_QUESTION = re.compile(
    r"\b(passage|passages|excerpt|the text|this text|the document|this document"
    r"|the paper|this paper|the article|this article|the chunk|this chunk"
    r"|mentioned|referred to|described in|stated in|according to the (text|paper|passage))\b",
    re.I,
)

# "Fig. 1" answers the question "which figure" and nothing else. It is a
# pointer into layout, not a fact about the subject.
_LABEL_ONLY_ANSWER = re.compile(
    r"^\s*[\(\[]?\s*(fig|figure|table|tab|eq|equation|section|sec|chapter|appendix"
    r"|scheme|algorithm|listing)\s*\.?\s*\d*\s*[\)\]]?\s*[:.]?\s*$",
    re.I,
)


def quality_reject_reason(question: str, answer: str) -> str | None:
    """Return why this pair is not usable evaluation data, or None if it is.

    Kept deliberately narrow. Short answers are usually good answers — "42.7%",
    "Chatlaw", "davinci" are precise, checkable values, and a blanket length
    rule would throw away exactly the samples Exact Match is best at. Only
    three shapes are rejected, all of which make the metric meaningless
    rather than merely terse.
    """
    q, a = question.strip(), answer.strip()

    if _META_QUESTION.search(q):
        return "meta_question_references_the_document"
    if _LABEL_ONLY_ANSWER.match(a):
        return "answer_is_a_figure_or_table_pointer"
    if not any(ch.isalnum() for ch in a):
        return "answer_has_no_substantive_content"
    # A bare number is a real answer — "8" heads, "42.7%" — so a length floor
    # would reject the most precisely checkable samples there are. A single
    # letter or symbol is not.
    if len(a) == 1 and not a.isdigit():
        return "answer_is_a_single_non_numeric_character"
    return None


def load_chunks() -> list[dict]:
    return json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))


def usable_chunks(chunks: list[dict]) -> list[dict]:
    out = []
    for chunk in chunks:
        section = str((chunk.get("metadata") or {}).get("section", "")).strip().lower()
        if section in REFERENCE_SECTIONS:
            continue
        text = (chunk.get("text") or "").strip()
        if not 200 <= len(text) <= 1800:
            continue
        out.append(chunk)
    return out


def balanced_sample(chunks: list[dict], per_doc: int, rng: random.Random) -> list[dict]:
    """Equal quota per document, so no paper dominates the eval set."""
    by_doc: dict[str, list[dict]] = {}
    for chunk in chunks:
        by_doc.setdefault(chunk["doc_id"], []).append(chunk)

    picked: list[dict] = []
    for doc_id in sorted(by_doc):
        pool = by_doc[doc_id]
        rng.shuffle(pool)
        picked.extend(pool[:per_doc])
    rng.shuffle(picked)
    return picked


def parse_reply(text: str) -> tuple[str, str] | None:
    if not text:
        return None
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    question = str(payload.get("question", "")).strip()
    answer = str(payload.get("answer", "")).strip()
    if not question or not answer:
        return None
    return question, answer


def generate(client, model: str, chunk: dict, retries: int = 2) -> tuple[str, str] | None:
    body = (chunk.get("text") or "")[:1800]
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": PROMPT.format(chunk_text=body)}],
                max_tokens=MAX_TOKENS,
                temperature=0.4,
            )
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"    API error ({type(exc).__name__}): {str(exc)[:120]}")
            time.sleep(2 * (attempt + 1))
            continue

        choice = response.choices[0]
        # A truncated completion ends mid-JSON and parses as nothing. Without
        # this check the row is merely "rejected: no_valid_reply", which hides
        # that the real cause was the token budget rather than the model.
        if choice.finish_reason == "length":
            rejected_reason = f"truncated_at_max_tokens({MAX_TOKENS})"
            print(f"    {rejected_reason} — raise MAX_TOKENS")
            return None

        content = choice.message.content or ""
        parsed = parse_reply(content)
        if parsed:
            return parsed
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-doc", type=int, default=30,
                        help="questions per document (7 documents in this corpus)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    parser.add_argument("--dry-run", type=int, default=0,
                        help="stop after N chunks and write nothing useful")
    args = parser.parse_args()

    key = os.getenv("GROQ_API_KEY")
    if not key:
        print("GROQ_API_KEY is not set.", file=sys.stderr)
        return 2

    # A placeholder that looks like a key produces the same 401 as an expired
    # key, and the 401 does not show which it was. Saying so up front saves a
    # round trip to the console.
    lowered = key.lower()
    if any(marker in lowered for marker in ("your", "here", "xxx", "changeme", "placeholder")):
        print(
            "GROQ_API_KEY looks like a placeholder, not a real key.\n"
            "  prefix: " + key[:12] + "...  length: " + str(len(key)) + "\n"
            "Get a key at https://console.groq.com/keys and either\n"
            "  export GROQ_API_KEY='gsk_...'                      (this shell only)\n"
            "  or fix the stale export in ~/.bashrc, which currently\n"
            "     shadows any .env value in an interactive terminal.",
            file=sys.stderr,
        )
        return 2

    from groq import Groq
    client = Groq(api_key=key)

# Preflight. Without this the loop fails the same way on every chunk, the
    # script exits 0 and writes a file containing zero rows, and a bad API key
    # looks exactly like "the model refused to answer anything".
    #
    # The budget is not arbitrary. gpt-oss-20b is a reasoning model: below
    # ~512 max_tokens it spends the whole allowance thinking and returns an
    # empty string with finish_reason="length" and NO error. A preflight on a
    # small budget would therefore pass against a model that cannot produce
    # output, which is worse than no preflight at all.
    try:
        probe = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
            max_tokens=512,
            temperature=0.0,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Preflight failed — {type(exc).__name__}: {str(exc)[:200]}", file=sys.stderr)
        print("Check GROQ_API_KEY. Nothing was written.", file=sys.stderr)
        return 2

    _text = (probe.choices[0].message.content or "").strip()
    if not _text:
        print(
            f"Preflight reached {args.model} but it returned no text "
            f"(finish_reason={probe.choices[0].finish_reason!r}).\n"
            "A reasoning model can return empty content when max_tokens is too\n"
            "small for it to finish thinking. Raise the preflight budget, or\n"
            "switch --model.",
            file=sys.stderr,
        )
        return 2
    print(f"preflight ok ({args.model}, replied {_text[:20]!r})")

    rng = random.Random(SEED)
    all_chunks = load_chunks()
    chunks = usable_chunks(all_chunks)
    targets = balanced_sample(chunks, args.per_doc, rng)

    print(f"usable chunks : {len(chunks)} of {len(all_chunks)}")
    print(f"target chunks : {len(targets)} ({args.per_doc}/doc)")
    print(f"model         : {args.model}\n")

    kept: list[dict] = []
    rejected: list[dict] = []
    seen_questions: set[str] = set()
    limit = args.dry_run or len(targets)

    for i, chunk in enumerate(targets[:limit], 1):
        parsed = generate(client, args.model, chunk)
        if parsed is None:
            rejected.append({"chunk_id": chunk["chunk_id"], "reason": "no_valid_reply"})
            continue

        question, answer = parsed

        # Quality gate before the grounding gate: a pair that fails either is
        # unusable, but a meta-question is a defect of the prompt rather than
        # of grounding, so it is reported separately.
        bad = quality_reject_reason(question, answer)
        if bad:
            rejected.append({
                "chunk_id": chunk["chunk_id"],
                "reason": bad,
                "question": question,
                "answer": answer,
            })
            continue

        key_q = question.lower().strip()
        if key_q in seen_questions:
            rejected.append({"chunk_id": chunk["chunk_id"], "reason": "duplicate_question"})
            continue

        matched, match_type = match_gold_chunk(answer, [chunk])
        if matched is None:
            # The model stated an answer the passage does not contain. This is
            # the case that manufactured gold would slip through.
            rejected.append({
                "chunk_id": chunk["chunk_id"],
                "reason": f"answer_not_in_chunk({match_type})",
                "question": question,
                "answer": answer,
            })
            continue

        seen_questions.add(key_q)
        kept.append({
            "query_id": f"llm_{len(kept):03d}",
            "question": question,
            "gold_answer": answer,
            "ground_truth": answer,
            "gold_chunk_id": chunk["chunk_id"],
            "gold_doc_id": chunk["doc_id"],
            "question_type": "lexical",
            "difficulty": "medium",
            "should_abstain": False,
            "provenance": {
                "source": "llm_generated_span_verified",
                "model": args.model,
                "match_type": match_type,
            },
        })
        if i % 10 == 0 or i == limit:
            print(f"  [{i}/{limit}] kept={len(kept)} rejected={len(rejected)}")

    if args.dry_run:
        print("\ndry run — nothing written")
        return 0

    if not kept:
        print(
            f"\nAll {len(rejected)} candidate(s) were rejected; nothing written.\n"
            "This is a generation or matching problem, not a success.",
            file=sys.stderr,
        )
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "generated_by": args.model,
        "per_doc_target": args.per_doc,
        "kept": len(kept),
        "rejected": len(rejected),
        "records": kept,
        "rejected_detail": rejected,
    }, indent=2, ensure_ascii=False))

    print(f"\nkept     : {len(kept)}")
    print(f"rejected : {len(rejected)}")
    print(f"written  : {args.out}")
    print(
        "\nReview before merging. This is a fragment, not a dataset:\n"
        "  python scripts/build_eval_dataset_v4.py   # then merge manually"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())