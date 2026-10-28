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

from scripts.build_eval_dataset_v4 import match_gold_chunk

CHUNKS_PATH = ROOT / "data" / "processed" / "chunks_semantic_v2.json"
OUT_PATH = ROOT / "data" / "evaluation" / "llm_generated_questions.json"

DEFAULT_MODEL = "openai/gpt-oss-20b"
SEED = 42
MAX_TOKENS = 2048
MAX_ESCALATION = 4096
MIN_INTERVAL = 8.0
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

_META_QUESTION = re.compile(
    r"\b(passage|passages|excerpt|the text|this text|the document|this document"
    r"|the paper|this paper|the article|this article|the chunk|this chunk"
    r"|mentioned|referred to|described in|stated in|according to the (text|paper|passage))\b",
    re.IGNORECASE,
)

# "Fig. 1" answers the question "which figure" and nothing else. It is a
_LABEL_ONLY_ANSWER = re.compile(
    r"^\s*[\(\[]?\s*(fig|figure|table|tab|eq|equation|section|sec|chapter|appendix"
    r"|scheme|algorithm|listing)\s*\.?\s*\d*\s*[\)\]]?\s*[:.]?\s*$",
    re.IGNORECASE,
)


def quality_reject_reason(question: str, answer: str) -> str | None:
    q, a = question.strip(), answer.strip()

    if _META_QUESTION.search(q):
        return "meta_question_references_the_document"
    if _LABEL_ONLY_ANSWER.match(a):
        return "answer_is_a_figure_or_table_pointer"
    if not any(ch.isalnum() for ch in a):
        return "answer_has_no_substantive_content"
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


def balanced_pairs(chunks: list[dict], per_doc: int, rng: random.Random) -> list[tuple[dict, dict]]:
    by_doc: dict[str, list[dict]] = {}
    for chunk in chunks:
        by_doc.setdefault(chunk["doc_id"], []).append(chunk)

    pairs: list[tuple[dict, dict]] = []
    for doc_id in sorted(by_doc):
        pool = by_doc[doc_id]
        rng.shuffle(pool)
        taken = 0
        for i in range(len(pool) - 1):
            if taken >= per_doc:
                break
            a, b = pool[i], pool[i + 1]
            if _jaccard(a.get("text", ""), b.get("text", "")) > 0.5:
                continue
            pairs.append((a, b))
            taken += 1
    rng.shuffle(pairs)
    return pairs


def _jaccard(a: str, b: str) -> float:
    ta = set(re.findall(r"\w+", (a or "").lower()))
    tb = set(re.findall(r"\w+", (b or "").lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


MULTIHOP_PROMPT = """You are writing evaluation questions for a retrieval system.

Here are TWO separate passages from the SAME research paper:

<passage id="A">
{chunk_a}
</passage>

<passage id="B">
{chunk_b}
</passage>

Write ONE question that CANNOT be answered from either passage alone, and
requires BOTH of them.

Rules:
- The answer must combine a fact from A with a fact from B.
- If one passage alone would answer it, pick a different question.
- Do NOT use outside knowledge. Do NOT invent facts.
- The question must be about the SUBJECT MATTER, never about the text itself.
  Banned: "the passage", "the text", "this document", "this paper", "mentioned in",
  "referred to", "described in", "according to the text".
- The answer must be a short fact or phrase stated across A and B.
- BANNED answers: pointers into the document's furniture ("Fig. 1", "Table 2").

Reply with JSON only, exactly these keys:
{{"question": "...", "answer": "...", "uses": "A and B"}}"""


def parse_multihop_reply(text: str) -> tuple[str, str] | None:
    return parse_reply(text)


def _sleep_for_api_error(exc: Exception, attempt: int) -> float:
    if type(exc).__name__ == "RateLimitError":
        return 15.0 * (2**attempt)
    return 2.0 * (attempt + 1)


def _complete(client, model: str, prompt: str, retries: int = 2):
    budget = MAX_TOKENS
    for _ in range(retries + 1):
        time.sleep(MIN_INTERVAL)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=budget,
            temperature=0.4,
        )
        choice = response.choices[0]
        if choice.finish_reason == "length":
            if budget >= MAX_ESCALATION:
                print(f"    still truncated at {budget}; giving up on this prompt")
                return "", "length"
            budget = min(budget * 2, MAX_ESCALATION)
            print(f"    truncated at {budget // 2}, retrying with {budget}")
            continue
        return choice.message.content or "", choice.finish_reason
    return "", "length"


def generate_multihop(
    client, model: str, pair: tuple[dict, dict], retries: int = 2
) -> tuple[str, str] | None:
    chunk_a, chunk_b = pair
    prompt = MULTIHOP_PROMPT.format(
        chunk_a=(chunk_a.get("text") or "")[:1200],
        chunk_b=(chunk_b.get("text") or "")[:1200],
    )
    for attempt in range(retries + 1):
        try:
            content, finish = _complete(client, model, prompt)
        except Exception as exc:  # noqa: BLE001
            wait = _sleep_for_api_error(exc, attempt)
            print(f"    API error ({type(exc).__name__}): {str(exc)[:90]} — retry in {wait:.0f}s")
            time.sleep(wait)
            continue
        if finish == "length":
            return None
        parsed = parse_multihop_reply(content)
        if parsed:
            return parsed
    return None


def parse_reply(text: str) -> tuple[str, str] | None:
    if not text:
        return None
    match = re.search(r"\{.*\}", text, re.DOTALL)
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
            wait = _sleep_for_api_error(exc, attempt)
            print(f"    API error ({type(exc).__name__}): {str(exc)[:90]} — retry in {wait:.0f}s")
            time.sleep(wait)
            continue

        choice = response.choices[0]
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
    parser.add_argument(
        "--per-doc",
        type=int,
        default=30,
        help="questions per document (7 documents in this corpus)",
    )
    parser.add_argument(
        "--mode",
        choices=["lexical", "multihop"],
        default="lexical",
        help="lexical = one chunk per question; "
        "multihop = a same-document pair, answer must need both",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    parser.add_argument(
        "--dry-run", type=int, default=0, help="stop after N chunks and write nothing useful"
    )
    args = parser.parse_args()

    key = os.getenv("GROQ_API_KEY")
    if not key:
        print("GROQ_API_KEY is not set.", file=sys.stderr)
        return 2

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
    if args.mode == "multihop":
        targets = balanced_pairs(chunks, args.per_doc, rng)
        label = "pairs"
    else:
        targets = balanced_sample(chunks, args.per_doc, rng)
        label = "chunks"

    print(f"usable chunks : {len(chunks)} of {len(all_chunks)}")
    print(f"mode          : {args.mode}")
    print(f"target {label} : {len(targets)} ({args.per_doc}/doc)")
    print(f"model         : {args.model}\n")

    kept: list[dict] = []
    rejected: list[dict] = []
    seen_questions: set[str] = set()
    limit = args.dry_run or len(targets)

    multihop = args.mode == "multihop"
    if multihop:
        for pair in targets[:limit]:
            parsed = generate_multihop(client, args.model, pair)
            if parsed is None:
                rejected.append(
                    {
                        "reason": "no_valid_reply",
                        "chunks": [c["chunk_id"] for c in pair],
                    }
                )
                continue

            question, answer = parsed
            key_q = question.lower().strip()
            if key_q in seen_questions:
                rejected.append({"reason": "duplicate_question", "question": question})
                continue

            bad = quality_reject_reason(question, answer)
            if bad:
                rejected.append({"reason": bad, "question": question, "answer": answer})
                continue

            # Grounding: the answer must be supported by the evidence handed to
            # the model, which for this mode is the pair.
            matched, match_type = match_gold_chunk(answer, list(pair))
            if matched is None:
                rejected.append(
                    {
                        "reason": f"answer_not_in_evidence({match_type})",
                        "question": question,
                        "answer": answer,
                    }
                )
                continue

            alone_a, _ = match_gold_chunk(answer, [pair[0]])
            alone_b, _ = match_gold_chunk(answer, [pair[1]])
            single_chunk = alone_a is not None or alone_b is not None
            if single_chunk:
                rejected.append(
                    {
                        "reason": "answer_resolvable_from_one_chunk",
                        "question": question,
                        "answer": answer,
                        "chunks": [c["chunk_id"] for c in pair],
                    }
                )
                continue

            seen_questions.add(key_q)
            kept.append(
                {
                    "query_id": f"mh_{len(kept):03d}",
                    "question": question,
                    "gold_answer": answer,
                    "ground_truth": answer,
                    "gold_chunk_ids": [pair[0]["chunk_id"], pair[1]["chunk_id"]],
                    "gold_doc_id": pair[0]["doc_id"],
                    "question_type": "multihop",
                    "difficulty": "hard",
                    "should_abstain": False,
                    "provenance": {
                        "source": "llm_multihop_pair_verified",
                        "model": args.model,
                        "match_type": match_type,
                        "single_chunk_resolvable": False,
                    },
                }
            )
            if len(kept) % 10 == 0 or len(kept) == limit:
                print(f"  [{len(kept)}/{limit}] kept={len(kept)} rejected={len(rejected)}")
        return _finish(args, kept, rejected, limit)

    for i, chunk in enumerate(targets[:limit], 1):
        parsed = generate(client, args.model, chunk)
        if parsed is None:
            rejected.append({"chunk_id": chunk["chunk_id"], "reason": "no_valid_reply"})
            continue

        question, answer = parsed

        bad = quality_reject_reason(question, answer)
        if bad:
            rejected.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "reason": bad,
                    "question": question,
                    "answer": answer,
                }
            )
            continue

        key_q = question.lower().strip()
        if key_q in seen_questions:
            rejected.append({"chunk_id": chunk["chunk_id"], "reason": "duplicate_question"})
            continue

        matched, match_type = match_gold_chunk(answer, [chunk])
        if matched is None:
            rejected.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "reason": f"answer_not_in_chunk({match_type})",
                    "question": question,
                    "answer": answer,
                }
            )
            continue

        seen_questions.add(key_q)
        kept.append(
            {
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
            }
        )
        if i % 10 == 0 or i == limit:
            print(f"  [{i}/{limit}] kept={len(kept)} rejected={len(rejected)}")

    return _finish(args, kept, rejected, limit)


def _finish(args, kept: list[dict], rejected: list[dict], limit: int) -> int:
    if args.dry_run:
        print("\n[dry run] — nothing written")
        return 0

    if not kept:
        print(
            f"\nAll {len(rejected)} candidate(s) were rejected; nothing written.\n"
            "This is a generation or matching problem, not a success.",
            file=sys.stderr,
        )
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "generated_by": args.model,
                "mode": args.mode,
                "per_doc_target": args.per_doc,
                "kept": len(kept),
                "rejected": len(rejected),
                "records": kept,
                "rejected_detail": rejected,
            },
            indent=2,
            ensure_ascii=False,
        )
    )

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
