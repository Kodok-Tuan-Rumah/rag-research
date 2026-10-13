"""CI contracts that do not need an LLM, a network, or a Qdrant index.

Replaces the previous `rag-eval` CI job, which pointed
`scripts/eval_ci.py --eval-dir data/adversarial` at a file that has no
`answer`, `contexts` or `ground_truth` field, so the job raised
``ValueError: Evaluation row 0 is not RAGAS-ready`` on every run. It could
never have passed.

Answer-quality measurement needs a Groq key and minutes of API time, so it
belongs in the manual `quality-gate` workflow, not in a per-push gate. What is
worth checking on every push is the part that silently breaks:

1. every module under ``src/`` imports. This is the check that was missing and
   that let two separate import-time failures survive: ragas needing
   ``langchain_community.chat_models.vertexai``, and ``ragas/executor.py``
   calling ``nest_asyncio.apply()`` at module level, which breaks
   ``asyncio.wait_for`` on Python 3.14.
2. the corpus still matches the sha256 recorded in the dataset envelopes
   (§15). The hashes were already there and matched; nothing read them, so any
   writer could have replaced the corpus without anything noticing.
3. every labeled dataset still matches the corpus. A gold chunk id pointing at
   a chunk that no longer exists is an evaluation_data_error, not a retrieval
   failure, and it silently deflates every recall number.
4. every labeled dataset is readable by the evaluator. Validation and
   evaluation disagree about what a gold field is called, so a dataset can be
   valid and still crash run_retrieval_eval.py.

Run:  python scripts/check_ci_contracts.py
Exit: 0 when all four hold, 1 otherwise.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import pathlib
import sys
from typing import Iterable

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src._compat  # noqa: F401  must precede qdrant_client on 3.14

from src.evaluation.gold_data_validator import validate_dataset
from src.evaluation.gold_schema import dataset_records, normalize_gold_chunk_ids

CHUNKS_PATH = REPO_ROOT / "data" / "processed" / "chunks_semantic_v2.json"
DOCUMENTS_PATH = REPO_ROOT / "data" / "processed" / "documents.json"
EVAL_DIR = REPO_ROOT / "data" / "evaluation"

# Datasets the evaluation code can actually read. adversarial_queries.json is
# deliberately absent: it holds queries with `should_abstain`, no reference
# answers, and is used by the retrieval-stage adversarial check instead.
DATASET_GLOBS = ("retrieval_eval_dataset*.json", "retrieval_train_eval.json",
                 "retrieval_holdout_eval.json")


def dataset_paths() -> list[pathlib.Path]:
    paths: list[pathlib.Path] = []
    for pattern in DATASET_GLOBS:
        paths.extend(sorted(EVAL_DIR.glob(pattern)))
    return paths


# envelope key -> corpus file. Resolved at call time, not import time, so the
# module-level CHUNKS_PATH / DOCUMENTS_PATH stay overridable (tests, and any
# caller pointing at an alternative corpus).
_CORPUS_KEYS = ("corpus_chunks_sha256", "corpus_documents_sha256")


def _corpus_files() -> tuple[tuple[str, pathlib.Path], ...]:
    return (("corpus_chunks_sha256", CHUNKS_PATH),
            ("corpus_documents_sha256", DOCUMENTS_PATH))


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_corpus_fingerprint() -> list[str]:
    """Verify the corpus still matches the hash the dataset recorded.

    PROJECT_GOALS.md §15 requires dataset and index versioning. The dataset
    envelopes already carry corpus_chunks_sha256 and corpus_documents_sha256,
    and both match today. Nothing was checking them, so any writer could
    replace the corpus and every baseline recorded against it would silently
    become meaningless.

    The concrete hazard is src/ingestion/chunker.py:469, whose __main__ block
    calls save_chunks(chunks, "data/processed/chunks_semantic_v2.json") — the
    same path the live corpus uses. chunk_semantic emits 582 chunks with
    {doc}_s0000 ids against the live 796 with {doc}_rs_0000 ids, zero overlap,
    so one direct invocation replaces the whole index. Redirecting that write
    addresses the one known writer; this check covers every writer, known and
    unknown.

    A dataset that records no corpus hash is skipped rather than failed, so a
    dataset built against a different corpus is not reported as corruption.
    """
    failures = []
    verified = 0

    for path in dataset_paths():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue

        for key, corpus_path in _corpus_files():
            expected = payload.get(key)
            if not expected:
                continue
            if not corpus_path.exists():
                failures.append(f"{path.name}: records {key} but {corpus_path.name} is missing")
                continue
            actual = _sha256(corpus_path)
            verified += 1
            if actual != expected:
                failures.append(
                    f"{path.name}: {corpus_path.name} does not match {key}\n"
                    f"      expected {expected}\n"
                    f"      actual   {actual}\n"
                    f"      Every metric recorded against this corpus is now invalid. "
                    f"Regenerate the dataset, or restore the corpus from git."
                )

    if verified == 0:
        return ["no dataset records a corpus sha256; fingerprint check is inactive"]
    if failures:
        return failures

    print(f"    verified {verified} hash(es) against the dataset envelopes")
    return []


def src_modules() -> Iterable[str]:
    for path in sorted((REPO_ROOT / "src").rglob("*.py")):
        if "__pycache__" in path.parts or path.name == "__init__.py":
            continue
        yield ".".join(path.relative_to(REPO_ROOT).with_suffix("").parts)


def check_imports() -> list[str]:
    failures = []
    for name in src_modules():
        try:
            importlib.import_module(name)
        except Exception as exc:
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
    return failures


def check_datasets(chunks: list[dict]) -> list[str]:
    failures = []
    if not CHUNKS_PATH.exists():
        return [f"corpus missing: {CHUNKS_PATH}"]

    paths = dataset_paths()
    if not paths:
        return [f"no dataset matched {DATASET_GLOBS} in {EVAL_DIR}"]

    for path in paths:
        records = dataset_records(json.loads(path.read_text(encoding="utf-8")))
        report = validate_dataset(records, chunks, require_answer=True)
        summary = report["summary"]
        invalid = summary["TOTAL_QA"] - summary["VALID_QA"]
        status = "ok" if invalid == 0 else "FAIL"
        print(f"    {path.name:38s} {status:4s} "
              f"valid={summary['VALID_QA']}/{summary['TOTAL_QA']}")
        if invalid:
            detail = {k: v for k, v in summary.items() if k.startswith(("MISSING", "INVALID", "DUPLICATE")) and v}
            failures.append(f"{path.name}: {invalid} invalid rows {detail}")
    return failures


def check_evaluable_schema() -> list[str]:
    """Every row must satisfy the reader the evaluator actually uses.

    validate_dataset() and normalize_gold_chunk_ids() answer different
    questions. The validator accepts the legacy aliases ``source_chunk``,
    ``source_chunk_id``, ``chunk_id`` and ``gold_chunk`` so old datasets can
    still be audited; the evaluator accepts only ``gold_chunk_id`` or
    ``gold_chunk_ids``. A row using an alias therefore passes validation and
    then raises inside run_retrieval_eval.py, which calls
    normalize_gold_chunk_ids() with no try/except around it — so one such row
    aborts the entire evaluation run.

    That gap made this gate report ok for a dataset the evaluator cannot read.
    This check closes it by asserting the evaluator's contract directly.
    """
    failures = []
    paths = dataset_paths()
    if not paths:
        return [f"no dataset matched {DATASET_GLOBS} in {EVAL_DIR}"]

    for path in paths:
        records = dataset_records(json.loads(path.read_text(encoding="utf-8")))
        broken = []
        for index, record in enumerate(records):
            try:
                normalize_gold_chunk_ids(record)
            except ValueError as exc:
                row_id = record.get("query_id", f"index {index}")
                broken.append(f"{row_id} ({exc})")
        status = "ok" if not broken else "FAIL"
        print(f"    {path.name:38s} {status:4s} readable={len(records) - len(broken)}/{len(records)}")
        if broken:
            failures.append(
                f"{path.name}: {len(broken)} row(s) unreadable by the evaluator -> "
                + "; ".join(broken[:3])
            )
    return failures


def check_query_leakage(chunks: list, embedder=None, report_path: pathlib.Path | None = None) -> list[str]:
    """PROJECT_GOALS.md §10.2 — reject samples whose question restates its gold chunk.

    A sample with similarity > 0.9 needs no ranking skill to retrieve: returning
    that chunk scores Recall@10 = 1.0. It reports perfect retrieval for a
    system that cannot retrieve, and that number is inherited by every metric
    computed from the set.

    The embedder is injected so this gate can be exercised without loading a
    model. It loads one on first call, and reports the failure rather than
    crashing CI if the model is unavailable — an unrunnable gate would be
    indistinguishable from a passing one.
    """
    if not chunks:
        return []

    try:
        from src.evaluation.query_leakage import dataset_leakage
    except Exception as exc:  # noqa: BLE001
        return [f"query leakage check could not be imported: {exc}"]

    failures: list[str] = []
    # Built once, reused for every dataset. default_embedder() loads the model
    # on each call, and six datasets meant six loads — roughly 40 seconds of
    # CI time for the same weights.
    try:
        from src.evaluation.query_leakage import default_embedder, dataset_leakage

        embedder = embedder or default_embedder()
    except Exception as exc:  # noqa: BLE001
        return [f"query leakage check could not be imported: {exc}"]

    reports: list[dict] = []
    for path in dataset_paths():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue  # dataset_records reports the parse failure elsewhere

        try:
            report = dataset_leakage(payload, chunks, embedder=embedder)
        except Exception as exc:  # noqa: BLE001
            failures.append(
                f"{path.name}: leakage check did not run ({type(exc).__name__}: "
                f"{str(exc)[:120]}). §10.2 is unenforced while this happens."
            )
            continue

        report["dataset"] = path.name
        reports.append(report)
        print(f"    {path.name:38s} scored={report['scored']:3d} "
              f"leaked={report['leaked']:2d} max_sim={report['max_similarity']}")
        if report["leaked"]:
            names = ", ".join(
                f"{d['query_id']}({d['similarity']})" for d in report["detail"]
                if d["leaked"]
            )
            failures.append(
                f"{path.name}: {report['leaked']} row(s) exceed the §10.2 "
                f"similarity threshold {report['threshold']} -> {names}. "
                f"A question that restates its gold chunk retrieves trivially, "
                f"so it inflates recall without measuring retrieval."
            )

    if report_path is not None and reports:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps({
            "rule": "PROJECT_GOALS.md §10.2 — similarity(query, gold_chunk) > 0.9",
            "datasets_checked": len(reports),
            "total_scored": sum(r["scored"] for r in reports),
            "total_leaked": sum(r["leaked"] for r in reports),
            "passing": not failures,
            "reports": reports,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
    return failures


def main() -> int:
    print("[CI contracts] 1/5 import surface of src/")
    import_failures = check_imports()
    total = len(list(src_modules()))
    print(f"    {total - len(import_failures)}/{total} modules import cleanly")
    for line in import_failures:
        print(f"    FAIL {line}", file=sys.stderr)

    print("[CI contracts] 2/5 corpus still matches the dataset fingerprints")
    fingerprint_failures = check_corpus_fingerprint()
    for line in fingerprint_failures:
        print(f"    FAIL {line}", file=sys.stderr)

    print("[CI contracts] 3/5 labeled datasets match the corpus")
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8")) if CHUNKS_PATH.exists() else []
    dataset_failures = check_datasets(chunks)
    for line in dataset_failures:
        print(f"    FAIL {line}", file=sys.stderr)

    print("[CI contracts] 4/5 datasets are readable by the evaluator")
    schema_failures = check_evaluable_schema()
    for line in schema_failures:
        print(f"    FAIL {line}", file=sys.stderr)

    print("[CI contracts] 5/5 no query restates its gold chunk (§10.2)")
    leakage_report = REPO_ROOT / "results" / "metrics" / "query_leakage_report.json"
    leakage_failures = check_query_leakage(chunks, report_path=leakage_report)
    for line in leakage_failures:
        print(f"    FAIL {line}", file=sys.stderr)

    failures = (import_failures + fingerprint_failures
                + dataset_failures + schema_failures + leakage_failures)
    if failures:
        print(f"\n[CI contracts] FAILED ({len(failures)} problem(s))", file=sys.stderr)
        return 1
    print("\n[CI contracts] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())