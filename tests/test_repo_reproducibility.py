"""Files the project cannot be reproduced without must be tracked in git.

`data/processed/*` was in .gitignore while the comment above it promised to keep
the processed JSON. Nothing under data/processed/ was tracked, which meant a
fresh clone had no corpus at all:

- `config.py:bm25_chunks_path` points at `chunks_semantic_v2.json`; when absent,
  `MasterHybridRetriever.__init__` leaves `self._bm25 = None` and
  `bm25_search` returns `[]` — no error, just silently degraded retrieval.
- The corpus sha256 recorded as `corpus_chunks_sha256` in the v2 dataset
  envelope had nothing to verify against, so `check_ci_contracts.py` was
  guarding a file a clone would not have.

The 796 chunks are ~1 MB. That is small next to the cost of a baseline that
cannot be re-measured.

This test runs `git` because the failure is a repository-configuration failure,
not an import or runtime one. It skips when git is unavailable or the checkout
is not a repository, so it cannot break a non-git environment.
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

# Path -> why a fresh clone breaks without it
REQUIRED_TRACKED = {
    "data/processed/chunks_semantic_v2.json":
        "the live corpus: BM25 index, reranker input, and the sha256 recorded "
        "as corpus_chunks_sha256 in the v2 dataset envelope",
    "data/processed/documents.json":
        "source documents; semantic_chunker.py and document_loader.py read it",
    "data/evaluation/retrieval_eval_dataset_v2.json":
        "the canonical labeled set every recorded baseline was measured on",
    "data/evaluation/retrieval_eval_dataset.json": "v1 labeled set, still referenced",
    "data/evaluation/retrieval_eval_dataset_v3.json": "v3 labeled set, still referenced",
    "data/evaluation/retrieval_train_eval.json": "§10.1 train split",
    "data/evaluation/retrieval_holdout_eval.json": "§10.1 holdout split",
    "data/adversarial/adversarial_queries.json": "used by run_retrieval_eval.py --include-adversarial",
    "scripts/MANIFEST.json": "the classification every scripts/*.py must appear in",
    "results/metrics/k1_baseline_freeze.json":
        "the §14 reference every before/after comparison is made against",
}

MUST_STAY_IGNORED = {
    ".env": "API keys",
    "data/qdrant_storage": "3.8 MB local vector store, rebuildable",
    "data/raw": "source PDFs, tens of MB",
    "src/ingestion/semantic_chunker_rust/target": "255 MB Rust build directory",
}


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True
    )


pytestmark = pytest.mark.skipif(
    shutil.which("git") is None or not (REPO_ROOT / ".git").exists(),
    reason="not a git checkout",
)


def _tracked_files() -> set[str]:
    out = _git("ls-files").stdout
    return {line for line in out.splitlines() if line}


@pytest.mark.parametrize("path,why", sorted(REQUIRED_TRACKED.items()))
def test_reproducibility_critical_file_is_tracked(path, why):
    tracked = _tracked_files()
    assert path in tracked, (
        f"{path} is not tracked by git. {why}. A fresh clone would be missing "
        f"it. Add it with `git add -f {path}` and check .gitignore."
    )


@pytest.mark.parametrize("path,why", sorted(MUST_STAY_IGNORED.items()))
def test_large_or_secret_path_is_ignored(path, why):
    result = _git("check-ignore", "-q", path)
    assert result.returncode == 0, (
        f"{path} is not ignored ({why}). Check .gitignore before committing."
    )


def test_gitignore_does_not_contradict_itself_about_processed_json():
    """The original defect: a comment promising what the rule prevented."""
    text = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "Keep processed datasets" in text, "the intent comment was removed"
    assert "!data/processed/*.json" in text, (
        "the negation is what keeps the corpus. Without it the comment above it "
        "is false and the corpus is untracked."
    )


def test_no_global_pdf_rule():
    """`*.pdf` anywhere would ignore a document that belongs in the repo."""
    text = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert not any(
        line.strip() == "*.pdf" for line in text.splitlines()
    ), "a global *.pdf rule ignores every PDF in the repo; scope it to data/raw/"


def test_env_is_still_ignored_after_the_extra_patterns():
    """`.env.*` and `*.env` were added; a plain `.env` must still match."""
    assert _git("check-ignore", "-q", ".env").returncode == 0
    assert _git("check-ignore", "-q", ".env.production").returncode == 0