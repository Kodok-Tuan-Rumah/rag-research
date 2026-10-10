"""RAGAS evaluation must be importable.

`src/evaluation/ragas_evaluator.py` could not be imported at all:

    ragas/llms/base.py:8
        from langchain_community.chat_models.vertexai import ChatVertexAI

That module was removed in langchain-community 0.4.2. Because ragas declares no
upper bound, a clean install resolves the newest release and reproduces the
breakage every time.

It stayed invisible because nothing in the suite imported the module, and
because a 0-import scan of the repo listed langchain-community as "unused" —
a conclusion that does not hold for transitive dependencies.

`langchain-community==0.4.1` is the last release that still ships
`chat_models/vertexai.py`, and it accepts langchain-core>=1.0.1, satisfied by the
pinned 1.6.6.

PROJECT_GOALS.md §2.1 requires Faithfulness, Answer Relevancy and Hallucination
Rate, all RAGAS-based. Phase 2 requires `run_eval.py` to run end to end. Both
are blocked while this import fails.
"""
from __future__ import annotations

import pathlib
import re
import zipfile

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
LLM_REQUIREMENTS = REPO_ROOT / "requirements" / "llm.txt"
TOP_REQUIREMENTS = REPO_ROOT / "requirements.txt"


def _pinned_version(text: str, package: str) -> str | None:
    m = re.search(rf"^{re.escape(package)}==([^\s#]+)", text, re.M)
    return m.group(1) if m else None


def test_langchain_community_is_pinned_in_both_files():
    """An unpinned transitive dependency reproduces the breakage on every install."""
    for path in (TOP_REQUIREMENTS, LLM_REQUIREMENTS):
        text = path.read_text(encoding="utf-8")
        assert _pinned_version(text, "langchain-community"), (
            f"{path.name} must pin langchain-community; ragas imports "
            f"langchain_community.chat_models.vertexai and resolving the newest "
            f"release breaks it"
        )


def test_pinned_version_is_not_the_breaking_one():
    """0.4.2 removed chat_models/vertexai."""
    version = _pinned_version(TOP_REQUIREMENTS.read_text(encoding="utf-8"),
                             "langchain-community")
    assert version != "0.4.2", (
        "langchain-community 0.4.2 removed chat_models/vertexai.py, which "
        "ragas 0.2.15 imports"
    )


def test_installed_version_still_ships_the_module():
    from importlib.metadata import version

    try:
        import langchain_community.chat_models.vertexai  # noqa: F401
    except ImportError as exc:
        pytest.fail(
            f"langchain-community {version('langchain-community')} does not provide "
            f"chat_models.vertexai, which ragas imports: {exc}. Pin it to 0.4.1."
        )


def test_ragas_evaluator_is_importable():
    """The module must import. This is the test that was missing."""
    import src._compat  # noqa: F401  (must precede qdrant_client on 3.14)
    import src.evaluation.ragas_evaluator as evaluator

    assert hasattr(evaluator, "RAGASEvaluator")


def test_run_eval_script_imports():
    """Phase 2 exit criterion: run_eval.py must load its module body cleanly.

    It pulls in src.evaluation.ragas_evaluator, so this is the check that would
    have caught the breakage. An earlier draft of this test expected an
    exception and passed vacuously once the import was fixed.
    """
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "run_eval_probe", REPO_ROOT / "scripts" / "run_eval.py"
    )
    assert spec is not None and spec.loader is not None

    # run_single_query.py inserts the repo root onto sys.path; run_eval.py relies
    # on being invoked from the repo root, so make the import explicit here.
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert hasattr(module, "RAGASEvaluator"), (
        "run_eval.py loaded but does not expose RAGASEvaluator"
    )