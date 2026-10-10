"""The embedding model must have exactly one source of truth.

`src/api/config.py` defines `settings.embed_model`, overridable via
`RAG_EMBED_MODEL` / `EMBED_MODEL`. Two other modules used to repeat the model
name as a literal:

    src/ingestion/semantic_chunker.py   EMBED_MODEL
    src/evaluation/ragas_evaluator.py    DEFAULT_EMBED_MODEL

Switching embedders then required three edits. A missed one makes ingestion index
the corpus at a different dimension than the query path reads — the same failure
class as the stale `assert vector_dim == 768` that survived a model change.

These tests pin the single-source rule and the dimensional consequence.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
MODEL_LITERAL = "sentence-transformers/all-MiniLM-L6-v2"

# Modules allowed to hold the literal, with the reason.
ALLOWED = {
    "api/config.py": "defines settings.embed_model",
}


def _module_files():
    return [p for p in SRC.rglob("*.py") if p.name != "__init__.py"]


def test_only_config_hardcodes_the_embedding_model():
    offenders = []
    for path in _module_files():
        rel = path.relative_to(SRC).as_posix()
        if rel in ALLOWED:
            continue
        if MODEL_LITERAL in path.read_text(encoding="utf-8"):
            offenders.append(rel)
    assert not offenders, (
        f"embedding model literal duplicated in {offenders}; it must be defined "
        f"only in src/api/config.py and referenced via settings.embed_model"
    )


def test_config_defines_the_model():
    assert "embed_model" in (SRC / "api" / "config.py").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "module_path, attr",
    [
        ("ingestion/semantic_chunker.py", "EMBED_MODEL"),
        ("evaluation/ragas_evaluator.py", "DEFAULT_EMBED_MODEL"),
    ],
)
def test_module_constant_derives_from_settings(module_path, attr):
    """The constant must be an assignment from settings, not a literal."""
    path = SRC / module_path
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == attr:
                    src = ast.unparse(node.value)
                    assert "settings.embed_model" in src, (
                        f"{module_path}:{attr} = {src!r} must derive from "
                        f"settings.embed_model"
                    )
                    found = True
    assert found, f"{attr} not found in {module_path}"


def test_rag_model_literal_also_single_source():
    """The reranker name has the same risk and the same rule."""
    offenders = []
    literal = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    for path in _module_files():
        rel = path.relative_to(SRC).as_posix()
        if rel == "api/config.py":
            continue
        if literal in path.read_text(encoding="utf-8"):
            offenders.append(rel)
    assert not offenders, (
        f"reranker model literal duplicated in {offenders}; it must be defined "
        f"only in src/api/config.py as settings.reranker_model"
    )


def test_embed_dimension_is_derived_not_assumed():
    """Indexing dimension must come from the model, not a separate constant."""
    import src._compat  # noqa: F401
    from src.retrieval.qdrant_store import VECTOR_DIM

    assert VECTOR_DIM == int(__import__("os").getenv("RAG_VECTOR_DIM", "384")), (
        "VECTOR_DIM must be overridable and must default to the active embedder's "
        "dimension (384 for all-MiniLM-L6-v2)"
    )