"""No unreferenced functions in src/.

The 7.2% figure was not alarming by itself. What mattered was that nothing
prevented it from growing: 261 tests never noticed a single dead function,
because they test behaviour and a function nobody calls produces no behaviour.

Thirteen were removed on 2026-10-11 (`extend_chunks`, `_sigmoid`,
`doc_level_hit`, `save_response`, `_hard_split`, `remove_symbol_noise`,
`is_fragment_chunk`, `passes_minimum_filter_text`, `_expand_queries`,
`_generate_hyde`, `_is_title_or_author_chunk`, `_is_valid_section`, and the
module-level `generate`). This pins the count at zero.

The scan is name-based over src/ plus scripts/ and tests/. That is deliberately
crude: it cannot prove a function is dead, only that nothing mentions its name.
A false positive is resolved by deleting a genuinely unused function, which is
the correct outcome either way.

Deliberately excluded: dunder methods (`__post_init__` and friends are invoked
by the language, not by name), and nested helpers whose names collide with
callers elsewhere in the corpus.
"""

from __future__ import annotations

import ast
import pathlib
import re

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

SRC = [
    p for p in (REPO_ROOT / "src").rglob("*.py")
    if "__pycache__" not in p.parts
    and "target" not in p.parts
    and p.name != "__init__.py"
]
OTHER = list((REPO_ROOT / "scripts").glob("*.py")) + list((REPO_ROOT / "tests").glob("*.py"))


def _corpus() -> str:
    return "\n".join(
        p.read_text(encoding="utf-8", errors="ignore") for p in SRC + OTHER
    )


def _unreferenced(corpus: str) -> list[str]:
    found: list[str] = []
    for path in SRC:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name.startswith("__"):
                continue  # invoked by the language, never by name
            if len(re.findall(rf"\b{re.escape(node.name)}\b", corpus)) <= 1:
                rel = path.relative_to(REPO_ROOT)
                found.append(f"{rel}:{node.lineno} {node.name}()")
    return sorted(found)


def test_no_unreferenced_functions_in_src():
    dead = _unreferenced(_corpus())
    assert not dead, (
        f"{len(dead)} function(s) in src/ are never referenced by anything in "
        f"src/, scripts/ or tests/:\n  " + "\n  ".join(dead)
    )


def test_dead_code_ratio_stays_low():
    """A soft tripwire. Not every unused function is worth deleting, but a
    large jump means something stopped being called."""
    corpus = _corpus()
    total = 0
    for path in SRC:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8", errors="ignore"))):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("__"):
                total += 1
    dead = len(_unreferenced(corpus))
    ratio = dead / max(total, 1)
    assert ratio <= 0.05, (
        f"{dead}/{total} functions in src/ are unreferenced ({ratio:.1%}); "
        f"the ceiling is 5%"
    )