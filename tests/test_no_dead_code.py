from __future__ import annotations

import ast
import pathlib
import re

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

SRC = [
    p
    for p in (REPO_ROOT / "src").rglob("*.py")
    if "__pycache__" not in p.parts and "target" not in p.parts and p.name != "__init__.py"
]
OTHER = list((REPO_ROOT / "scripts").glob("*.py")) + list((REPO_ROOT / "tests").glob("*.py"))


def _corpus() -> str:
    return "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in SRC + OTHER)


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
    corpus = _corpus()
    total = 0
    for path in SRC:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8", errors="ignore"))):
            if isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef)
            ) and not node.name.startswith("__"):
                total += 1
    dead = len(_unreferenced(corpus))
    ratio = dead / max(total, 1)
    assert ratio <= 0.05, (
        f"{dead}/{total} functions in src/ are unreferenced ({ratio:.1%}); the ceiling is 5%"
    )
