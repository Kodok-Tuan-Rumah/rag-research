"""Every script that touches the API must load .env before it runs.

`scripts/run_eval.py` was rewritten on 2026-10-11 and the rewrite dropped
`load_dotenv()`. Every other script that needs the key kept it, so this one was
the only eval entry point that did not read its credentials.

Nothing caught it:

- the test suite does not execute scripts, so 305 green tests said nothing
- `--stage pipeline` still worked, because retrieval needs no API key and the
  generator holds its key from a previous successful call in the same process
- `--stage ragas` failed with "GROQ_API_KEY tidak ditemukan", which looks like a
  missing key in the environment rather than a missing line in the script

The failure mode is the dangerous kind: it depends on process history. Run
`run_eval.py` standalone and it dies; run it after something else has already
loaded the environment and it appears to work.
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"

# Scripts that construct a client needing an API key at import or run time.
REQUIRES_ENV = (
    "run_eval.py",
    "run_eval_phase2.py",
    "run_single_query.py",
    "diagnose_generation_flakiness.py",
    "coverage_metric.py",
    "eval_ci.py",
    "test_manual.py",
)


@pytest.mark.parametrize("name", REQUIRES_ENV)
def test_script_loads_dotenv(name: str):
    """Presence of load_dotenv, and early enough to matter.

    The ordering check only applies when the script builds a client directly.
    Scripts that reach the key indirectly — eval_ci.py imports RAGASEvaluator,
    coverage_metric.py reads GROQ_API_KEY inside a method — cannot be checked
    that way, and demanding it produced false failures on scripts that were
    already correct.
    """
    path = SCRIPTS / name
    if not path.exists():
        pytest.skip(f"{name} not present")
    source = path.read_text(encoding="utf-8")

    assert "load_dotenv" in source, (
        f"scripts/{name} never calls load_dotenv(), so GROQ_API_KEY is missing "
        f"unless some earlier import happened to load it first"
    )

    load_at = source.index("load_dotenv")
    build_at = re.search(r"Groq\(|QdrantVectorStore\(|ChatGroq\(", source)
    if build_at is None:
        return  # the key is reached indirectly; presence is the check that holds
    assert load_at < build_at.start(), (
        f"scripts/{name} builds a client before calling load_dotenv()"
    )


def test_run_eval_works_standalone():
    """run_eval.py must see GROQ_API_KEY when the environment is cold.

    Uses a report path that does not exist. That makes the script fail with
    "report not found" — which only happens after the environment has been
    loaded. If load_dotenv() were missing, the failure would instead be
    "GROQ_API_KEY tidak ditemukan", and the two messages are easy to tell apart.

    This deliberately does not run the scoring itself: one sample across five
    metrics takes minutes, which is far too slow for a unit test. The point here
    is the credential, not the measurement.
    """
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": "/tmp",
        "PYTHONPATH": str(REPO_ROOT),
    }
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_eval.py",
            "--stage", "ragas",
            "--from-report", str(REPO_ROOT / "tests" / "fixtures" / "does_not_exist.json"),
            "--out-ragas", "/tmp/opencode/standalone_ragas.json",
            "--out-report", "/tmp/opencode/standalone_report.json",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    output = proc.stdout + proc.stderr
    assert "GROQ_API_KEY tidak ditemukan" not in output, (
        "run_eval.py cannot see GROQ_API_KEY when the environment is cold. "
        "Add load_dotenv() at the top of the module.\n"
        f"output: {output[-600:]}"
    )
    assert "report not found" in output, (
        "expected the script to reach the report check, which only happens "
        "after the environment is loaded\n"
        f"output: {output[-600:]}"
    )


def test_a_pipeline_fixture_exists_for_that_standalone_test():
    path = REPO_ROOT / "tests" / "fixtures" / "pipeline_report.json"
    if not path.exists():
        pytest.skip(
            "no pipeline fixture yet; create one with "
            "`python3 scripts/run_eval.py --stage pipeline --query-ids q11`"
        )
    payload = __import__("json").loads(path.read_text(encoding="utf-8"))
    rows = payload["queries"]
    assert any(r.get("answered") and r.get("contexts") for r in rows), (
        "the fixture must contain at least one answered row with contexts, "
        "otherwise --stage ragas scores nothing"
    )