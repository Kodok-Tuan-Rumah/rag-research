"""README numbers must match reality.

README.md claimed `166 passed` and `119 artifacts` while the suite was at 206
and results/metrics/ held 119 files. The test count was wrong in three places
and went stale unnoticed because nothing read it. PROJECT_GOALS.md §14 requires
measuring before and after every change; a document that misreports the
measurement undermines every other number beside it.

The check is cheap (pytest --collect-only, no test execution) and it is the
only thing that stops this recurring.
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"
METRICS_DIR = REPO_ROOT / "results" / "metrics"


def _collect_count() -> int:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "--collect-only", "-q"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    match = re.search(r"(\d+) tests? collected", proc.stdout)
    if not match:
        raise AssertionError(
            f"could not read the test count from pytest:\n{proc.stdout[-800:]}\n{proc.stderr[-400:]}"
        )
    return int(match.group(1))


def _claimed(pattern: str, text: str) -> set[int]:
    return {int(n) for n in re.findall(pattern, text)}


# "K7.10-K7.12 artifacts" is a version range, not a count. The lookbehind keeps
# a number that is part of an identifier from being read as a quantity.
_COUNT = r"(?<![\w.\-–])(\d+)\s+"


def test_readme_test_count_is_current():
    actual = _collect_count()
    claimed = _claimed(_COUNT + r"passed", README.read_text(encoding="utf-8"))
    assert claimed, "README no longer states a test count; keep the baseline line"

    # A tolerance, not equality: this very test changes the number it checks,
    # so demanding exactness would make every added test break CI. The failure
    # that matters is gross staleness — README sat at 166 while the suite was
    # at 206, understating it by 20%.
    limit = max(2, actual * 0.05)
    worst = max(claimed)
    assert abs(worst - actual) <= limit, (
        f"README claims {sorted(claimed)} passed but pytest collects {actual}. "
        f"A tolerance of {limit:.0f} applies; anything beyond that is stale."
    )


def test_readme_artifact_count_is_current():
    actual = len(list(METRICS_DIR.iterdir()))
    claimed = _claimed(_COUNT + r"artifacts", README.read_text(encoding="utf-8"))
    assert claimed, "README no longer states an artifact count"
    wrong = sorted(n for n in claimed if n != actual)
    assert not wrong, (
        f"README claims {sorted(claimed)} artifacts but results/metrics/ holds "
        f"{actual}. Stale: {wrong}"
    )


def test_readme_test_command_is_the_one_that_runs():
    """`--ignore` flags came back once; the command must stay plain."""
    text = README.read_text(encoding="utf-8")
    for line in text.splitlines():
        if "pytest" in line and "--ignore" in line:
            pytest.fail(f"README suggests a filtered pytest command: {line.strip()}")