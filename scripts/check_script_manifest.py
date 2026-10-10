"""Generate and verify the script manifest.

scripts/ is 8,453 lines against 5,281 in src/ — the operational surface is
larger than the product. 29 of 38 scripts were referenced by no document, CI
workflow or Makefile target, and nothing ever noticed, because 248 tests
execute no script at all. The pile-up is not caused by copy-paste; exact
cross-layer duplication is only 3 cases. It is caused by scripts being written,
run once, and never revisited.

The recurrence guard is this manifest. Every script must declare what it is,
and CI fails when one has no entry. An unclassified script is the condition
that produced 124 artifacts for a 14-query dataset.

    python scripts/check_script_manifest.py            # verify (CI uses this)
    python scripts/check_script_manifest.py --write    # regenerate
"""

from __future__ import annotations

import argparse
import ast
import json
import pathlib
import re
import sys
from typing import Iterable

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCRIPTS_DIR = REPO_ROOT / "scripts"
MANIFEST_PATH = REPO_ROOT / "scripts" / "MANIFEST.json"
ARCHIVED_DIR = REPO_ROOT / "archive"

STATUSES = ("canonical", "library", "diagnostic", "forensic", "dead")

# Statuses assigned by hand, because they encode intent that no static scan can
# recover. Anything unlisted here falls back to "diagnostic" and must be
# reviewed.
STATUS_OVERRIDES = {
    # The documented entrypoints. Anything here should be reachable from
    # README.md, a Makefile target, or a workflow.
    "run_eval.py": "canonical",
    "run_retrieval_eval.py": "canonical",
    "run_single_query.py": "canonical",
    "check_ci_contracts.py": "canonical",
    "check_script_manifest.py": "canonical",
    "measure_reference_impact.py": "canonical",
    # Imported by tests for its metric functions (normalize, exact_match,
    # token_f1). Its CLI entrypoint is dead but the module is not.
    "run_eval_phase2.py": "library",
    # Reads data/processed/holdout_eval_v2.json, which no longer exists. The
    # functions are still used; only the entrypoint is broken.
    "coverage_metric.py": "diagnostic",
    "eval_ci.py": "diagnostic",
}

# Missing data a script is allowed to reference, and why. Anything not listed
# here is treated as a defect.
ACKNOWLEDGED_DEAD_PATHS = {
    "run_single_query.py": (
        # Only reachable behind --from_gt. Its default and --dataset paths work.
        "data/processed/ground_truth_qa.json",
    ),
}

_DOC_REFS = None


def doc_references() -> str:
    global _DOC_REFS
    if _DOC_REFS is None:
        parts = []
        for pattern in ("*.md", "Makefile", "scripts/bootstrap_env.sh"):
            for p in REPO_ROOT.glob(pattern):
                parts.append(p.read_text(encoding="utf-8", errors="ignore"))
        for p in (REPO_ROOT / ".github").rglob("*.yml"):
            parts.append(p.read_text(encoding="utf-8", errors="ignore"))
        _DOC_REFS = "\n".join(parts)
    return _DOC_REFS


def module_level_paths(path: pathlib.Path) -> tuple[set[str], set[str]]:
    """Split module-level data paths into (read, write).

    A bare scan of module-level constants is not enough: build_ground_truth.py
    assigned the very paths that no longer exist, because they are the files it
    CREATES. Flagging those would report a healthy generator as broken. The
    constant is only interesting once it is consumed, and only `open(..., 'r')`
    / `.read_text()` make it a dependency.

    Returns (reads, writes) of repo-relative path literals.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError:
        return set(), set()

    const_paths: dict[str, str] = {}
    for node in tree.body:
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
                and re.match(r"^(data|results|configs|reports)/", node.value.value)
                and "{" not in node.value.value):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    const_paths[target.id] = node.value.value

    reads: set[str] = set()
    writes: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fname = (node.func.attr if isinstance(node.func, ast.Attribute)
                 else (node.func.id if isinstance(node.func, ast.Name) else ""))
        first = node.args[0] if node.args else None

        if fname == "open" and isinstance(first, ast.Name):
            mode = "r"
            if len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
                mode = node.args[1].value or "r"
            for kw in node.keywords:
                if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                    mode = kw.value.value or "r"
            target = const_paths.get(first.id)
            if target:
                (writes if "w" in str(mode) or "a" in str(mode) else reads).add(target)
        elif fname in ("read_text", "read_bytes") and isinstance(first, ast.Name):
            target = const_paths.get(first.id)
            if target:
                reads.add(target)
        elif fname in ("write_text", "write_bytes") and isinstance(first, ast.Name):
            target = const_paths.get(first.id)
            if target:
                writes.add(target)

    # inline literals passed straight to open()/read_text()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fname = (node.func.attr if isinstance(node.func, ast.Attribute)
                     else (node.func.id if isinstance(node.func, ast.Name) else ""))
            if fname not in ("open", "read_text", "read_bytes") or not node.args:
                continue
            mode = "r"
            if fname == "open" and len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
                mode = node.args[1].value or "r"
            bucket = writes if "w" in str(mode) or "a" in str(mode) else reads
            for sub in ast.walk(node):
                if (isinstance(sub, ast.Constant) and isinstance(sub.value, str)
                        and re.match(r"^(data|results|configs|reports)/", sub.value)
                        and "{" not in sub.value):
                    bucket.add(sub.value)

    return reads, writes


def live_scripts() -> list[pathlib.Path]:
    return sorted(p for p in SCRIPTS_DIR.glob("*.py") if p.stem != "__init__")


def classify(name: str) -> str:
    return STATUS_OVERRIDES.get(name, "diagnostic")


def build_manifest() -> dict:
    docs = doc_references()
    entries = {}
    for p in live_scripts():
        reads, _writes = module_level_paths(p)
        missing = sorted(
            v for v in reads if not pathlib.Path(REPO_ROOT / v).exists()
        )
        acknowledged = [v for v in missing if v in ACKNOWLEDGED_DEAD_PATHS.get(p.name, ())]
        entries[p.name] = {
            "status": classify(p.name),
            # Whether a script is documented is deliberately NOT stored here.
            # It is derived at verify time from README.md, the Makefile, the
            # workflows and bootstrap_env.sh. Caching it in the manifest made
            # the gate stale the moment any of those changed, which is how a
            # "documented" script can be reported undocumented with no code
            # change involved.
            "read_paths_missing": missing,
            "acknowledged_dead_paths": acknowledged,
        }

    archived = sorted(
        str(p.relative_to(REPO_ROOT))
        for p in ARCHIVED_DIR.rglob("*.py")
    )
    return {
        "_doc": (
            "Every scripts/*.py must have an entry here. Statuses: "
            + ", ".join(STATUSES)
            + ". A new script with no entry fails CI — that absence is exactly "
              "how scripts/ grew past src/."
        ),
        "scripts": entries,
        "archived": archived,
    }


def verify(manifest: dict) -> list[str]:
    failures = []
    entries = manifest.get("scripts", {})
    docs = doc_references()

    unlisted = [p.name for p in live_scripts() if p.name not in entries]
    for name in unlisted:
        failures.append(
            f"scripts/{name} has no MANIFEST.json entry. Run "
            f"`python scripts/check_script_manifest.py --write` after deciding "
            f"whether it is canonical, library, diagnostic, forensic or dead."
        )

    for name, entry in entries.items():
        status = entry.get("status")
        if status not in STATUSES:
            failures.append(f"scripts/{name}: status {status!r} not in {STATUSES}")

        # A script that reads data which does not exist cannot do its job. That
        # is a hard failure, not a warning: it means the script is dead but
        # still sitting in scripts/ looking available.
        acknowledged = set(entry.get("acknowledged_dead_paths", []))
        for missing in entry.get("read_paths_missing", []):
            if missing in acknowledged:
                continue
            failures.append(
                f"scripts/{name}: reads data path {missing} which does not "
                f"exist. Either restore the data, move the script to archive/, "
                f"or list it in ACKNOWLEDGED_DEAD_PATHS with the reason."
            )

        # Anything canonical or library must be discoverable.
        if status in ("canonical", "library") and name not in docs:
            failures.append(
                f"scripts/{name} is {status} but is not mentioned in README.md, "
                f"the Makefile, a workflow or bootstrap_env.sh"
            )

    stale = [n for n in entries if not (SCRIPTS_DIR / n).exists()]
    for name in stale:
        failures.append(f"MANIFEST.json lists scripts/{name}, which no longer exists")

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="regenerate the manifest")
    args = parser.parse_args()

    if args.write:
        payload = build_manifest()
        MANIFEST_PATH.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        by_status: dict[str, int] = {}
        for entry in payload["scripts"].values():
            by_status[entry["status"]] = by_status.get(entry["status"], 0) + 1
        print(f"[Written] {MANIFEST_PATH.relative_to(REPO_ROOT)}")
        for status in STATUSES:
            if by_status.get(status):
                print(f"  {status:12s} {by_status[status]}")
        print(f"  {'total':12s} {sum(by_status.values())}")
        return 0

    if not MANIFEST_PATH.exists():
        print(f"FAIL: {MANIFEST_PATH} does not exist. "
              f"Run with --write.", file=sys.stderr)
        return 1

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    failures = verify(manifest)
    entries = manifest.get("scripts", {})
    docs = doc_references()
    documented = sum(1 for name in entries if name in docs)
    print(f"    {len(entries)} scripts classified, {documented} documented, "
          f"{len(manifest.get('archived', []))} archived")

    if failures:
        for line in failures:
            print(f"    FAIL {line}", file=sys.stderr)
        print(f"\n[script manifest] FAILED ({len(failures)} problem(s))", file=sys.stderr)
        return 1
    print("    ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())