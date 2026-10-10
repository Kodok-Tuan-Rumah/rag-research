#!/usr/bin/env bash
# =============================================================================
# bootstrap_env.sh — rebuild the rag-research Python environment from scratch.
#
# WHY THIS EXISTS
#   The environment used to live in a shared venv OUTSIDE this repo
#   (~/projects/ai-infra-workspace/venv). It was wiped twice by external events,
#   which silently broke:
#     - all Python dependencies        (pip cache still warm, so rebuild is fast)
#     - the editable install of `src`   (scripts/ could no longer `import src`)
#     - the maturin-built Rust chunker  (`make ingest` hard-raises without it)
#
#   `pip install -r requirements.txt` alone is NOT sufficient. All three steps
#   below are required.
#
# USAGE
#   bash scripts/bootstrap_env.sh              # rebuild the active interpreter
#   bash scripts/bootstrap_env.sh --local      # build a repo-local .venv first
#   bash scripts/bootstrap_env.sh --verify     # check only, install nothing
#
# NOTE: run this in the FOREGROUND. Do not background it — a terminal or
#       agent-host restart mid-install leaves a half-built environment.
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VERIFY_ONLY=0
USE_LOCAL=0
for arg in "$@"; do
    case "$arg" in
        --verify) VERIFY_ONLY=1 ;;
        --local)  USE_LOCAL=1 ;;
        -h|--help)
            sed -n '2,22p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "unknown flag: $arg" >&2; exit 2 ;;
    esac
done

step() { printf '\n\033[1;34m==> %s\033[0m\n' "$1"; }
ok()   { printf '    \033[32mOK\033[0m  %s\n' "$1"; }
warn() { printf '    \033[33m!\033[0m   %s\n' "$1"; }
die()  { printf '    \033[31mFAIL\033[0m %s\n' "$1" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 0. Optional: build an isolated in-repo venv so this repo cannot be broken by
#    (or break) a sibling project sharing the workspace interpreter.
# ---------------------------------------------------------------------------
if [ "$USE_LOCAL" -eq 1 ] && [ "$VERIFY_ONLY" -eq 0 ]; then
    step "Creating repo-local .venv"
    [ -d .venv ] || python3 -m venv .venv
    ok ".venv exists at $REPO_ROOT/.venv"
    # shellcheck disable=SC1091
    source .venv/bin/activate
    ok "activated: $(command -v python3)"
fi

PY="$(command -v python3)"
echo "    interpreter: $PY"
echo "    repo root  : $REPO_ROOT"

# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
check() {
    local label="$1"; shift
    if "$@" >/dev/null 2>&1; then ok "$label"; else warn "$label MISSING"; return 1; fi
}

run_checks() {
    local failures=0
    step "Verification"
    check "python deps (qdrant_client/groq/rank_bm25/transformers)" \
        python3 -c "import qdrant_client, groq, rank_bm25, sentence_transformers, transformers" || failures=1
    check "pydantic-settings (src/api/config.py hard dependency)" \
        python3 -c "import pydantic_settings" || failures=1
    check "editable install (import src)" \
        python3 -c "import src, src.api.config" || failures=1
    check "Rust chunker (semantic_chunker_rust)" \
        python3 -c "import semantic_chunker_rust" || failures=1
    check "protobuf upb shim (src/_compat.py, needed on Python 3.14)" \
        python3 -c "import src._compat, qdrant_client" || failures=1

    if [ "$failures" -eq 0 ]; then
        step "Test suite"
        if python3 -m pytest tests/ -q; then
            ok "suite green"
        else
            die "suite FAILED — see output above"
        fi
    else
        die "environment incomplete — re-run without --verify"
    fi
}

if [ "$VERIFY_ONLY" -eq 1 ]; then
    run_checks
    exit 0
fi

# ---------------------------------------------------------------------------
# 1. Dependencies
# ---------------------------------------------------------------------------
step "Installing dependencies from requirements.txt"
python3 -m pip install --upgrade pip >/dev/null
python3 -m pip install -r requirements.txt
ok "dependencies installed"

# ---------------------------------------------------------------------------
# 2. The project itself
#   requirements.txt lists third-party deps only. Without this, `import src`
#   fails from inside scripts/ because only scripts/ lands on sys.path.
# ---------------------------------------------------------------------------
step "Installing rag-research in editable mode"
python3 -m pip install -e . --no-deps
ok "editable install done (src importable)"

# ---------------------------------------------------------------------------
# 3. Native Rust chunker
#   src/ingestion/semantic_chunker.py raises ImportError without it, so the
#   whole ingestion stage is dead if this step is skipped.
# ---------------------------------------------------------------------------
step "Building Rust chunker (maturin develop)"
if ! command -v cargo >/dev/null 2>&1; then
    warn "cargo not found — skipping."
    warn "src/ingestion/semantic_chunker.py will raise ImportError until you"
    warn "install Rust and re-run: make rust-build"
else
    ( cd src/ingestion/semantic_chunker_rust && maturin develop )
    ok "semantic_chunker_rust built"
fi

# ---------------------------------------------------------------------------
step "Done"
cat <<'EOF'

    Next:
      python3 -m pytest tests/ -q
      python3 scripts/run_single_query.py --query "What is Hybrid Search in RAG?" --mode full

    If retrieval returns nothing, the Qdrant index may be stale or absent:
      python3 src/ingestion/document_loader.py
      python3 src/ingestion/semantic_chunker.py
      python3 src/retrieval/qdrant_store.py

EOF

run_checks