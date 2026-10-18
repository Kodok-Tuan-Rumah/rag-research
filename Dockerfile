# syntax=docker/dockerfile:1
# rag-research / rag-api Dockerfile
#
# Uses arxiv-python-ml:latest as base image because it already has all Python
# packages needed: FastAPI, uvicorn, torch, sentence-transformers,
# transformers, nomic-embed-text model cache, etc.
# We only need to add the rag-research source code.
#
# Build:
#   docker build --network none -t ai-infra-workspace-rag-api:latest .
#
# ★ Base = TAG BEKU (ai-infra-workspace-rag-base:v1) — lihat docs/ops-postmortem.md S-5:
#   dependensi RAG-spesifik sempat hilang ketika memakai tag :latest milik service
#   lain yang ikut di-rebuild tanpa paket tersebut. Base beku + wheel lokal di
#   bawah membuat build ini tidak lagi bergantung pada kondisi tag lain.
FROM ai-infra-workspace-rag-base:v1

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Copy rag-research source code
COPY src/ /app/src/
COPY data/processed/ /app/data/processed/

# Install additional dependencies needed by rag-api
# (offline: versi dipin sama dengan wheel di requirements/wheels-server,
#  supaya build --network none deterministik — tidak menyentuh PyPI/DNS)
COPY requirements/wheels-server/ /tmp/wheels/
RUN pip install --no-cache-dir --no-index --find-links=/tmp/wheels \
        "pydantic_settings==2.16.0" "rank_bm25==0.2.2" \
        "python-multipart==0.0.32" \
    && rm -rf /tmp/wheels

# Baked HF cache (all-MiniLM-L6-v2) — embedder boot tanpa retry/download Hub
COPY hf-cache/ /app/hf-hub/
ENV HF_HOME=/app/hf-hub \
    HF_HUB_OFFLINE=1

# Create directory for runtime data
RUN mkdir -p /app/data/cache

# Expose port
EXPOSE 8000

# Health check: python:slim ships no curl/wget — probe with urllib.
# Embedder is baked in (mounted from host), so startup is fast.
HEALTHCHECK --interval=15s --timeout=10s --start-period=60s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=10)" || exit 1

# Run application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
