# syntax=docker/dockerfile:1
# rag-research / rag-api — image mandiri (tanpa base image lokal workspace)
#
# Kebijakan build (lihat docs/ops-postmortem.md S-5):
#   - base PUBLIK dipin by digest — bukan tag milik service lain yang bergerak
#   - seluruh dependency runtime berasal dari requirements/runtime.txt (pins repo)
#   - torch sengaja CPU-only: base lama membawa CUDA 13 (~10 GB) yang tidak dipakai
#   - builder butuh jaringan (PyPI + index CPU PyTorch); image akhir self-contained
#
# Build (dari root workspace):
#   docker compose build rag-api
#
# Catatan: `--network none` tidak lagi didukung kecuali seluruh wheel di-vendor.
# Runtime image tetap offline: tidak ada pip/network saat container berjalan.

ARG PYTHON_IMAGE=python:3.12-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1

# ── Stage 1: dependency builder ─────────────────────────────────────────────
FROM ${PYTHON_IMAGE} AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_RETRIES=10 \
    PIP_TIMEOUT=60

WORKDIR /build
COPY requirements/runtime.txt ./requirements/runtime.txt

RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip

# torch CPU dulu supaya sentence-transformers tidak menarik wheel CUDA dari PyPI.
# Setiap langkah dipisah menjadi layer sendiri: kegagalan jaringan saat
# me-resolve requirements tidak boleh memaksa unduh ulang torch (~200 MB).
RUN /opt/venv/bin/pip install \
        --index-url https://download.pytorch.org/whl/cpu \
        "torch==2.14.1+cpu"

RUN /opt/venv/bin/pip install -r requirements/runtime.txt

# ── Stage 2: runtime ────────────────────────────────────────────────────────
FROM ${PYTHON_IMAGE} AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:${PATH}"

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY src/ /app/src/
COPY data/processed/ /app/data/processed/

# Calibration artifacts are build outputs, not tracked inputs: models/ is not in
# git, so a clone that only ran `docker compose build` used to start with no
# model and silently fall back to raw heuristic confidence. Fit them here from
# the labeled dataset; the script refuses ERROR rows instead of training on
# failed pipeline runs.
COPY data/evaluation/calibration_dataset.json /app/data/evaluation/calibration_dataset.json
COPY scripts/fit_calibration.py /app/scripts/fit_calibration.py
RUN PYTHONPATH=/app python scripts/fit_calibration.py \
        --dataset data/evaluation/calibration_dataset.json \
        --out models

# Download the models into the image so a fresh clone needs no host cache.
# The reranker is included because `use_reranker=true` is a documented feature.
ENV HF_HOME=/app/hf-hub \
    HF_HUB_OFFLINE=0
RUN python - <<'PY'
from sentence_transformers import SentenceTransformer, CrossEncoder

SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
PY
ENV HF_HUB_OFFLINE=1

RUN mkdir -p /app/data/cache

EXPOSE 8000

# python:slim ships no curl/wget — probe with urllib.
HEALTHCHECK --interval=15s --timeout=10s --start-period=60s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=10)" || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
