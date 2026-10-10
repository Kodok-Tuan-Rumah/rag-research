# RAG Research

> Failure-aware RAG QA pipeline with Rust-powered chunking, hybrid retrieval, and evidence-first retrieval evaluation.  
> Part of the [AI Infrastructure Workspace](https://github.com/malikimayzar) — a polyglot research platform built with Python, Rust, and Go.

![Python](https://img.shields.io/badge/Python-3.14-blue)
![Rust](https://img.shields.io/badge/Rust-PyO3%20%2B%20Maturin-orange)
![Go](https://img.shields.io/badge/Go-Dashboard-cyan)
![Retrieval Eval](https://img.shields.io/badge/Eval-Retrieval%20%2B%20RAGAS-green)

---

## ⚠️ Current State — Read This First

The canonical evaluation is **retrieval-only, no LLM judge**. The RAGAS generation-metric numbers further down are **LEGACY** — retained as audit trail, not current measurements.

| Item | CURRENT (verified 2026-10-05) | LEGACY (historical) |
|---|---|---|
| Corpus | **796 chunks** (`chunks_semantic_v2.json`) | 982 chunks |
| Embedding | `sentence-transformers/all-MiniLM-L6-v2` (384-dim) | same |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | `BAAI/bge-reranker-base` |
| Generator | `openai/gpt-oss-20b` | `llama-3.3-70b-versatile` |
| Evaluation dataset | **14 queries** (`data/evaluation/retrieval_eval_dataset.json`) | 55 QA pairs |
| Evaluation scope | Retrieval-only (Recall@K, MRR@K) + optional RAGAS | RAGAS 5-metric as primary |
| Python | **3.14.4** | 3.12 |

> ⚠️ `llama-3.3-70b-versatile` and `llama-3.1-8b-instant` return **HTTP 404** on the current API key. Use `openai/gpt-oss-20b`.
> Full drift audit: [`results/metrics/j6_documentation_drift.md`](results/metrics/j6_documentation_drift.md)

### Current Retrieval Baseline

All values restricted to the canonical 14-query sample.

| Stage | Recall@10 | MRR@10 |
|---|---|---|
| Dense | 0.7857 | 0.7551 |
| BM25 | **1.0000** | **0.8833** |
| RRF (weighted + section-boosted) | 0.8571 | 0.8571 |
| RRF + Reranker | 0.8571 | 0.7627 |
| Final (filtered) | 0.8571 | 0.7589 |

> **Latency:** P95 ≈ 4065 ms vs a 3500 ms budget → `P95_BUDGET_MISSED`. Reranker is the dominant component.

### Retrieval Ranking Mode

Hybrid retrieval uses **RRF-only by default**. The cross-encoder remains available as an explicit opt-in:

```python
chunks = retriever.search(query, top_k=10)  # RRF-only
reranked_chunks = retriever.search(query, top_k=10, rerank=True)
```

The API request uses `use_reranker: false` by default; set it to `true` to opt in. `RetrievalTools.search_hybrid()` likewise accepts `rerank=True`. Returned chunks and search traces identify the selected mode as `ranking_mode: "rrf_only"` or `"rrf_reranker"`. Evaluation scripts that measure reranking pass the option explicitly so their stage remains enabled regardless of the runtime default.

---

## What This Is

A failure-aware Retrieval-Augmented Generation QA system — comparing chunking strategies (Python semantic vs Rust semantic) and retrieval methods (Dense, BM25, weighted + section-boosted RRF + cross-encoder reranking).

Built as the **Precision Engine** of a larger AI infrastructure stack. The system does not just answer questions — it measures how well it answers them, and it abstains when retrieval fails.

> Per `PROJECT_GOALS.md` §9, this is **single-turn QA**, not a chatbot. HyDE and multi-query expansion are explicitly disabled.

---

## Architecture

```
PDF/TXT Documents
       │
       ▼
┌─────────────────────┐
│  document_loader.py │  PyMuPDF — layout-aware PDF parsing
└──────────┬──────────┘
           │
    ┌──────┴──────┐
    ▼             ▼
┌────────┐  ┌──────────────────────┐
│chunker │  │ semantic_chunker_rust │  Rust + PyO3 + Maturin
│(Python)│  │ (unicode-aware, fast) │  796 chunks from 7 papers
└────────┘  └──────────┬───────────┘
                       │
           ┌───────────┴───────────┐
           ▼                       ▼
    ┌─────────────┐        ┌──────────────┐
    │  Qdrant     │        │  BM25Okapi   │
    │  HNSW       │        │  (sparse)    │
    │  (dense)    │        └──────┬───────┘
    └──────┬──────┘               │
           │                      │
           └──────────┬───────────┘
                      ▼
           ┌──────────────────────┐
           │  Hybrid RRF (k=60)   │  Weighted + section-boosted
           │                      │  Reciprocal Rank Fusion
           └──────────┬───────────┘
                      │
                      ▼
           ┌──────────────────────┐
           │  Cross-Encoder       │  cross-encoder/ms-marco-
           │  Reranker            │  MiniLM-L-6-v2
           └──────────┬───────────┘
                      │
                      ▼
           ┌──────────────────────┐
           │  Groq Generator      │  openai/gpt-oss-20b
           │  System Prompt Strict│  Citation rules, no hallucination
           └──────────┬───────────┘
                      │
                      ▼
           ┌──────────────────────┐
           │  Retrieval Evaluator │  Recall@K, MRR@K
           │  (no LLM required)   │  ← CANONICAL PATH
           ├──────────────────────┤
           │  RAGAS Evaluator     │  5 metrics — Faithfulness,
           │  (optional: needs    │  Answer Relevancy, Context
           │   API key + judge)   │  Precision, Recall, Correctness
           └──────────────────────┘
```

---

## Stack

| Layer | Tool | Role |
|---|---|---|
| Chunking | Rust + PyO3 + Maturin | Unicode-aware semantic segmentation, 10-50x faster than Python |
| Embedding | all-MiniLM-L6-v2 (384-dim) | Dense vector encoding |
| Vector Store | Qdrant HNSW (local persistent) | 796 vectors, filterable, scalable |
| Sparse Retrieval | BM25Okapi (rank-bm25) | Keyword matching |
| Hybrid Fusion | Weighted + section-boosted RRF, k=60 | Combine dense + BM25 |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | Cross-encoder reranking |
| LLM | Groq `openai/gpt-oss-20b` | Sub-second generation (reasoning model) |
| Evaluation | Retrieval-only (Recall@K, MRR@K) | **Canonical** — no LLM judge |
| Evaluation (opt.) | RAGAS + LangChain | 5 generation metrics, needs API key |
| Observability | structlog + prometheus-client | JSON logs, latency/abstain metrics |
| Dashboard | Go + Fiber | Analytics leaderboard |

---

## Evaluation Results

### CANONICAL — Retrieval Evaluation

7 ArXiv papers (RAG, LLM, Attention domains), **14 gold-labeled queries**, retrieval-only (no LLM judge). Baseline shown in [Current State](#-current-state--read-this-first).

Reproduce:

```bash
python3 scripts/run_retrieval_eval.py --out results/metrics/retrieval_eval_latest.json
```

Known limitations, quantified in `results/metrics/j3_statistical_sufficiency.md`:

- N=14 → McNemar exact p=0.50, Wilson 95% CIs overlap. **Not sufficient for definitive architecture conclusions.**
- 0 dense-only gold cases → RRF's hybrid value is **untestable** on this sample.
- `MRR@10` uses RR@10 (`1/rank` only for `1 <= rank <= 10`). Older `retrieval_stage_eval_*.json` artifacts use unrestricted `1/rank` — do not treat their MRR@10 as canonical.

---

### LEGACY — RAGAS Generation Metrics (historical, not current)

> **LEGACY.** Retained as audit trail. These were measured on a 55-pair dataset, a different reranker (`BAAI/bge-reranker-base`), and a generator model that is now 404 on the current API key. They are **not** the current measurement baseline. Numbers are left unmodified.

All runs on 7 ArXiv papers. 55 clean QA pairs, RAGAS framework, `llama-3.3-70b` generator + judge.

#### RAGAS Metrics — Best Run (clean_dataset_v2, 70b judge, n=55)

| Metric | Score | Notes |
|---|---|---|
| **Faithfulness** | **0.8000** | Every claim traceable to source |
| **Context Precision** | 0.7500 | Relevant chunks retrieved |
| **Context Recall** | 0.9313 | Near-complete coverage |
| **Answer Relevancy** | 0.7438 | On-topic responses |
| **Answer Correctness** | 0.6846 | Factual accuracy vs ground truth |
| **Hallucination Rate** | 18.4% | Detected and logged |

#### Progression Across Runs

| Experiment | Faithfulness | Samples | Judge | Notes |
|---|---|---|---|---|
| semantic_hybrid_rerank_v1 | 0.6754 | 20 | Manual | Corrupted dataset baseline |
| clean_dataset_v1 | 0.7987 | 33 | 8b | Clean dataset, first run |
| clean_dataset_v2 | **0.8000** | 55 | 70b | Full dataset, best result |
| final_v1 | 0.7297 | 55 | 8b+json | 70b generator, 8b judge |

**+18.4% Faithfulness improvement** from dataset cleanup + system prompt upgrade.

#### Original Ablation Study (Phase 1, Ollama/Mistral)

> **LEGACY.** Phase 1, pre-Qdrant stack, 18 evaluations total. Too small to be meaningful; kept for provenance.

6 experiments × 3 queries = 18 evaluations. Runtime: 241 minutes.

| Rank | Method | Chunk | Faithfulness | Hallucination |
|---|---|---|---|---|
| 🥇 1 | BM25 | 256 | 1.000 | 0% |
| 🥈 2 | Hybrid RRF | 512 | 1.000 | 0% |
| 🥉 3 | Dense | 512 | 0.933 | 0% |
| 4 | Dense | 256 | 0.917 | 0% |
| 5 | BM25 | 512 | 0.833 | 0% |
| 6 | Hybrid RRF | 256 | 0.633 | 0% |

**Zero hallucinations across all 18 queries** with Ollama/Mistral + context-grounding prompt.

---

## Quick Start

```bash
# Clone and setup
git clone https://github.com/malikimayzar/rag-research
cd rag-research

python -m venv venv
source venv/bin/activate
pip install -e .

# Set API key — https://console.groq.com/keys  (format: gsk_...)
echo "GROQ_API_KEY=gsk_your_actual_key_here" > .env

# Build Rust chunker
cd src/ingestion/semantic_chunker_rust && maturin develop && cd ../../..

# Run full pipeline
make all
```

Verify the key and the pipeline:

```bash
# Key check
python3 -c "import os;from dotenv import load_dotenv;load_dotenv(dotenv_path='.env');\
k=os.getenv('GROQ_API_KEY','');print('VALID' if k.startswith('gsk_') and len(k)>20 else 'INVALID')"

# End-to-end single query (needs API key)
python3 scripts/run_single_query.py --query "What is Hybrid Search in RAG?" --mode full

# Retrieval eval (NO API key needed)
python3 scripts/run_retrieval_eval.py --out results/metrics/retrieval_eval_latest.json
```

### Manual Steps

```bash
# Ingest + chunk (Rust engine)
python src/ingestion/document_loader.py
python src/ingestion/semantic_chunker.py    # generates 796 chunks

# Index to Qdrant
python src/retrieval/qdrant_store.py

# Run retrieval eval (no API key required) — CANONICAL
python scripts/run_retrieval_eval.py

# Run RAGAS generation eval (optional — needs API key + LLM judge)
python scripts/run_eval.py

# Visualize (needs matplotlib)
python scripts/visualize.py

# Start Go dashboard
go run cmd/server/main.go
```

### Tests

```bash
python3 -m pytest tests/ -q
```

Baseline: **87 passed, 3 failed**. The 3 failures are a pre-existing `numpy 2.5.3` / Python 3.14 incompatibility (`numpy.fft` fails to load twice in one process), unrelated to these requirements. Work around by running those two files separately:

```bash
python3 -m pytest tests/test_qdrant_store.py -q
python3 -m pytest tests/test_semantic_chunker.py -q
```

### Live Demo

```bash
python demo.py
```

```
Q: What is Retrieval-Augmented Generation?
A: Retrieval-Augmented Generation (RAG) is a technique that combines
   information retrieval with language model generation [Source 1],
   enhancing LLMs by retrieving relevant document chunks from an
   external knowledge base [Source 2].
⚡ Latency: 1.471s
```

---

## Makefile

```bash
make rust-build   # Compile Rust chunker (PyO3 + Maturin)
make ingest       # Parse PDFs + semantic chunk
make index        # Index 796 vectors to Qdrant
make eval         # Run RAGAS evaluation (optional, needs API key)
make viz          # Generate figures
make all          # Full pipeline: rust-build → ingest → index → eval → viz
make dashboard    # Start Go analytics dashboard
```

---

## Project Structure

```
rag-research/
├── src/
│   ├── _compat.py                    # Python 3.14 protobuf shim (import FIRST)
│   ├── ingestion/
│   │   ├── document_loader.py        # PyMuPDF PDF parsing
│   │   ├── chunker.py                # Fixed-size chunker (regression baseline)
│   │   ├── semantic_chunker.py       # Python semantic chunker
│   │   ├── semantic_chunker_rust/    # Rust chunker (PyO3 + Maturin)
│   │   └── dataset_builder.py        # QA dataset generation
│   ├── retrieval/
│   │   ├── hybrid_retriever.py       # CANONICAL pipeline (dense+BM25→RRF→rerank)
│   │   ├── qdrant_store.py           # Qdrant HNSW vector store
│   │   ├── bm25_retriever.py         # BM25Okapi sparse retrieval
│   │   ├── chunk_filters.py          # 3-layer filter + completeness heuristic
│   │   ├── model_registry.py         # Shared embedder/reranker loading
│   │   ├── tools.py                  # RetrievalTools adapter
│   │   └── embedder.py               # Legacy FAISS embedder (dead code)
│   ├── generation/
│   │   └── generator.py              # Groq generator, strict system prompt
│   ├── controller/
│   │   ├── agent.py                  # Orchestration loop
│   │   ├── policy_engine.py          # Routing policy
│   │   ├── confidence_engine.py      # Confidence gatekeeping
│   │   └── agent_state.py            # Agent state model
│   ├── evaluation/
│   │   ├── retrieval_evaluator.py    # CANONICAL: Recall@K, MRR@K
│   │   ├── gold_matching.py          # EXACT_ANSWER_SPAN / TOKEN_RECALL
│   │   ├── gold_data_validator.py    # Gold integrity checks
│   │   ├── ragas_evaluator.py        # Optional RAGAS 5-metric
│   │   └── evaluator.py              # Legacy Ollama evaluator (dead code)
│   └── api/
│       ├── main.py                   # FastAPI endpoint
│       └── config.py                 # Settings (pydantic-settings)
├── scripts/                          # ~40 diagnostic + eval scripts
├── tests/                            # pytest suite (87 passed, 3 pre-existing fails)
├── experiments/                      # Phase 1 ablation (legacy stack)
├── data/
│   ├── raw/                          # 7 ArXiv PDFs + TXT
│   ├── processed/
│   │   ├── chunks_semantic_v2.json   # 796 chunks — CANONICAL corpus
│   │   ├── chunks_semantic.json      # 796 chunks (v1)
│   │   └── documents.json            # 7 documents
│   ├── evaluation/
│   │   ├── retrieval_eval_dataset.json     # 14 queries — CANONICAL
│   │   ├── retrieval_eval_dataset_v3.json  # 10 validated rows
│   │   ├── retrieval_train_eval.json       # 8 rows
│   │   └── retrieval_holdout_eval.json     # 6 rows
│   ├── adversarial/                  # 6 adversarial queries (separate set)
│   ├── debug/failure_analysis.json   # Failure mode breakdown
│   └── qdrant_storage/               # Local Qdrant persistent (796 vectors)
├── results/
│   ├── metrics/                      # 93 artifacts (Phase A–J evidence)
│   ├── debug/
│   └── logs/
├── cmd/server/main.go                # Go dashboard server
├── demo.py                           # Live demo script
├── Makefile
├── requirements.txt                  # Pinned, audit-verified
├── requirements/                     # Modular: base, llm, api, dev, research, runtime
├── pyproject.toml                    # Editable install (pip install -e .)
└── docker-compose.yml
```

> `src/_compat.py` works around a broken `google._upb` C-extension in protobuf on Python 3.14. It must be imported **before** `qdrant_client`; `qdrant_store.py` already handles this.

---

## Key Engineering Decisions

**Rust Chunker over Python**  
> **LEGACY figure.** The 652 vs 982 comparison predates the v2 corpus. Current corpus is **796 chunks** for both `chunks_semantic.json` and `chunks_semantic_v2.json`.

**Qdrant over FAISS**  
FAISS is flat and memory-bound. Qdrant is HNSW-based, filterable by `doc_id`, persistent, and REST-accessible. (`src/retrieval/embedder.py` still hard-imports `faiss` but is imported by nothing — dead code.)

**Groq over Ollama**  
Ollama latency: 30-60s per query. Groq latency: 0.3-1.5s. Critical for evaluation loops that call the LLM hundreds of times.

**Retrieval eval before generation eval**  
RAGAS needs an LLM judge, which costs money and makes runs non-deterministic. Retrieval metrics (Recall@K, MRR@K) are free, deterministic, and pinpoint *where* a failure originates. Retrieval eval is therefore the canonical path; RAGAS is optional.

**`MRR@10` means RR@10**  
`1/rank` only when `1 <= rank <= 10`, else 0. Some older `results/metrics/retrieval_stage_eval_*.json` artifacts use unrestricted `1/rank`, inflating MRR@10 (RRF reads 0.8666 there vs the correct 0.8571). Documented in `results/metrics/h0_metric_reconciliation.md`.

**Strict System Prompt**  
Generator uses `system` + `user` message separation with explicit rules: answer only from context, cite sources with [Source N], refuse if context insufficient.

**Gold labels are evidence-derived, never generated**  
Questions are derived from existing corpus chunks that carry the answer evidence, then validated against it (`EXACT_ANSWER_SPAN` or `TOKEN_RECALL_VALIDATED`). Never the reverse.

---

## Dataset

7 ArXiv papers across 3 tiers:

| Tier | Paper | Domain |
|---|---|---|
| 1 | 2005.11401 — RAG (Lewis et al.) | RAG |
| 1 | 2312.10997 — Advanced RAG Survey | RAG |
| 1 | test_intro.txt | RAG intro |
| 2 | 2210.11610 — Large Language Models | LLM |
| 2 | 2212.10560 — Self-Instruct | LLM |
| 3 | 1706.03762 — Attention Is All You Need | Transformers |
| 3 | 2307.09288 — Llama 2 | LLM |

### CANONICAL evaluation set — 14 queries

`data/evaluation/retrieval_eval_dataset.json`. Every row carries `query_id`, `question`, `gold_answer`, `gold_chunk_id`, `gold_doc_id`, `match_type`, `provenance`, `question_type`, `difficulty`, `should_abstain`.

Source coverage across the 14:

| Coverage | Count | Queries |
|---|---:|---|
| Dense + BM25 both retrieve gold | 12 | q01–q04, q09–q11, q13, q14, q16, q19, q20 |
| BM25-only | 2 | q06, q15 |
| Dense-only | **0** | — |
| Adversarial (separate set) | 6 | `data/adversarial/adversarial_queries.json` |

> ⚠️ The 0 dense-only cases mean RRF's hybrid contribution is **untestable** here. This is the main dataset limitation, quantified in `results/metrics/j3_statistical_sufficiency.md`.

> **LEGACY:** an earlier 55-pair dataset (`ground_truth_qa.json`) no longer exists in the repo. Its RAGAS scores are preserved above as historical record only.

### Expansion feasibility

Audited every available evidence source (84 candidate rows). **0 new defensible queries** — all validated rows are a subset of the canonical 14. Growing N requires new evidence, not re-labeling. See `results/metrics/j1_dataset_expansion_feasibility.md`.

---

## Environment

```
Python  : 3.14.4                (tested: 3.14; see requirements.txt)
Rust    : stable (maturin + PyO3)
Go      : 1.22+
OS      : WSL2 Ubuntu
Hardware: CPU-only, 7.6GB RAM, 8 cores
Qdrant  : local persistent (data/qdrant_storage/) — 796 vectors
Groq    : openai/gpt-oss-20b (generator)
Embed   : sentence-transformers/all-MiniLM-L6-v2 (384-dim)
Rerank  : cross-encoder/ms-marco-MiniLM-L-6-v2
```

> ⚠️ **`openai/gpt-oss-20b` is a reasoning model.** It spends `max_tokens` on internal reasoning before producing content. The project default of 768 tokens is sufficient; smaller values truncate output to near-empty. Verified models for the current API key are listed in `requirements/llm.txt`.
>
> ⚠️ `llama-3.3-70b-versatile` and `llama-3.1-8b-instant` return **HTTP 404** on the current key.

---

## Part of AI Infrastructure Workspace

This repo is the **Precision Engine** in a 4-service AI research platform:

| Repo | Role | Stack |
|---|---|---|
| [mcp-gateway](https://github.com/malikimayzar/mcp-gateway) | Orchestration — routes queries, manages tool execution | Go |
| [arxiv-research-assistant](https://github.com/malikimayzar/arxiv-research-assistant) | Librarian — ArXiv ingestion, metadata, observability | Go + Python |
| **rag-research** | **Precision Engine — hybrid retrieval, evidence-first retrieval eval** | **Python + Rust + Go** |
| [llm-eval-framework](https://github.com/malikimayzar/llm-eval-framework) | Auditor — faithfulness without LLM judge | Python |

---

*Maliki Mayzar · 2025-2026 · [github.com/malikimayzar](https://github.com/malikimayzar)*