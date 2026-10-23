.PHONY: setup rust-build ingest index eval viz phase1-smoke phase1-review phase1-check all clean

# Vars
PYTHON = python
SCRIPTS = scripts

setup:
	pip install -e .
	go mod tidy

rust-build:
	cd src/ingestion/semantic_chunker_rust && maturin develop

ingest:
	$(PYTHON) src/ingestion/document_loader.py
	$(PYTHON) src/ingestion/semantic_chunker.py

index:
	$(PYTHON) src/retrieval/qdrant_store.py

eval:
	$(PYTHON) $(SCRIPTS)/run_eval.py

viz:
	$(PYTHON) $(SCRIPTS)/visualize.py

phase1-smoke:
	QDRANT_COLLECTION=rag_research_phase1_v3 \
	BM25_CHUNKS_PATH=data/processed/chunks_semantic_v3.json \
	$(PYTHON) $(SCRIPTS)/phase1_smoke.py \
		--dataset data/evaluation/phase1_smoke_set_v3.json \
		--output results/metrics/phase1_smoke_v3.json

# Annotate human_review for the 20 acceptance cases. Reads the existing report
# and shows the exact generation contexts. No LLM call. Finish with `make phase1-check`.
phase1-review:
	$(PYTHON) $(SCRIPTS)/phase1_smoke.py --review-report results/metrics/phase1_smoke_v3.json

phase1-check:
	$(PYTHON) $(SCRIPTS)/phase1_smoke.py --check-report results/metrics/phase1_smoke_v3.json

# THE MASTER COMMAND
all: rust-build ingest index eval viz

# Start the dashboard 
dashboard:
	go run cmd/server/main.go
