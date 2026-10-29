.PHONY: setup rust-build ingest index eval viz phase1-smoke phase1-review phase1-check all clean refit-calibration

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
	BM25_CHUNKS_PATH=data/processed/chunks_semantic_v2.json \
	$(PYTHON) $(SCRIPTS)/phase1_smoke.py \
		--dataset data/evaluation/phase1_smoke_set_v3.json \
		--output results/metrics/phase1_smoke_v3.json

# Annotate human_review for the 20 acceptance cases. Reads the existing report
# and shows the exact generation contexts. No LLM call. Finish with `make phase1-check`.
phase1-review:
	$(PYTHON) $(SCRIPTS)/phase1_smoke.py --review-report results/metrics/phase1_smoke_v3.json

phase1-check:
	$(PYTHON) $(SCRIPTS)/phase1_smoke.py --check-report results/metrics/phase1_smoke_v3.json

refit-calibration:
	@$(PYTHON) $(SCRIPTS)/build_calibration_from_audit.py --min-samples 50 || true
	@if [ -f data/evaluation/calibration_dataset_from_audit.json ]; then \
		$(PYTHON) $(SCRIPTS)/fit_calibration.py --dataset data/evaluation/calibration_dataset_from_audit.json --out models; \
	else \
		echo "skip: calibration dataset not produced (insufficient data or DB unavailable)"; \
	fi

# Start the dashboard 
dashboard:
	go run cmd/server/main.go
