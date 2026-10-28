# rag-inference

Rust/ONNX inference service for embeddings and reranking.

Endpoints: `/health`, `/ready`, `/metrics`, `POST /embed`, `POST /rerank`.

Env vars: `INFERENCE_PORT`, `EMBEDDING_*`, `RERANKER_*`, `EMBEDDING_MODEL_PATH`, etc. `ORT_DYLIB_PATH` must point to `libonnxruntime.so`.

Build: `cargo build --release`. Run with model files under `/models`.