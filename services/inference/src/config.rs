use std::env;

#[derive(Debug, Clone)]
pub struct Config {
    pub port: u16,
    pub embedding_workers: usize,
    pub embedding_intra_threads: usize,
    pub reranker_workers: usize,
    pub reranker_intra_threads: usize,
    pub embedding_max_length: usize,
    pub reranker_max_length: usize,
    pub embedding_model_path: String,
    pub embedding_tokenizer_path: String,
    pub reranker_model_path: String,
    pub reranker_tokenizer_path: String,
    pub queue_capacity: usize,
    pub batch_window_ms: u64,
    pub max_batch_size: usize,
    pub max_texts: usize,
    pub max_documents: usize,
    pub request_timeout_ms: u64,
}

impl Config {
    pub fn from_env() -> Self {
        Self {
            port: env::var("INFERENCE_PORT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(8081),
            embedding_workers: env::var("EMBEDDING_WORKERS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(2),
            embedding_intra_threads: env::var("EMBEDDING_INTRA_THREADS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(2),
            reranker_workers: env::var("RERANKER_WORKERS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(2),
            reranker_intra_threads: env::var("RERANKER_INTRA_THREADS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(2),
            embedding_max_length: env::var("EMBEDDING_MAX_LENGTH")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(256),
            reranker_max_length: env::var("RERANKER_MAX_LENGTH")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(256),
            embedding_model_path: env::var("EMBEDDING_MODEL_PATH")
                .unwrap_or_else(|_| "/models/embedding/model.onnx".to_string()),
            embedding_tokenizer_path: env::var("EMBEDDING_TOKENIZER_PATH")
                .unwrap_or_else(|_| "/models/embedding/tokenizer.json".to_string()),
            reranker_model_path: env::var("RERANKER_MODEL_PATH")
                .unwrap_or_else(|_| "/models/reranker/model.onnx".to_string()),
            reranker_tokenizer_path: env::var("RERANKER_TOKENIZER_PATH")
                .unwrap_or_else(|_| "/models/reranker/tokenizer.json".to_string()),
            queue_capacity: 256,
            batch_window_ms: 3,
            max_batch_size: 32,
            max_texts: 128,
            max_documents: 64,
            request_timeout_ms: 5000,
        }
    }
}
