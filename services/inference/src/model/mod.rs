use crate::error::ModelError;
use std::sync::OnceLock;

pub mod embedder;
pub mod reranker;

pub use embedder::OnnxEmbedder;
pub use reranker::{OnnxReranker, scores_from_logits};

static ORT_ENV: OnceLock<ort::environment::Environment> = OnceLock::new();

pub fn get_ort_env() -> Result<&'static ort::environment::Environment, ModelError> {
    if let Some(env) = ORT_ENV.get() {
        return Ok(env);
    }
    let env = ort::init()
        .build()
        .map_err(|e| ModelError::Other(format!("ORT init: {}", e)))?;
    let _ = ORT_ENV.set(env);
    Ok(ORT_ENV.get().unwrap())
}
