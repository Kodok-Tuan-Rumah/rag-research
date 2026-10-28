use crate::error::HttpError;
use serde::{Deserialize, Serialize};

// ... existing DTOs ...

pub fn validate_rerank(req: &RerankRequest, max_documents: usize) -> Result<(), HttpError> {
    if req.query.trim().is_empty() {
        return Err(HttpError::bad_request("blank_query", "query must not be blank"));
    }
    if req.documents.len() > max_documents {
        return Err(HttpError::payload_too_large("too_many_documents", "too many documents"));
    }
    for d in &req.documents {
        if d.id.trim().is_empty() {
            return Err(HttpError::bad_request("blank_id", "id must not be blank"));
        }
        if d.text.trim().is_empty() {
            return Err(HttpError::bad_request("blank_text", "text must not be blank"));
        }
    }
    Ok(())
}

pub fn validate_embed(texts: &[String], max_texts: usize) -> Result<(), HttpError> {
    if texts.is_empty() {
        return Err(HttpError::bad_request("empty_input", "texts must not be empty"));
    }
    for t in texts {
        if t.trim().is_empty() {
            return Err(HttpError::bad_request("empty_text", "text must not be blank"));
        }
    }
    if texts.len() > max_texts {
        return Err(HttpError::payload_too_large("too_many", "too many texts"));
    }
    Ok(())
}

#[derive(Debug, Clone, Deserialize)]
pub struct EmbedRequest {
    pub texts: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct EmbedResponse {
    pub model: String,
    pub dim: usize,
    pub embeddings: Vec<Vec<f32>>,
    pub truncated: usize,
    pub max_length: usize,
    pub latency_ms: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Document {
    pub id: String,
    pub text: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RerankRequest {
    pub query: String,
    pub documents: Vec<Document>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RerankScore {
    pub id: String,
    pub score: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct RerankResponse {
    pub model: String,
    pub scores: Vec<RerankScore>,
    pub truncated: usize,
    pub max_length: usize,
    pub latency_ms: f64,
}

#[derive(Debug, Clone)]
pub struct EmbedResult {
    pub embeddings: Vec<Vec<f32>>,
    pub truncated: usize,
}

#[derive(Debug, Clone)]
pub struct RerankResult {
    pub scores: Vec<f32>,
    pub ids: Vec<String>,
    pub truncated: usize,
}
