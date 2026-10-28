use rag_inference::config::Config;
use rag_inference::http::{EmbedRequest, EmbedResponse, RerankRequest, RerankResponse, RerankScore};
use rag_inference::http;
use rag_inference::model::{OnnxEmbedder, OnnxReranker};
use rag_inference::worker::WorkerPool;
use serde_json::json;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::time::Instant;

struct AppState {
    config: Config,
    embed_pool: Option<WorkerPool<OnnxEmbedder, String, Vec<f32>>>,
    rerank_pool: Option<WorkerPool<OnnxReranker, RerankRequest, http::RerankResult>>,
    embed_model_name: String,
    reranker_model_name: String,
    ready: bool,
}

fn main() {
    let config = Config::from_env();
    let port = config.port;

    // load models if files exist
    let embedder = OnnxEmbedder::load(
        &config.embedding_model_path,
        &config.embedding_tokenizer_path,
        config.embedding_intra_threads,
        config.embedding_max_length,
    );
    if let Err(e) = &embedder {
        eprintln!("embedder load failed: {}", e);
    }
    let embedder = embedder.ok();

    let reranker = OnnxReranker::load(
        &config.reranker_model_path,
        &config.reranker_tokenizer_path,
        config.reranker_intra_threads,
        config.reranker_max_length,
    );
    if let Err(e) = &reranker {
        eprintln!("reranker load failed: {}", e);
    }
    let reranker = reranker.ok();

    let ready = embedder.is_some() && reranker.is_some();

    let embed_pool = embedder.map(|m| {
        WorkerPool::new(
            m,
            config.embedding_workers,
            config.queue_capacity,
            config.batch_window_ms,
            config.max_batch_size,
        )
    });
    let rerank_pool = reranker.map(|m| {
        WorkerPool::new(
            m,
            config.reranker_workers,
            config.queue_capacity,
            config.batch_window_ms,
            config.max_batch_size,
        )
    });

    let state = Arc::new(AppState {
        config: config.clone(),
        embed_pool,
        rerank_pool,
        embed_model_name: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
        reranker_model_name: "cross-encoder/ms-marco-MiniLM-L-6-v2".to_string(),
        ready,
    });

    let listener = TcpListener::bind(format!("0.0.0.0:{}", port)).expect("bind failed");
    println!("rag-inference listening on {}", port);
    for stream in listener.incoming() {
        match stream {
            Ok(s) => {
                let st = Arc::clone(&state);
                std::thread::spawn(move || handle_connection(s, st));
            }
            Err(_) => {}
        }
    }
}

fn handle_connection(mut stream: TcpStream, state: Arc<AppState>) {
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    // read request line
    let mut request_line = String::new();
    if reader.read_line(&mut request_line).is_err() {
        return;
    }
    let parts: Vec<&str> = request_line.split_whitespace().collect();
    if parts.len() < 2 {
        return;
    }
    let method = parts[0];
    let path = parts[1];
    // read headers
    let mut headers = std::collections::HashMap::new();
    let mut content_length = 0usize;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).is_err() {
            return;
        }
        let line = line.trim_end_matches("\r\n").trim_end_matches('\n');
        if line.is_empty() {
            break;
        }
        if let Some((k, v)) = line.split_once(':') {
            let key = k.trim().to_lowercase();
            let val = v.trim().to_string();
            if key == "content-length" {
                content_length = val.parse().unwrap_or(0);
            }
            headers.insert(key, val);
        }
    }
    // read body
    let mut body = vec![0u8; content_length];
    let _ = reader.read_exact(&mut body);
    let body_str = String::from_utf8_lossy(&body).to_string();

    let (status, resp_body, content_type) = route(method, path, &body_str, &state);

    let response = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        status,
        status_text(status),
        content_type,
        resp_body.len(),
        resp_body
    );
    let _ = stream.write_all(response.as_bytes());
}

fn status_text(code: u16) -> &'static str {
    match code {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        413 => "Payload Too Large",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        _ => "Error",
    }
}

fn route(method: &str, path: &str, body: &str, state: &AppState) -> (u16, String, &'static str) {
    match (method, path) {
        ("GET", "/health") => {
            let v = json!({
                "status": "ok",
                "inference_backend": "service",
                "inference_ready": state.ready,
                "embedding_model": state.embed_model_name,
                "reranker_model": state.reranker_model_name,
            });
            (200, v.to_string(), "application/json")
        }
        ("GET", "/ready") => {
            if state.ready {
                (200, "{\"ready\":true}".to_string(), "application/json")
            } else {
                (503, "{\"ready\":false}".to_string(), "application/json")
            }
        }
        ("GET", "/metrics") => {
            ("200".parse().unwrap_or(200), "# metrics placeholder\n".to_string(), "text/plain")
        }
        ("POST", "/embed") => handle_embed(body, state),
        ("POST", "/rerank") => handle_rerank(body, state),
        _ => (404, "{\"error\":{\"code\":\"not_found\",\"message\":\"not found\"}}".to_string(), "application/json"),
    }
}

fn handle_embed(body: &str, state: &AppState) -> (u16, String, &'static str) {
    let req: Result<EmbedRequest, _> = serde_json::from_str(body);
    let req = match req {
        Ok(r) => r,
        Err(e) => {
            return (
                400,
                format!("{{\"error\":{{\"code\":\"invalid_json\",\"message\":\"{}\"}}}}", e),
                "application/json",
            )
        }
    };
    if let Err(e) = http::validate_embed(&req.texts, state.config.max_texts) {
        return (e.status, serde_json::to_string(&e.body).unwrap(), "application/json");
    }
    let pool = match &state.embed_pool {
        Some(p) => p,
        None => return (503, "{\"error\":{\"code\":\"model_unavailable\",\"message\":\"embedder not loaded\"}}".to_string(), "application/json"),
    };
    let start = Instant::now();
    let texts = req.texts.clone();
    match pool.submit(texts) {
        Ok(emb) => {
            let truncated = 0usize;
            let dim = emb.first().map(|v| v.len()).unwrap_or(state.config.embedding_max_length);
            let resp = EmbedResponse {
                model: state.embed_model_name.clone(),
                dim,
                embeddings: emb,
                truncated,
                max_length: state.config.embedding_max_length,
                latency_ms: start.elapsed().as_secs_f64() * 1000.0,
            };
            (200, serde_json::to_string(&resp).unwrap(), "application/json")
        }
        Err(e) => (e.status, serde_json::to_string(&e.body).unwrap(), "application/json"),
    }
}

fn handle_rerank(body: &str, state: &AppState) -> (u16, String, &'static str) {
    let req: Result<RerankRequest, _> = serde_json::from_str(body);
    let req = match req {
        Ok(r) => r,
        Err(e) => {
            return (
                400,
                format!("{{\"error\":{{\"code\":\"invalid_json\",\"message\":\"{}\"}}}}", e),
                "application/json",
            )
        }
    };
    if let Err(e) = http::validate_rerank(&req, state.config.max_documents) {
        return (e.status, serde_json::to_string(&e.body).unwrap(), "application/json");
    }
    if req.documents.is_empty() {
        let resp = RerankResponse {
            model: state.reranker_model_name.clone(),
            scores: Vec::new(),
            truncated: 0,
            max_length: state.config.reranker_max_length,
            latency_ms: 0.0,
        };
        return (200, serde_json::to_string(&resp).unwrap(), "application/json");
    }
    let pool = match &state.rerank_pool {
        Some(p) => p,
        None => return (503, "{\"error\":{\"code\":\"model_unavailable\",\"message\":\"reranker not loaded\"}}".to_string(), "application/json"),
    };
    let start = Instant::now();
    let rerank_req = req.clone();
    // wrap single request in vec for pool
    match pool.submit(vec![rerank_req]) {
        Ok(mut results) => {
            let r = results.pop().unwrap();
            // map ids+scores
            let mut scores = Vec::new();
            for (i, id) in r.ids.iter().enumerate() {
                scores.push(RerankScore {
                    id: id.clone(),
                    score: r.scores.get(i).copied().unwrap_or(0.0),
                });
            }
            // preserve input order: r.ids already input order
            let resp = RerankResponse {
                model: state.reranker_model_name.clone(),
                scores,
                truncated: r.truncated,
                max_length: state.config.reranker_max_length,
                latency_ms: start.elapsed().as_secs_f64() * 1000.0,
            };
            (200, serde_json::to_string(&resp).unwrap(), "application/json")
        }
        Err(e) => (e.status, serde_json::to_string(&e.body).unwrap(), "application/json"),
    }
}
