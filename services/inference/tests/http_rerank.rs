use rag_inference::http::{validate_rerank, Document, RerankRequest};

fn req(query: &str, docs: Vec<(&str, &str)>) -> RerankRequest {
    RerankRequest {
        query: query.to_string(),
        documents: docs
            .into_iter()
            .map(|(id, text)| Document {
                id: id.to_string(),
                text: text.to_string(),
            })
            .collect(),
    }
}

#[test]
fn rerank_rejects_blank_query() {
    let r = req("  ", vec![]);
    assert!(validate_rerank(&r, 64).is_err());
}

#[test]
fn rerank_empty_documents_ok() {
    let r = req("q", vec![]);
    assert!(validate_rerank(&r, 64).is_ok());
}

#[test]
fn rerank_rejects_too_many_documents() {
    let mut docs = vec![];
    for i in 0..65 {
        docs.push((format!("id{}", i), "text".to_string()));
    }
    let r = RerankRequest {
        query: "q".to_string(),
        documents: docs
            .into_iter()
            .map(|(id, text)| Document { id, text })
            .collect(),
    };
    assert!(validate_rerank(&r, 64).is_err());
}

#[test]
fn rerank_rejects_blank_id_or_text() {
    let r = req("q", vec![("", "text")]);
    assert!(validate_rerank(&r, 64).is_err());
    let r = req("q", vec![("a", "  ")]);
    assert!(validate_rerank(&r, 64).is_err());
}
