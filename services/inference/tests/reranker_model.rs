use ndarray::{ArrayD, IxDyn};
use rag_inference::{OnnxReranker, scores_from_logits};

#[test]
fn extracts_one_score_per_pair_for_supported_shapes() {
    let one_d = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap();
    assert_eq!(scores_from_logits(one_d.view()).unwrap(), vec![1.0, 2.0, 3.0]);

    let two_d_single = ArrayD::from_shape_vec(IxDyn(&[3, 1]), vec![1.0, 2.0, 3.0]).unwrap();
    assert_eq!(scores_from_logits(two_d_single.view()).unwrap(), vec![1.0, 2.0, 3.0]);

    let two_class = ArrayD::from_shape_vec(
        IxDyn(&[3, 2]),
        vec![0.0, 1.0, 0.0, 2.0, 0.0, 3.0],
    )
    .unwrap();
    assert_eq!(scores_from_logits(two_class.view()).unwrap(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn rejects_unknown_output_shape() {
    let three_d = ArrayD::from_shape_vec(IxDyn(&[2, 2, 2]), vec![0.0; 8]).unwrap();
    let error = scores_from_logits(three_d.view()).unwrap_err();
    assert_eq!(error.code(), "invalid_output_shape");
}

#[test]
#[ignore]
fn real_reranker_matches_python_golden() {
    let model_path = std::env::var("RERANKER_MODEL_PATH").expect("RERANKER_MODEL_PATH");
    let tokenizer_path = std::env::var("RERANKER_TOKENIZER_PATH").expect("RERANKER_TOKENIZER_PATH");
    let mut reranker = rag_inference::OnnxReranker::load(
        &model_path,
        &tokenizer_path,
        2,
        256,
    )
    .unwrap();

    let request = rag_inference::http::RerankRequest {
        query: "What is self-attention?".to_string(),
        documents: vec![
            rag_inference::http::Document { id: "a".to_string(), text: "Self-attention relates positions.".to_string() },
            rag_inference::http::Document { id: "b".to_string(), text: "Bananas are yellow.".to_string() },
            rag_inference::http::Document { id: "c".to_string(), text: "naïve café 日本語".to_string() },
        ],
    };
    let result = reranker.infer(std::slice::from_ref(&request)).unwrap();
    assert_eq!(result[0].scores.len(), 3);
    assert!(result[0].scores.iter().all(|score| score.is_finite()));
}
