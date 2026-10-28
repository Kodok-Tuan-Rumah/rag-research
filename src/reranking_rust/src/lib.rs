use pyo3::prelude::*;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Value;
use ort::ep::CPU;
use tokenizers::Tokenizer;
use ndarray::Array2;
use std::sync::OnceLock;

/// Global ONNX Runtime environment (created once per process).
static ORT_ENV: OnceLock<ort::environment::Environment> = OnceLock::new();

fn get_ort_env() -> &'static ort::environment::Environment {
    ORT_ENV.get_or_init(|| {
        ort::init().build().expect("Failed to create ORT environment")
    })
}

/// Cross-encoder reranker backed by ONNX Runtime.
///
/// Loads a cross-encoder model (e.g. cross-encoder/ms-marco-MiniLM-L-6-v2
/// exported to ONNX) and scores (query, text) pairs for reranking.
pub struct CrossEncoderReranker {
    session: Session,
    tokenizer: Tokenizer,
    _env: &'static ort::environment::Environment,
}

impl CrossEncoderReranker {
    /// Load the model from an ONNX file and tokenizer from a JSON file.
    pub fn load(model_path: &str, tokenizer_path: &str) -> PyResult<Self> {
        let env = get_ort_env();

        let cpu_ep = CPU::default().build();
        let mut session_builder = Session::builder(env)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("ORT session builder: {}", e)))?;
        session_builder = session_builder
            .with_execution_providers([cpu_ep])
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("ORT with_execution_providers: {}", e)))?;
        session_builder = session_builder
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("ORT with_optimization_level: {}", e)))?;
        session_builder = session_builder
            .with_intra_threads(4)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("ORT with_intra_threads: {}", e)))?;

        let session = session_builder
            .commit_from_file(model_path)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("ORT commit_from_file: {}", e)))?;

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Tokenizer load: {}", e)))?;

        Ok(Self { session, tokenizer, _env: env })
    }

    /// Score a batch of (query, text) pairs.
    ///
    /// Returns a Vec of f32 relevance scores, one per pair.
    pub fn score_pairs(&mut self, pairs: &[(String, String)]) -> PyResult<Vec<f32>> {
        if pairs.is_empty() {
            return Ok(Vec::new());
        }

        // Tokenize all pairs
        let mut all_encodings: Vec<(Vec<i64>, Vec<i64>, Vec<i64>)> = Vec::with_capacity(pairs.len());

        for (query, text) in pairs {
            let encoding = self.tokenizer
                .encode((query.as_str(), text.as_str()), true)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Tokenize: {}", e)))?;

            let ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
            let mask: Vec<i64> = encoding.get_attention_mask().iter().map(|&x| x as i64).collect();
            let types: Vec<i64> = encoding.get_type_ids().iter().map(|&x| x as i64).collect();

            all_encodings.push((ids, mask, types));
        }

        // Find max sequence length in batch (pad to max, not 512)
        let max_seq_len = all_encodings.iter().map(|e| e.0.len()).max().unwrap_or(0);

        let batch_size = pairs.len();

        // Build input tensors with dynamic padding (pad to max seq len in batch)
        let mut input_ids_flat: Vec<i64> = Vec::with_capacity(batch_size * max_seq_len);
        let mut attention_mask_flat: Vec<i64> = Vec::with_capacity(batch_size * max_seq_len);
        let mut token_type_ids_flat: Vec<i64> = Vec::with_capacity(batch_size * max_seq_len);

        for (ids, mask, types) in &all_encodings {
            input_ids_flat.extend(ids.iter().copied());
            attention_mask_flat.extend(mask.iter().copied());
            token_type_ids_flat.extend(types.iter().copied());

            // Pad to max_seq_len
            let pad = max_seq_len - ids.len();
            input_ids_flat.extend(std::iter::repeat(0).take(pad));
            attention_mask_flat.extend(std::iter::repeat(0).take(pad));
            token_type_ids_flat.extend(std::iter::repeat(0).take(pad));
        }

        let input_ids_array = Array2::from_shape_vec((batch_size, max_seq_len), input_ids_flat)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Array2 input_ids: {}", e)))?;
        let attention_mask_array = Array2::from_shape_vec((batch_size, max_seq_len), attention_mask_flat)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Array2 attention_mask: {}", e)))?;
        let token_type_ids_array = Array2::from_shape_vec((batch_size, max_seq_len), token_type_ids_flat)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Array2 token_type_ids: {}", e)))?;

        // Create Ort tensors
        let input_ids_tensor = Value::from_array(input_ids_array)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Tensor input_ids: {}", e)))?;
        let attention_mask_tensor = Value::from_array(attention_mask_array)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Tensor attention_mask: {}", e)))?;
        let token_type_ids_tensor = Value::from_array(token_type_ids_array)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Tensor token_type_ids: {}", e)))?;

        // Run inference
        let outputs = self.session.run(ort::inputs![
            "input_ids" => input_ids_tensor,
            "attention_mask" => attention_mask_tensor,
            "token_type_ids" => token_type_ids_tensor,
        ]).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("ORT run: {}", e)))?;

        // Extract logits from output
        let output_value = outputs
            .get("logits")
            .or_else(|| outputs.get("output"))
            .or_else(|| outputs.get("0"))
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("No logits output found"))?;

        let logits_array: ndarray::ArrayViewD<'_, f32> = output_value
            .try_extract_array()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Extract output: {}", e)))?;

        let scores: Vec<f32> = logits_array.iter().copied().collect();

        Ok(scores)
    }
}

/// Python-facing wrapper that holds an optional reranker instance.
#[pyclass]
struct PyCrossEncoderReranker {
    inner: Option<CrossEncoderReranker>,
}

#[pymethods]
impl PyCrossEncoderReranker {
    #[new]
    fn new() -> Self {
        Self { inner: None }
    }

    /// Load model and tokenizer from files.
    #[pyo3(signature = (model_path, tokenizer_path))]
    fn load_model(&mut self, model_path: &str, tokenizer_path: &str) -> PyResult<()> {
        let reranker = CrossEncoderReranker::load(model_path, tokenizer_path)?;
        self.inner = Some(reranker);
        Ok(())
    }

    /// Score (query, text) pairs.
    ///
    /// Args:
    ///   pairs: List of [query, text] string pairs.
    ///
    /// Returns:
    ///   List of f32 relevance scores.
    fn score_pairs(&mut self, pairs: Vec<(String, String)>) -> PyResult<Vec<f32>> {
        match &mut self.inner {
            Some(reranker) => reranker.score_pairs(&pairs),
            None => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Model not loaded. Call load_model() first.",
            )),
        }
    }

    /// Score a single (query, text) pair.
    fn score_pair(&mut self, query: String, text: String) -> PyResult<f32> {
        match &mut self.inner {
            Some(reranker) => {
                let scores = reranker.score_pairs(&[(query, text)])?;
                Ok(scores.into_iter().next().unwrap_or(0.0))
            }
            None => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Model not loaded. Call load_model() first.",
            )),
        }
    }
}

/// Module entry point.
#[pymodule]
fn cross_encoder_reranker(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCrossEncoderReranker>()?;
    Ok(())
}
