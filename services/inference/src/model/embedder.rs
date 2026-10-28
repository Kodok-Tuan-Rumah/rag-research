use crate::error::ModelError;
use ndarray::Array2;
use ort::ep::CPU;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Value;
use crate::model::get_ort_env;
use tokenizers::Tokenizer;

pub struct OnnxEmbedder {
    session: Session,
    tokenizer: Tokenizer,
    _env: &'static ort::environment::Environment,
    max_length: usize,
    intra_threads: usize,
}

impl OnnxEmbedder {
    pub fn load(
        model_path: &str,
        tokenizer_path: &str,
        intra_threads: usize,
        max_length: usize,
    ) -> Result<Self, ModelError> {
        let env = get_ort_env()?;
        let cpu_ep = CPU::default().build();
        let mut session_builder = Session::builder(env)
            .map_err(|e| ModelError::Other(format!("ORT session builder: {}", e)))?;
        session_builder = session_builder
            .with_execution_providers([cpu_ep])
            .map_err(|e| ModelError::Other(format!("ORT with_execution_providers: {}", e)))?;
        session_builder = session_builder
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| ModelError::Other(format!("ORT with_optimization_level: {}", e)))?;
        session_builder = session_builder
            .with_intra_threads(intra_threads)
            .map_err(|e| ModelError::Other(format!("ORT with_intra_threads: {}", e)))?;

        let session = session_builder
            .commit_from_file(model_path)
            .map_err(|e| ModelError::Other(format!("ORT commit_from_file: {}", e)))?;

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| ModelError::Other(format!("Tokenizer load: {}", e)))?;

        Ok(Self {
            session,
            tokenizer,
            _env: env,
            max_length,
            intra_threads,
        })
    }

    pub fn max_length(&self) -> usize {
        self.max_length
    }

    pub fn infer(&mut self, texts: &[String]) -> Result<(Vec<Vec<f32>>, usize), ModelError> {
        if texts.is_empty() {
            return Ok((Vec::new(), 0));
        }

        // Tokenize
        let mut all_encodings: Vec<(Vec<i64>, Vec<i64>)> = Vec::with_capacity(texts.len());
        let mut truncated_count = 0usize;
        for text in texts {
            let enc = self
                .tokenizer
                .encode(text.as_str(), true)
                .map_err(|e| ModelError::Other(format!("Tokenize: {}", e)))?;

            let ids: Vec<i64> = enc.get_ids().iter().map(|&x| x as i64).collect();
            let mask: Vec<i64> = enc
                .get_attention_mask()
                .iter()
                .map(|&x| x as i64)
                .collect();
            // manual truncation to max_length handled below
            let (ids, mask) = if ids.len() > self.max_length {
                (ids[..self.max_length].to_vec(), mask[..self.max_length].to_vec())
            } else {
                (ids, mask)
            };
            all_encodings.push((ids, mask));
        }

        // Recompute truncated count correctly: count texts whose original token len > max_length
        // We'll recompute to avoid double counting
        truncated_count = 0;
        for text in texts {
            if let Ok(enc) = self.tokenizer.encode(text.as_str(), true) {
                if enc.get_ids().len() > self.max_length {
                    truncated_count += 1;
                }
            }
        }

        // pad to max seq len in batch
        let max_seq_len = all_encodings.iter().map(|e| e.0.len()).max().unwrap_or(0);
        let batch_size = texts.len();

        let mut input_ids_flat: Vec<i64> = Vec::with_capacity(batch_size * max_seq_len);
        let mut attention_mask_flat: Vec<i64> = Vec::with_capacity(batch_size * max_seq_len);
        let mut token_type_ids_flat: Vec<i64> = Vec::with_capacity(batch_size * max_seq_len);

        for (ids, mask) in &all_encodings {
            input_ids_flat.extend(ids.iter().copied());
            attention_mask_flat.extend(mask.iter().copied());
            token_type_ids_flat.extend(std::iter::repeat(0i64).take(ids.len()));
            let pad = max_seq_len - ids.len();
            input_ids_flat.extend(std::iter::repeat(0i64).take(pad));
            attention_mask_flat.extend(std::iter::repeat(0i64).take(pad));
            token_type_ids_flat.extend(std::iter::repeat(0i64).take(pad));
        }

        let input_ids_array = Array2::from_shape_vec((batch_size, max_seq_len), input_ids_flat)
            .map_err(|e| ModelError::Other(format!("Array2 input_ids: {}", e)))?;
        let attention_mask_array =
            Array2::from_shape_vec((batch_size, max_seq_len), attention_mask_flat.clone())
                .map_err(|e| ModelError::Other(format!("Array2 attention_mask: {}", e)))?;
        let token_type_ids_array =
            Array2::from_shape_vec((batch_size, max_seq_len), token_type_ids_flat)
                .map_err(|e| ModelError::Other(format!("Array2 token_type_ids: {}", e)))?;

        let input_ids_tensor = Value::from_array(input_ids_array)
            .map_err(|e| ModelError::Other(format!("Tensor input_ids: {}", e)))?;
        let attention_mask_tensor = Value::from_array(attention_mask_array)
            .map_err(|e| ModelError::Other(format!("Tensor attention_mask: {}", e)))?;
        let token_type_ids_tensor = Value::from_array(token_type_ids_array)
            .map_err(|e| ModelError::Other(format!("Tensor token_type_ids: {}", e)))?;

        let outputs = self
            .session
            .run(ort::inputs![
                "input_ids" => input_ids_tensor,
                "attention_mask" => attention_mask_tensor,
                "token_type_ids" => token_type_ids_tensor,
            ])
            .map_err(|e| ModelError::Other(format!("ORT run: {}", e)))?;

        // extract last_hidden_state
        let out_val = outputs
            .get("last_hidden_state")
            .or_else(|| outputs.get("output"))
            .or_else(|| outputs.get("0"))
            .ok_or_else(|| ModelError::Other("No last_hidden_state output".to_string()))?;

        let hidden: ndarray::ArrayViewD<'_, f32> = out_val
            .try_extract_array()
            .map_err(|e| ModelError::Other(format!("Extract output: {}", e)))?;

        let hidden = hidden
            .into_dimensionality::<ndarray::Ix3>()
            .map_err(|_| ModelError::Other("hidden not 3D".to_string()))?;

        let hidden_size = hidden.shape()[2];
        let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(batch_size);
        for b in 0..batch_size {
            // mean pool over tokens where attention_mask==1
            let mut sum = vec![0f32; hidden_size];
            let mut count = 0f32;
            for s in 0..max_seq_len {
                if attention_mask_flat[b * max_seq_len + s] == 1 {
                    for h in 0..hidden_size {
                        sum[h] += hidden[[b, s, h]];
                    }
                    count += 1.0;
                }
            }
            if count > 0.0 {
                for h in 0..hidden_size {
                    sum[h] /= count;
                }
            }
            // L2 normalize
            let norm: f32 = sum.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for h in 0..hidden_size {
                    sum[h] /= norm;
                }
            }
            embeddings.push(sum);
        }

        Ok((embeddings, truncated_count))
    }
}

// trait impl
impl crate::worker::BatchModel<String, Vec<f32>> for OnnxEmbedder {
    fn infer(&mut self, batch: &[String]) -> Result<Vec<Vec<f32>>, String> {
        let (emb, _tr) = OnnxEmbedder::infer(self, batch).map_err(|e| e.to_string())?;
        Ok(emb)
    }
}

