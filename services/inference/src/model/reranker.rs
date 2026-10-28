use crate::error::ModelError;
use ndarray::ArrayViewD;
use ort::ep::CPU;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Value;
use crate::model::get_ort_env;
use tokenizers::Tokenizer;

pub fn scores_from_logits(logits: ArrayViewD<f32>) -> Result<Vec<f32>, ModelError> {
    match logits.shape() {
        [batch] => Ok(logits.iter().copied().collect()),
        [batch, 1] => Ok(logits.iter().copied().collect()),
        [batch, 2] => {
            let matrix = logits
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|_| ModelError::InvalidOutputShape)?;
            let b = matrix.shape()[0];
            Ok((0..b).map(|row| matrix[[row, 1]]).collect())
        }
        _ => Err(ModelError::InvalidOutputShape),
    }
}

pub struct OnnxReranker {
    session: Session,
    tokenizer: Tokenizer,
    _env: &'static ort::environment::Environment,
    max_length: usize,
}

impl OnnxReranker {
    pub fn load(
        model_path: &str,
        tokenizer_path: &str,
        intra_threads: usize,
        max_length: usize,
    ) -> Result<Self, ModelError> {
        let env = get_ort_env()?;
        let cpu_ep = CPU::default().build();
        let mut sb = Session::builder(env)
            .map_err(|e| ModelError::Other(format!("ORT session builder: {}", e)))?;
        sb = sb
            .with_execution_providers([cpu_ep])
            .map_err(|e| ModelError::Other(format!("{}", e)))?;
        sb = sb
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| ModelError::Other(format!("{}", e)))?;
        sb = sb
            .with_intra_threads(intra_threads)
            .map_err(|e| ModelError::Other(format!("{}", e)))?;
        let session = sb
            .commit_from_file(model_path)
            .map_err(|e| ModelError::Other(format!("{}", e)))?;
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| ModelError::Other(format!("{}", e)))?;
        Ok(Self {
            session,
            tokenizer,
            _env: env,
            max_length,
        })
    }

    pub fn infer(
        &mut self,
        requests: &[crate::http::RerankRequest],
    ) -> Result<Vec<crate::http::RerankResult>, ModelError> {
        // flatten pairs
        let mut pairs: Vec<(usize, usize)> = Vec::new(); // (request_idx, doc_idx)
        for (ri, req) in requests.iter().enumerate() {
            for di in 0..req.documents.len() {
                pairs.push((ri, di));
            }
        }

        // encode each pair
        let mut all_ids: Vec<Vec<i64>> = Vec::with_capacity(pairs.len());
        let mut all_masks: Vec<Vec<i64>> = Vec::with_capacity(pairs.len());
        let mut all_types: Vec<Vec<i64>> = Vec::with_capacity(pairs.len());
        let mut truncated = 0usize;
        for (ri, di) in &pairs {
            let q = &requests[*ri].query;
            let t = &requests[*ri].documents[*di].text;
            let mut enc = self
                .tokenizer
                .encode((q.as_str(), t.as_str()), true)
                .map_err(|e| ModelError::Other(format!("Tokenize: {}", e)))?;
            let orig_len = enc.get_ids().len();
            let mut ids: Vec<i64> = enc.get_ids().iter().map(|&x| x as i64).collect();
            let mut mask: Vec<i64> = enc.get_attention_mask().iter().map(|&x| x as i64).collect();
            let types: Vec<i64> = enc.get_type_ids().iter().map(|&x| x as i64).collect();
            if ids.len() > self.max_length {
                truncated += 1;
                ids.truncate(self.max_length);
                // mask/types need truncate too
            }
            let mask = if mask.len() > self.max_length {
                mask[..self.max_length].to_vec()
            } else {
                mask
            };
            let types = if types.len() > self.max_length {
                types[..self.max_length].to_vec()
            } else {
                types
            };
            all_ids.push(ids);
            all_masks.push(mask);
            all_types.push(types);
            let _ = orig_len;
        }

        if pairs.is_empty() {
            let mut out = Vec::new();
            for req in requests {
                out.push(crate::http::RerankResult {
                    scores: Vec::new(),
                    ids: req.documents.iter().map(|d| d.id.clone()).collect(),
                    truncated: 0,
                });
            }
            return Ok(out);
        }

        let max_seq_len = all_ids.iter().map(|v| v.len()).max().unwrap_or(0);
        let batch_size = pairs.len();
        let mut input_ids_flat = Vec::with_capacity(batch_size * max_seq_len);
        let mut attention_mask_flat = Vec::with_capacity(batch_size * max_seq_len);
        let mut token_type_ids_flat = Vec::with_capacity(batch_size * max_seq_len);
        for i in 0..batch_size {
            input_ids_flat.extend(all_ids[i].iter().copied());
            attention_mask_flat.extend(all_masks[i].iter().copied());
            token_type_ids_flat.extend(all_types[i].iter().copied());
            let pad = max_seq_len - all_ids[i].len();
            input_ids_flat.extend(std::iter::repeat(0i64).take(pad));
            attention_mask_flat.extend(std::iter::repeat(0i64).take(pad));
            token_type_ids_flat.extend(std::iter::repeat(0i64).take(pad));
        }

        let input_ids_array = ndarray::Array2::from_shape_vec((batch_size, max_seq_len), input_ids_flat)
            .map_err(|e| ModelError::Other(format!("{}", e)))?;
        let attention_mask_array = ndarray::Array2::from_shape_vec((batch_size, max_seq_len), attention_mask_flat)
            .map_err(|e| ModelError::Other(format!("{}", e)))?;
        let token_type_ids_array = ndarray::Array2::from_shape_vec((batch_size, max_seq_len), token_type_ids_flat)
            .map_err(|e| ModelError::Other(format!("{}", e)))?;

        let input_ids_tensor = Value::from_array(input_ids_array)
            .map_err(|e| ModelError::Other(format!("{}", e)))?;
        let attention_mask_tensor = Value::from_array(attention_mask_array)
            .map_err(|e| ModelError::Other(format!("{}", e)))?;
        let token_type_ids_tensor = Value::from_array(token_type_ids_array)
            .map_err(|e| ModelError::Other(format!("{}", e)))?;

        let outputs = self
            .session
            .run(ort::inputs![
                "input_ids" => input_ids_tensor,
                "attention_mask" => attention_mask_tensor,
                "token_type_ids" => token_type_ids_tensor,
            ])
            .map_err(|e| ModelError::Other(format!("{}", e)))?;

        let out_val = outputs
            .get("logits")
            .or_else(|| outputs.get("output"))
            .or_else(|| outputs.get("0"))
            .ok_or_else(|| ModelError::Other("no logits output".to_string()))?;

        let arr: ndarray::ArrayViewD<'_, f32> = out_val
            .try_extract_array()
            .map_err(|e| ModelError::Other(format!("{}", e)))?;

        let scores_flat = scores_from_logits(arr)?;

        // split by request
        // count docs per request
        let mut cursor = 0usize;
        let mut out = Vec::with_capacity(requests.len());
        for req in requests {
            let n = req.documents.len();
            let chunk = &scores_flat[cursor..cursor + n];
            out.push(crate::http::RerankResult {
                scores: chunk.to_vec(),
                ids: req.documents.iter().map(|d| d.id.clone()).collect(),
                truncated,
            });
            cursor += n;
        }
        Ok(out)
    }
}

// keep Document import used
impl crate::worker::BatchModel<crate::http::RerankRequest, crate::http::RerankResult> for OnnxReranker {
    fn infer(&mut self, batch: &[crate::http::RerankRequest]) -> Result<Vec<crate::http::RerankResult>, String> {
        OnnxReranker::infer(self, batch).map_err(|e| e.to_string())
    }
}
