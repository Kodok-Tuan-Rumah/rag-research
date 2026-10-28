use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Value;
use ort::ep::CPU;
use tokenizers::Tokenizer;
use ndarray::Array2;
use std::time::Instant;

const MAX_SEQ_LEN: usize = 512;

fn main() {
    let env = ort::init().build().unwrap();
    let cpu_ep = CPU::default().build();
    
    let mut session_builder = Session::builder(&env).unwrap();
    session_builder = session_builder.with_execution_providers([cpu_ep]).unwrap();
    session_builder = session_builder.with_optimization_level(GraphOptimizationLevel::Level3).unwrap();
    session_builder = session_builder.with_intra_threads(4).unwrap();
    
    let model_path = "/home/malikimayzar/projects/ai-infra-workspace/rag-research/models/model_optimized.onnx";
    let mut session = session_builder.commit_from_file(model_path).unwrap();
    
    let tokenizer = Tokenizer::from_file("/home/malikimayzar/projects/ai-infra-workspace/rag-research/models/tokenizer.json").unwrap();
    
    let query = "What is self-attention?";
    let text = "Self-attention allows the model to attend to different positions.";
    
    let encoding = tokenizer.encode((query, text), true).unwrap();
    let mut ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
    let mut mask: Vec<i64> = encoding.get_attention_mask().iter().map(|&x| x as i64).collect();
    let mut types: Vec<i64> = encoding.get_type_ids().iter().map(|&x| x as i64).collect();
    
    if ids.len() > MAX_SEQ_LEN {
        ids.truncate(MAX_SEQ_LEN); mask.truncate(MAX_SEQ_LEN); types.truncate(MAX_SEQ_LEN);
    } else {
        let pad = MAX_SEQ_LEN - ids.len();
        ids.extend(std::iter::repeat(0).take(pad));
        mask.extend(std::iter::repeat(0).take(pad));
        types.extend(std::iter::repeat(0).take(pad));
    }
    
    let input_ids_array = Array2::from_shape_vec((1, MAX_SEQ_LEN), ids).unwrap();
    let attention_mask_array = Array2::from_shape_vec((1, MAX_SEQ_LEN), mask).unwrap();
    let token_type_ids_array = Array2::from_shape_vec((1, MAX_SEQ_LEN), types).unwrap();
    
    // Warm up
    for _ in 0..5 {
        let iids = Value::from_array(input_ids_array.clone()).unwrap();
        let amask = Value::from_array(attention_mask_array.clone()).unwrap();
        let ttid = Value::from_array(token_type_ids_array.clone()).unwrap();
        let _ = session.run(ort::inputs![
            "input_ids" => iids,
            "attention_mask" => amask,
            "token_type_ids" => ttid,
        ]);
    }
    
    // Benchmark single inference
    let t0 = Instant::now();
    for _ in 0..100 {
        let iids = Value::from_array(input_ids_array.clone()).unwrap();
        let amask = Value::from_array(attention_mask_array.clone()).unwrap();
        let ttid = Value::from_array(token_type_ids_array.clone()).unwrap();
        let _ = session.run(ort::inputs![
            "input_ids" => iids,
            "attention_mask" => amask,
            "token_type_ids" => ttid,
        ]);
    }
    let t1 = Instant::now();
    println!("Single pair x 100 (optimized): {:?}", t1 - t0);
    println!("Per inference: {:?}", (t1 - t0) / 100);
}
