use ort::session::Session;
use ort::value::Value;
use ort::ep::CPU;
use ndarray::Array2;
use std::time::Instant;

fn main() {
    let env = ort::init().build().unwrap();
    let cpu_ep = CPU::default().build();
    
    let mut session_builder = Session::builder(&env).unwrap();
    session_builder = session_builder.with_execution_providers([cpu_ep]).unwrap();
    
    let model_path = "/home/malikimayzar/projects/ai-infra-workspace/rag-research/models/model_optimized.onnx";
    let mut session = session_builder.commit_from_file(model_path).unwrap();
    
    // Single pair with small sequence length (not padded to 512)
    let input_ids = Array2::from_shape_vec((1, 21), vec![1i64; 21]).unwrap();
    let attention_mask = Array2::from_shape_vec((1, 21), vec![1i64; 21]).unwrap();
    let token_type_ids = Array2::from_shape_vec((1, 21), vec![0i64; 21]).unwrap();
    
    // Warm up
    for _ in 0..5 {
        let iids = Value::from_array(input_ids.clone()).unwrap();
        let amask = Value::from_array(attention_mask.clone()).unwrap();
        let ttid = Value::from_array(token_type_ids.clone()).unwrap();
        let _ = session.run(ort::inputs![
            "input_ids" => iids,
            "attention_mask" => amask,
            "token_type_ids" => ttid,
        ]);
    }
    
    // Benchmark
    let t0 = Instant::now();
    for _ in 0..100 {
        let iids = Value::from_array(input_ids.clone()).unwrap();
        let amask = Value::from_array(attention_mask.clone()).unwrap();
        let ttid = Value::from_array(token_type_ids.clone()).unwrap();
        let _ = session.run(ort::inputs![
            "input_ids" => iids,
            "attention_mask" => amask,
            "token_type_ids" => ttid,
        ]);
    }
    let t1 = Instant::now();
    println!("Single pair (21 tokens) x 100: {:?}", t1 - t0);
    println!("Per inference: {:?}", (t1 - t0) / 100);
}
