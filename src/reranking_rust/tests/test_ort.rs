use ort::session::Session;
use ort::environment::Environment;
use ort::value::Value;
use ndarray::Array2;

fn main() {
    let env_builder = ort::init();
    let env = match env_builder.build() {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Environment build failed: {}", e);
            return;
        }
    };
    println!("Environment created successfully");

    let mut session_builder = match Session::builder(&env) {
        Ok(sb) => sb,
        Err(e) => {
            eprintln!("Session builder failed: {}", e);
            return;
        }
    };
    println!("Session builder created successfully");

    let editable = match session_builder.edit_from_file("models/model.onnx") {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Edit from file failed: {}", e);
            return;
        }
    };
    println!("Editable session created successfully");

    let session = match editable.into_session() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Into session failed: {}", e);
            return;
        }
    };
    println!("Session created successfully!");

    // Try running inference
    let input_ids = Array2::from_shape_vec((1, 21), vec![1i64; 21]).unwrap();
    let attention_mask = Array2::from_shape_vec((1, 21), vec![1i64; 21]).unwrap();
    let token_type_ids = Array2::from_shape_vec((1, 21), vec![0i64; 21]).unwrap();

    let input_ids_tensor = Value::from_array(input_ids).unwrap();
    let attention_mask_tensor = Value::from_array(attention_mask).unwrap();
    let token_type_ids_tensor = Value::from_array(token_type_ids).unwrap();

    let outputs = session.run(ort::inputs![
        "input_ids" => input_ids_tensor,
        "attention_mask" => attention_mask_tensor,
        "token_type_ids" => token_type_ids_tensor,
    ]);
    
    match outputs {
        Ok(outputs) => {
            let output_value = outputs.get("logits").unwrap();
            let logits: ndarray::ArrayViewD<f32> = output_value.try_extract_array().unwrap();
            println!("Inference successful! Output: {:?}", logits);
        }
        Err(e) => {
            eprintln!("Inference failed: {}", e);
        }
    }
}
