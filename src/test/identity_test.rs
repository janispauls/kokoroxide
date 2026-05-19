use ort::{
    inputs,
    session::{builder::GraphOptimizationLevel, Session},
    value::Tensor,
};

pub fn run_identity() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize the ONNX Runtime environment
    //let env = Arc::new(Environment::builder().with_name("id").build()?);

    // Build the session
    let mut session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .commit_from_file("models/identity_v8.onnx")?;

    // Create input tensor
    let input_tensor = Tensor::from_array(([1usize], vec![2.0f32]))?; // ? here

    // Run inference
    let outputs = session.run(inputs![input_tensor])?;
    let output = outputs[0].try_extract_array::<f32>()?;
    let output_view = output.view();
    println!("out = {:?}", output_view.as_slice().unwrap());
    Ok(())
}
