#[cfg(feature = "models-llama")]
pub(crate) use inference_models_llama::xlora::quantized_llama::ModelWeights as XLoraQLlama;
#[cfg(feature = "models-phi")]
pub(crate) use inference_models_phi::xlora::quantized_phi3::ModelWeights as XLoraQPhi3;
pub use inference_nn::xlora::NonGranularState;
pub(crate) use inference_nn::xlora::XLoraConfig;
