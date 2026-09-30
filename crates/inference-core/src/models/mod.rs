#[cfg(feature = "models-llama")]
pub(crate) use inference_models_llama::quantized_llama;
#[cfg(all(test, feature = "models-other"))]
pub(crate) use inference_models_other::granite;
