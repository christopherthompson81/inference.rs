#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

pub(crate) use inference_models_gemma::gemma3::*;

mod inputs_processor;
pub(crate) use inputs_processor::Gemma3Processor;
