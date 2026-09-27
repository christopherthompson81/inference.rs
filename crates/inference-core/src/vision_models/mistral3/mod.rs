#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

pub(crate) use inference_models_llama::mistral3::*;

pub(crate) use inputs_processor::Mistral3Processor;
pub(crate) mod inputs_processor;
