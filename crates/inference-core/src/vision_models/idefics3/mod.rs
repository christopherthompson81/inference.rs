#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

pub(crate) use inference_models_llama::idefics3::*;

pub(crate) mod inputs_processor;
pub(crate) use inputs_processor::Idefics3Processor;
