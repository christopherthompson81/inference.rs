#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

pub(crate) use inference_models_llama::voxtral::*;

pub(crate) mod processor;
pub(crate) use processor::VoxtralProcessor;
