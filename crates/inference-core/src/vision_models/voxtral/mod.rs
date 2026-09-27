#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

pub(crate) use inference_models_llama::voxtral::*;

pub(crate) mod audio_processing;
pub(crate) mod inputs_processor;
pub(crate) use inputs_processor::VoxtralProcessor;
