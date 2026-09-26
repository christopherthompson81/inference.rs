#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

pub(crate) use inference_models_gemma::gemma4::*;

pub(crate) mod audio_processing;
pub(crate) mod inputs_processor;
pub(crate) use inputs_processor::{Gemma4Processor, Gemma4ProcessorSettings};
