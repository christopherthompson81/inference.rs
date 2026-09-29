#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

pub(crate) use inference_models_gemma::gemma4::*;

pub(crate) mod processor;
pub(crate) use processor::{Gemma4Processor, Gemma4ProcessorSettings};
