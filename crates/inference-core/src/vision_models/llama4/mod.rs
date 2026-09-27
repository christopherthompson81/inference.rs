#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

pub(crate) use inference_models_llama::llama4::*;

pub(crate) mod inputs_processor;
pub(crate) use inputs_processor::{Llama4ImageProcessor, Llama4Processor, IMAGE_TOKEN};
