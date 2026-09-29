#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

pub(crate) use inference_models_llama::llama4::*;

pub(crate) use inference_models_llama::llama4::inputs_processor::{
    Llama4ImageProcessor, IMAGE_TOKEN,
};
pub(crate) mod processor;
pub(crate) use processor::Llama4Processor;
