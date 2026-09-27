#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

pub(crate) use inference_models_qwen::minicpmo::*;

pub(crate) use inputs_processor::MiniCpmOProcessor;
pub(crate) mod inputs_processor;
