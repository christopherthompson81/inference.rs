#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

pub(crate) use inference_models_qwen::qwen3_vl::*;

pub(crate) mod processor;
pub(crate) use processor::Qwen3VLProcessor;
