#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

pub(crate) use inference_models_qwen::qwen2_5_vl::*;

pub(crate) mod inputs_processor;
pub(crate) use inputs_processor::Qwen2_5VLProcessor;
