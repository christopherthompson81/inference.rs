#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

// Input processing is identical to Qwen3-VL.
pub(crate) use crate::vision_models::qwen3_vl::Qwen3VLProcessor as Qwen3_5Processor;
pub(crate) use inference_models_qwen::qwen3_5::*;
