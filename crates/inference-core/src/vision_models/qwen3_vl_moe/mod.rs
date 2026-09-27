#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

pub(crate) use inference_models_qwen::qwen3_vl_moe::*;

pub(crate) use crate::vision_models::qwen3_vl::Qwen3VLProcessor as Qwen3VLMoEProcessor;
