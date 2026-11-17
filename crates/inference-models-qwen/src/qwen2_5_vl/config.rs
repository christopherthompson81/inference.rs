// https://github.com/huggingface/transformers/blob/f2c388e3f946862f657acc1e21b272ec946fc66c/src/transformers/models/qwen2_vl/configuration_qwen2_vl.py

pub use crate::vision::qwen2_5_vl::VisionConfig;

pub type Config = crate::qwen2vl::config::QwenVlConfig<VisionConfig>;
