//! Qwen3-VL-MoE: Qwen3-VL whose text config names experts.

pub mod config;

pub use config::Config;

pub type Qwen3VLMoEModel = crate::qwen3_vl::Qwen3VLModel;
