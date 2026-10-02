//! Qwen3.5 MoE runs on the Qwen3.5 model, whose decoder layers turn sparse when the text config names experts.

pub use crate::qwen3_5::config;
pub use crate::qwen3_5::{Config, Qwen3_5Model as Qwen3_5MoeModel};
