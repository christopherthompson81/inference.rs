pub(crate) mod base_model;
mod chat_template;
#[cfg(feature = "models-gemma")]
pub(crate) mod gemma3_bindings;
#[cfg(feature = "models-gemma")]
pub(crate) mod gemma3_config;
#[cfg(feature = "models-gemma")]
pub(crate) mod gemma3n_bindings;
mod gguf_tokenizer;
pub(crate) mod idefics3_bindings;
mod lfm2_vl_bindings;
pub(crate) mod llama4_bindings;
pub(crate) mod metadata;
mod mistral3_bindings;
mod multimodal_binding_utils;
pub(crate) mod multimodal_bindings;
pub(crate) mod multimodal_vision_registry;
pub(crate) mod muse_glimmer_bindings;
pub(crate) mod normal_bindings;
pub(crate) mod normal_config;
pub(crate) mod normal_registry;
pub(crate) mod qwen_multimodal_bindings;

pub(crate) use chat_template::{get_gguf_chat_template, get_gguf_chat_template_from_metadata};
pub(crate) use gguf_tokenizer::{
    GgufTokenizerConversion, convert_gguf_metadata_to_hf_tokenizer,
    validate_external_gguf_tokenizer,
};
pub(crate) use inference_nn::gguf::Content;
pub use inference_nn::gguf::GGUFArchitecture;

pub const GGUF_MULTI_FILE_DELIMITER: &str = ";";
