pub mod base_model;
mod chat_template;
pub mod embedding_gguf;
#[cfg(any(test, feature = "models-gemma"))]
pub mod gemma3_bindings;
#[cfg(any(test, feature = "models-gemma"))]
pub mod gemma3_config;
#[cfg(any(test, feature = "models-gemma"))]
pub mod gemma3n_bindings;
mod gguf_tokenizer;
pub mod granite_multimodal_bindings;
pub mod idefics3_bindings;
pub mod lfm2_vl_bindings;
pub mod llama4_bindings;
pub mod metadata;
mod mistral3_bindings;
mod multimodal_binding_utils;
pub mod multimodal_bindings;
pub mod multimodal_vision_registry;
pub mod muse_glimmer_bindings;
pub mod normal_bindings;
pub mod normal_config;
pub mod normal_registry;
pub mod paddleocr_vl_bindings;
pub mod qwen_multimodal_bindings;

pub use chat_template::{get_gguf_chat_template, get_gguf_chat_template_from_metadata};
pub use gguf_tokenizer::{
    GgufTokenizerConversion, convert_gguf_metadata_to_hf_tokenizer,
    validate_external_gguf_tokenizer,
};
pub use inference_nn::gguf::Content;
pub use inference_nn::gguf::GGUFArchitecture;

pub const GGUF_MULTI_FILE_DELIMITER: &str = ";";

/// Whether a GGUF of `architecture` loads here: a text model, or the text side of a multimodal one.
pub fn loads_architecture(architecture: &str) -> bool {
    architecture
        .parse::<normal_registry::CanonicalGgufArchitecture>()
        .is_ok()
        || multimodal_vision_registry::is_multimodal_text_architecture(architecture)
        || qwen_multimodal_bindings::is_qwen_multimodal_architecture(architecture)
}
