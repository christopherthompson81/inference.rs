//! The integration tests, one binary so the stack is monomorphized and linked once.
// Each fixture includes the recording helpers itself, so server-core and ffi can include one fixture alone.
#![allow(clippy::duplicate_mod)]

mod embedding_tiny;
mod gguf_iq;
mod gguf_lora_tiny;
mod llama_tiny;
mod llava_tiny;
mod local_attention_tiny;
mod paddleocr_vl;
mod paddleocr_vl_tiny;
mod qwen3_5_mtp;
mod qwen3_5_text_tiny;
mod qwen_vl_tiny;
