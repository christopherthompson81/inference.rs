//! The inference.rs engine surface, free of HTTP: the HTTP server is built on it and the C ABI exposes it.

pub mod agentic;
pub mod anthropic;
pub mod api_error;
pub mod background_tasks;
pub mod blocking;
pub mod cached_responses;
pub mod dispatch;
pub mod engine;
pub mod engine_chat;
pub mod engine_completion;
pub mod engine_embeddings;
pub mod generation;
pub mod inference_for_server_builder;
#[doc(hidden)]
pub mod input_files;
pub mod lora_adapters;
#[doc(hidden)]
pub mod lora_routing;
#[doc(hidden)]
pub mod media_source;
pub mod models;
pub mod openai;
pub mod responses;
pub mod responses_types;
#[doc(hidden)]
pub mod sampling;
pub mod skill_store;
pub mod types;
pub mod util;
pub mod video;

pub use engine::{Engine, EngineLoadError, EngineSpec};
