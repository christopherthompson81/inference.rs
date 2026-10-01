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
#[cfg(test)]
mod engine_tests;
pub mod files;
pub mod generation;
pub(crate) mod inference_for_server_builder;
#[doc(hidden)]
pub mod input_files;
pub mod lora_adapters;
#[doc(hidden)]
pub mod lora_routing;
#[doc(hidden)]
pub mod media_source;
pub mod models;
pub use inference_protocol::openai;
pub mod operations;
pub mod request_body;
pub mod responses;
pub use inference_protocol::responses_types;
#[doc(hidden)]
pub mod sampling;
pub mod skill_store;
pub mod system;
pub mod types;
pub mod uqff;
pub mod util;
pub mod video;

pub use engine::{Engine, EngineLoadError, EngineSpec};
pub use inference_core::{
    INFERENCE_RS_GIT_REVISION, INFERENCE_RS_VERSION, LogVerbosity, initialize_inference_logging,
    initialize_logging,
};
pub use inference_core::{REQUEST_QUEUE_DURATION_METRIC, sandbox_key};
pub use inference_protocol::response;
