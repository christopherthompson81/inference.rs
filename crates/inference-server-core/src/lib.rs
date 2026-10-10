//! > **inference.rs server core**
//!
//! ## About
//!
//! This crate powers inference.rs server. It exposes the underlying functionality
//! allowing others to implement and extend the server implementation.
//!
//! ### Features
//! 1. Incorporate inference.rs server into another axum.rs project.
//! 2. Hook into the inference.rs server lifecycle.
//!
//! ### Example
//! ```no_run
//! use axum::{extract::State, routing::post, Json, Router};
//! use inference_api::{Engine, EngineSpec};
//! use inference_api::response::ChatCompletionChunkResponse;
//! use inference_server_core::{
//!     chat_completion::{
//!         create_streamer, ChatCompletionOnChunkCallback, ChatCompletionOnDoneCallback,
//!         ChatCompletionResponder,
//!     },
//!     inference_server_router_builder::InferenceRsServerRouterBuilder,
//!     openai::ChatCompletionRequest,
//! };
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let spec: EngineSpec = serde_json::from_value(serde_json::json!({
//!         "model": {"Plain": {"model_id": "meta-llama/Llama-3.2-1B-Instruct"}},
//!         "runtime": {"isq": "8"}
//!     }))?;
//!     let engine = Engine::load(spec).await?;
//!
//!     let inference_base_path = "/api/inference";
//!     let inference_routes = InferenceRsServerRouterBuilder::new()
//!         .with_engine(&engine)
//!         .with_include_swagger_routes(false)
//!         .build()
//!         .await?;
//!
//!     let app = Router::new()
//!         .route("/chat", post(custom_chat))
//!         .with_state(engine)
//!         .nest(inference_base_path, inference_routes);
//!
//!     let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
//!     axum::serve(listener, app).await?;
//!     Ok(())
//! }
//!
//! // Through the engine, so its agent policy applies; the callbacks see each chunk and then the whole stream.
//! async fn custom_chat(
//!     State(engine): State<Engine>,
//!     Json(request): Json<ChatCompletionRequest>,
//! ) -> ChatCompletionResponder {
//!     if !request.stream.unwrap_or(false) {
//!         return match engine.chat(request, Default::default()).await {
//!             Ok(response) => ChatCompletionResponder::Json(response),
//!             Err(error) => ChatCompletionResponder::Error(error),
//!         };
//!     }
//!     let stream = match engine.chat_stream(request, Default::default()).await {
//!         Ok(stream) => stream,
//!         Err(error) => return ChatCompletionResponder::Error(error),
//!     };
//!     let on_chunk: ChatCompletionOnChunkCallback =
//!         Box::new(|mut chunk: ChatCompletionChunkResponse| {
//!             if let Some(content) = &chunk.choices[0].delta.content {
//!                 chunk.choices[0].delta.content = Some(content.to_uppercase());
//!             }
//!             chunk
//!         });
//!     let on_done: ChatCompletionOnDoneCallback =
//!         Box::new(|chunks: &[ChatCompletionChunkResponse]| println!("{} chunks", chunks.len()));
//!     ChatCompletionResponder::Sse(create_streamer(stream, Some(on_chunk), Some(on_done)))
//! }
//! ```

pub mod anthropic;
pub mod approvals;
pub mod auth;
pub mod chat_completion;
mod completion_core;
pub mod completions;
pub mod embeddings;
pub mod files;
pub mod handler_core;
mod handlers;
pub mod image_generation;
pub mod lora_adapters;
pub mod mcp_server;
pub use media_source::configure_ui_upload_dir;
pub mod inference_server_router_builder;
pub mod metrics;
pub mod openapi_doc;
pub mod responses;
pub mod route_registry;
pub mod serve;
pub mod skills;
pub mod speech_generation;
pub mod streaming;
pub mod transcription;
pub mod types;
use inference_api::{
    agentic, anthropic as anthropic_api, api_error, engine_chat, engine_completion,
    files as files_api, lora_adapters as lora_adapters_api, media_source, models as models_api,
    responses as responses_api, responses_types, skill_store, system,
};
pub use inference_api::{openai, util, video};
