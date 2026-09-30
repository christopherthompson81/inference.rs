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
//! use std::sync::Arc;
//!
//! use axum::{
//!     extract::State,
//!     routing::{get, post},
//!     Json, Router,
//! };
//! use utoipa::OpenApi;
//! use utoipa_swagger_ui::SwaggerUi;
//!
//! use inference_core::{
//!     initialize_logging, AutoDeviceMapParams, ChatCompletionChunkResponse, ModelDType,
//! };
//! use inference_selection::ModelSelected;
//! use inference_server_core::{
//!     chat_completion::{
//!         create_streamer, process_non_streaming_response,
//!         ChatCompletionOnChunkCallback, ChatCompletionOnDoneCallback, ChatCompletionResponder,
//!         ChatEngine,
//!     },
//!     inference_for_server_builder::InferenceRsForServerBuilder,
//!     inference_server_router_builder::{AgenticDefaults, InferenceRsServerRouterBuilder},
//!     openai::{ChatCompletionRequest, OpenAiToolSurface},
//!     openapi_doc::get_openapi_doc,
//!     types::SharedInferenceRsState,
//! };
//!
//! #[derive(OpenApi)]
//! #[openapi(
//!     paths(root, custom_chat),
//!     tags(
//!         (name = "hello", description = "Hello world endpoints")
//!     ),
//!     info(
//!         title = "Hello World API",
//!         version = "1.0.0",
//!         description = "A simple API that responds with a greeting"
//!     )
//! )]
//! struct ApiDoc;
//!
//! #[derive(Clone)]
//! pub struct AppState {
//!     pub inference_state: SharedInferenceRsState,
//!     pub db_create: fn(),
//! }
//!
//! #[tokio::main]
//! async fn main() {
//!     initialize_logging();
//!
//!     let plain_model_id = String::from("meta-llama/Llama-3.2-1B-Instruct");
//!     let tokenizer_json = None;
//!     let arch = None;
//!     let organization = None;
//!     let write_uqff = None;
//!     let from_uqff = None;
//!     let imatrix = None;
//!     let calibration_file = None;
//!     let hf_cache_path = None;
//!
//!     let dtype = ModelDType::Auto;
//!     let topology = None;
//!     let max_seq_len = AutoDeviceMapParams::DEFAULT_MAX_SEQ_LEN;
//!     let max_batch_size = AutoDeviceMapParams::DEFAULT_MAX_BATCH_SIZE;
//!     let matformer_config_path = None;
//!     let matformer_slice_name = None;
//!
//!     let model = ModelSelected::Plain {
//!         model_id: plain_model_id,
//!         tokenizer_json,
//!         arch,
//!         dtype,
//!         topology,
//!         organization,
//!         write_uqff,
//!         from_uqff,
//!         imatrix,
//!         calibration_file,
//!         max_seq_len,
//!         max_batch_size,
//!         hf_cache_path,
//!         matformer_config_path,
//!         matformer_slice_name,
//!     };
//!
//!     let shared_inference = InferenceRsForServerBuilder::new()
//!         .with_model(model)
//!         .with_in_situ_quant("8".to_string())
//!         .set_paged_attn(Some(true))
//!         .build()
//!         .await
//!         .unwrap();
//!
//!     let inference_base_path = "/api/inference";
//!
//!     let inference_routes = InferenceRsServerRouterBuilder::new()
//!         .with_inference(shared_inference.clone())
//!         .with_include_swagger_routes(false)
//!         .with_base_path(inference_base_path)
//!         .build()
//!         .await
//!         .unwrap();
//!
//!     let inference_doc = get_openapi_doc(Some(inference_base_path));
//!
//!     let app_state = Arc::new(AppState {
//!         inference_state: shared_inference,
//!         db_create: mock_db_call,
//!     });
//!
//!     let app = Router::new()
//!         .route("/", get(root))
//!         .route("/chat", post(custom_chat))
//!         .with_state(app_state.clone())
//!         .nest(inference_base_path, inference_routes)
//!         .merge(
//!             SwaggerUi::new("/api-docs")
//!                 .url("/api-docs/openapi.json", ApiDoc::openapi())
//!                 .external_url_unchecked("/api-docs/inference.json", inference_doc),
//!         );
//!
//!     let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
//!     axum::serve(listener, app).await.unwrap();
//!
//!     println!("Listening on 0.0.0.0:3000");
//! }
//!
//! #[utoipa::path(
//!     get,
//!     path = "/",
//!     tag = "hello",
//!     responses(
//!         (status = 200, description = "Successful response with greeting message", body = String)
//!     )
//! )]
//! async fn root() -> &'static str {
//!     "Hello, World!"
//! }
//!
//! #[utoipa::path(
//!     post,
//!     tag = "Custom",
//!     path = "/chat",
//!     request_body = ChatCompletionRequest,
//!     responses((status = 200, description = "Chat completions"))
//! )]
//! pub async fn custom_chat(
//!     State(state): State<Arc<AppState>>,
//!     Json(oai_request): Json<ChatCompletionRequest>,
//! ) -> ChatCompletionResponder {
//!     let inference_state = state.inference_state.clone();
//!
//!     // Through the chat engine, so the agent policy applies; this route has its own defaults and approval broker.
//!     let chat = ChatEngine {
//!         state: inference_state.clone(),
//!         agentic: AgenticDefaults::default(),
//!         skill_store: None,
//!     };
//!     let prepared = match chat
//!         .prepare(oai_request, OpenAiToolSurface::ChatCompletions, Default::default())
//!         .await
//!     {
//!         Ok(prepared) => prepared,
//!         Err(e) => {
//!             let error = e.into_api_error(inference_state.clone());
//!             return ChatCompletionResponder::ValidationError(Box::new(error));
//!         }
//!     };
//!     let is_streaming = prepared.is_streaming;
//!     let mut rx = prepared.rx;
//!
//!     if is_streaming {
//!         let db_fn = state.db_create;
//!
//!         let on_chunk: ChatCompletionOnChunkCallback =
//!             Box::new(move |mut chunk: ChatCompletionChunkResponse| {
//!                 dbg!(&chunk);
//!
//!                 if let Some(original_content) = &chunk.choices[0].delta.content {
//!                     chunk.choices[0].delta.content = Some(format!("CHANGED! {}", original_content));
//!                 }
//!
//!                 chunk.clone()
//!             });
//!
//!         let on_done: ChatCompletionOnDoneCallback =
//!             Box::new(move |chunks: &[ChatCompletionChunkResponse]| {
//!                 dbg!(chunks);
//!                 (db_fn)();
//!             });
//!
//!         let streamer = create_streamer(rx, inference_state.clone(), Some(on_chunk), Some(on_done));
//!
//!         ChatCompletionResponder::Sse(streamer)
//!     } else {
//!         let response = process_non_streaming_response(&mut rx, inference_state.clone()).await;
//!
//!         match &response {
//!             ChatCompletionResponder::Json(json_response) => {
//!                 dbg!(json_response);
//!                 (state.db_create)();
//!             }
//!             _ => {
//!                 //
//!             }
//!         }
//!
//!         response
//!     }
//! }
//!
//! pub fn mock_db_call() {
//!     println!("Saving to DB");
//! }
//! ```

pub mod anthropic;
pub mod approvals;
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
pub mod skills;
pub mod speech_generation;
pub mod streaming;
pub mod types;
use inference_api::{
    agentic, anthropic as anthropic_api, api_error, dispatch, engine_chat, engine_completion,
    engine_embeddings, files as files_api, generation, lora_adapters as lora_adapters_api,
    lora_routing, media_source, models as models_api, responses as responses_api, responses_types,
    skill_store, system,
};
pub use inference_api::{inference_for_server_builder, openai, util, video};
