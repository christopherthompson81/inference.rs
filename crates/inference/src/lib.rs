//! # inference, Blazing-Fast LLM Inference in Rust
//!
//! The Rust SDK for [inference.rs](https://github.com/christopherthompson81/inference.rs), a high-performance
//! LLM inference engine supporting text, multimodal, speech, image generation, and embedding models.
//!
//! ## Quick Start
//!
//! ```no_run
//! use inference::{IsqBits, ModelBuilder, TextMessages, TextMessageRole};
//!
//! #[tokio::main]
//! async fn main() -> inference::error::Result<()> {
//!     let model = ModelBuilder::new("Qwen/Qwen3-4B")
//!         .with_auto_isq(IsqBits::Four)
//!         .build()
//!         .await?;
//!
//!     let response = model.chat("What is Rust's ownership model?").await?;
//!     println!("{response}");
//!     Ok(())
//! }
//! ```
//!
//! ## Capabilities
//!
//! | Capability | Builder | Example |
//! |---|---|---|
//! | Any model (auto-detect) | [`ModelBuilder`] | `examples/rust/getting_started/text_generation/` |
//! | Text generation | [`TextModelBuilder`] | `examples/rust/getting_started/text_generation/` |
//! | Multimodal (image+text) | [`MultimodalModelBuilder`] | `examples/rust/getting_started/multimodal/` |
//! | GGUF quantized models | [`GgufModelBuilder`] | `examples/rust/getting_started/gguf/` |
//! | Image generation | [`DiffusionModelBuilder`] | `examples/rust/models/diffusion/` |
//! | Speech synthesis | [`SpeechModelBuilder`] | `examples/rust/models/speech/` |
//! | Embeddings | [`EmbeddingModelBuilder`] | `examples/rust/getting_started/embedding/` |
//! | Structured output | [`Model::generate_structured`] | `examples/rust/advanced/json_schema/` |
//! | Tool calling | [`Tool`], [`ToolChoice`] | `examples/rust/advanced/tools/` |
//! | Agents (the engine's tool loop) | [`TextModelBuilder::with_tool`], [`TextModelBuilder::with_max_tool_rounds`] | `examples/rust/advanced/agent/` |
//! | Multi-model | [`MultiModelBuilder`] | `examples/rust/advanced/multi_model/` |
//! | LoRA / X-LoRA | [`LoraModelBuilder`], [`XLoraModelBuilder`] | `examples/rust/advanced/lora/` |
//! | AnyMoE | [`AnyMoeModelBuilder`] | `examples/rust/advanced/anymoe/` |
//! | MCP client | [`McpClientConfig`] | `examples/rust/advanced/mcp_client/` |
//!
//! ## Model Loading
//!
//! All models are created through builder structs that follow a consistent pattern:
//!
//! ```no_run
//! # use inference::*;
//! # async fn example() -> error::Result<()> {
//! let model = ModelBuilder::new("Qwen/Qwen3-4B")
//!     .with_auto_isq(IsqBits::Four)            // In-situ quantization (auto-selects best type)
//!     .with_logging()                        // Enable logging
//!     .with_paged_attn(PagedAttentionMetaBuilder::default().build()?)
//!     .build()
//!     .await?;
//! # Ok(())
//! # }
//! ```
//!
//! Use [`ModelBuilder::with_auto_isq`] for automatic platform-optimal quantization (e.g., `with_auto_isq(IsqBits::Four)`),
//! or [`ModelBuilder::with_isq`] with a specific [`IsqType`]: `Q4_0`, `Q4_1`, `Q4K`, `Q5_0`, `Q5_1`, `Q5K`,
//! `Q6K`, `Q8_0`, `Q8_1`, `HQQ4`, `HQQ8`, and more.
//!
//! ## Choosing a Request Type
//!
//! | Type | Use When | Sampling |
//! |---|---|---|
//! | [`TextMessages`] | Simple text-only chat, no special settings needed | Greedy |
//! | [`MultimodalMessages`] | Your prompt includes images or audio | Greedy |
//! | [`RequestBuilder`] | You need tools, logprobs, custom sampling, grammars, adapters, or web search | Greedy unless `set_sampler_topk` raises top-k |
//!
//! `TextMessages` and `MultimodalMessages` can be converted into a [`RequestBuilder`] via
//! `Into<RequestBuilder>` if you start simple and later need more control.
//!
//! ## Streaming
//!
//! The stream returned by [`Model::stream_chat_request`] implements
//! [`futures::Stream`] of [`ChatStreamEvent`]s, so you can use `StreamExt` combinators:
//!
//! ```no_run
//! use futures::StreamExt;
//! use inference::*;
//!
//! # async fn example(model: Model) -> error::Result<()> {
//! let messages = TextMessages::new()
//!     .add_message(TextMessageRole::User, "Tell me a joke.");
//!
//! let mut stream = model.stream_chat_request(messages).await?;
//! while let Some(event) = stream.next().await {
//!     if let ChatStreamEvent::Chunk(chunk) = event
//!         && let Some(text) = chunk.choices.first().and_then(|ch| ch.delta.content.as_ref())
//!     {
//!         print!("{text}");
//!     }
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ## Structured Output
//!
//! Derive [`schemars::JsonSchema`] on your type and the model will be constrained to
//! produce valid JSON matching the schema:
//!
//! ```no_run
//! use inference::*;
//! use schemars::JsonSchema;
//! use serde::Deserialize;
//!
//! #[derive(Deserialize, JsonSchema)]
//! struct City {
//!     name: String,
//!     country: String,
//!     population: u64,
//! }
//!
//! # async fn example(model: Model) -> error::Result<()> {
//! let messages = TextMessages::new()
//!     .add_message(TextMessageRole::User, "Give me info about Paris.");
//!
//! let city: City = model.generate_structured::<City>(messages).await?;
//! println!("{}: pop. {}", city.name, city.population);
//! # Ok(())
//! # }
//! ```
//!
//! ## Blocking API
//!
//! For non-async applications, use [`blocking::BlockingModel`]:
//!
//! ```no_run
//! use inference::blocking::BlockingModel;
//! use inference::{IsqBits, ModelBuilder};
//!
//! fn main() -> inference::error::Result<()> {
//!     let model = BlockingModel::from_auto_builder(
//!         ModelBuilder::new("Qwen/Qwen3-4B")
//!             .with_auto_isq(IsqBits::Four),
//!     )?;
//!     let answer = model.chat("What is 2+2?")?;
//!     println!("{answer}");
//!     Ok(())
//! }
//! ```
//!
//! ## Error Handling
//!
//! The SDK's methods return [`error::Result<T>`](error::Result). Its [`error::Error`] carries the engine's own
//! [`Api`](error::Error::Api) errors (whose kind says whether the request or the engine was at fault) and
//! [`ModelLoad`](error::Error::ModelLoad) failures. Engine methods reached through [`Model`]'s `Deref` return
//! [`ApiError`] directly. Both implement `std::error::Error`, so they work with `anyhow` and `eyre`.
//!
//! ## MCP (Model Context Protocol)
//!
//! ```no_run
//! # use inference::*;
//! # async fn example() -> error::Result<()> {
//! let mcp_config = McpClientConfig {
//!     servers: vec![/* your server configs */],
//!     auto_register_tools: true,
//!     tool_timeout_secs: Some(30),
//!     max_concurrent_calls: Some(5),
//! };
//!
//! let model = ModelBuilder::new("path/to/model")
//!     .with_auto_isq(IsqBits::Eight)
//!     .with_mcp_client(mcp_config)
//!     .build()
//!     .await?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Feature Flags
//!
//! | Flag | Effect |
//! |---|---|
//! | `cuda` | CUDA GPU support |
//! | `cudnn` | cuDNN for candle's convolutions (requires `cuda`; slower than the default, not recommended) |
//! | `nccl` | Multi-GPU via NCCL (requires `cuda` and NCCL) |
//! | `metal` | Apple Metal GPU support |
//! | `accelerate` | Apple Accelerate framework |
//! | `mkl` | Intel MKL acceleration |
//!
//! The default feature set (no flags) builds with pure Rust, no C compiler or system
//! libraries required.
//!
//! ## Architecture
//!
//! ```text
//! ModelBuilder / TextModelBuilder / MultimodalModelBuilder / GgufModelBuilder / ...
//!     │
//!     ▼
//!   Model ──── send_chat_request() ──► Engine ──► Pipeline ──► Output
//!     │                                  │
//!     ├── chat()                    Scheduler + PagedAttention
//!     ├── stream_chat_request()
//!     ├── generate_structured()
//!     └── send_*_with_model()       (multi-model dispatch)
//! ```

#[macro_use]
mod load;
mod anymoe;
mod auto_model;
pub mod blocking;
mod diffusion_model;
mod embedding;
mod embedding_model;
pub mod error;
mod gguf;
mod gguf_lora_model;
mod gguf_xlora_model;
mod lora_model;
mod model;
mod multi_model;
mod multimodal_model;
mod request;
mod speech_model;
mod text_model;
mod xlora_model;

pub use anymoe::AnyMoeModelBuilder;
pub use auto_model::ModelBuilder;
pub use diffusion_model::DiffusionModelBuilder;
pub use embedding::EmbeddingRequestBuilder;
pub use embedding_model::{EmbeddingModelBuilder, UqffEmbeddingModelBuilder};
pub use gguf::GgufModelBuilder;
pub use gguf_lora_model::GgufLoraModelBuilder;
pub use gguf_xlora_model::GgufXLoraModelBuilder;
pub use load::{IsqBits, MemoryGpuConfig, PagedAttentionMetaBuilder, ToolCallback};
pub use lora_model::LoraModelBuilder;
pub use model::{ChatEventStream, Model};
pub use multi_model::{IntoModelSpec, MultiModelBuilder};
pub use multimodal_model::{MultimodalModelBuilder, UqffMultimodalModelBuilder};
pub use request::{
    ChatRequest, DrySampling, EncodedKind, InputFile, MessageMedia, MultimodalMessages,
    RequestBuilder, TextMessageRole, TextMessages, empty_chat_request,
};
pub use speech_model::SpeechModelBuilder;
pub use text_model::{TextModelBuilder, UqffTextModelBuilder};
pub use xlora_model::XLoraModelBuilder;

pub use image::DynamicImage;
/// The engine surface the SDK builds on, for its request and response types by their own paths.
pub use inference_api as api;
pub use inference_api::{
    Engine, EngineLoadError, EngineSpec, INFERENCE_RS_GIT_REVISION, INFERENCE_RS_VERSION,
    api_error::{ApiError, ApiErrorKind},
    engine::{
        AgentPermission, AgentToolApproval, AgentToolApprovalDecision, AnyMoeSpec, CalledFunction,
        CodeExecutionConfig, CodeExecutionPermission, DiffusionLoaderType, EngineCallbacks,
        HfConfigOverrides, IsqOrganization, IsqType, LoraAdapterSpec, LoraRuntimeConfig,
        McpClientConfig, ModelDType, ModelSelected, ModelSpec, MtpDraftSampling, NormalLoaderType,
        PagedCacheSpec, PagedCacheType, SearchCallback, SearchEmbeddingModel, SearchResult,
        ShellConfig, SpeechGenerationSpec, SpeechLoaderType, TokenSource, Tool, ToolCallContext,
        ToolCallbackKind, ToolCallbackWithTool, UqffWriteConfig,
    },
    engine::{expand_isq_value, parse_isq_value},
    engine_chat::{AgenticToolCallData, AgenticToolCallPhase, ChatStreamEvent, Usage},
    engine_logits::{LogitsOutput, PromptInput, PromptLogits, PromptLogitsRequest},
    generation::SpeechAudio,
    initialize_logging,
    logits_processors::{CustomLogitsProcessor, in_place},
    media_source::MediaAttachment,
    models::{ModelOperationRequest, ModelStatus},
    openai::{
        AdapterSelection, AudioResponseFormat, ChatCompletionRequest, EmbeddingRequest,
        EmbeddingResponse, EmbeddingVector, Grammar, ImageGenerationRequest, OpenAiTool,
        SpeechGenerationRequest, StopTokens,
    },
    response::{
        ChatCompletionChunkResponse, ChatCompletionResponse, ChunkChoice, Delta,
        ImageGenerationResponse,
    },
    sdk::{
        AllowedToolChoice, AllowedToolsMode, AllowedToolsToolChoice, AllowedToolsToolChoiceType,
        AnyMoeConfig, AnyMoeExpertType, AudioInput, DiffusionGenerationParams, EmbeddingLoaderType,
        File, Function, ImageGenerationResponseFormat, LlguidanceGrammar, MultimodalLoaderType,
        ReasoningEffort, RequestedFile, ToolCallResponse, ToolChoice, ToolType, VideoInput,
        WebSearchOptions, fetch_url, llguidance,
    },
};
pub use inference_macros::tool;
pub use schemars;
