use std::{error::Error, fmt::Debug, sync::Arc};

use inference_tensor::Tensor;

pub use inference_protocol::response::*;

/// Images a diffusion pipeline produced; the caller encodes them, e.g. with `inference_protocol::images`.
#[derive(Debug, Clone)]
pub struct GeneratedImages {
    pub created: u128,
    pub images: Vec<image::DynamicImage>,
}

/// The response enum contains 3 types of variants:
/// - Error (-Error suffix)
/// - Chat (no prefix)
/// - Completion (Completion- prefix)
pub enum Response {
    InternalError(Box<dyn Error + Send + Sync>),
    ValidationError(Box<dyn Error + Send + Sync>),
    // Chat
    ModelError(String, ChatCompletionResponse),
    Done(ChatCompletionResponse),
    Chunk(ChatCompletionChunkResponse),
    // Completion
    CompletionModelError(String, CompletionResponse),
    CompletionDone(CompletionResponse),
    CompletionChunk(CompletionChunkResponse),
    // Image generation
    ImageGeneration(GeneratedImages),
    // Speech generation
    Speech {
        pcm: Arc<Vec<f32>>,
        rate: usize,
        channels: usize,
    },
    // Raw
    Raw {
        logits_chunks: Vec<Tensor>,
        tokens: Vec<u32>,
    },
    Embeddings {
        embeddings: Vec<f32>,
        prompt_tokens: usize,
        total_tokens: usize,
    },
    /// Progress event emitted by the agentic loop during tool execution.
    AgenticToolCallProgress {
        round: usize,
        /// The model's id for the call, which pairs its phases when a round makes several.
        tool_call_id: String,
        tool_name: String,
        phase: AgenticToolCallPhase,
    },
    BlockDenoisingProgress(BlockDenoisingProgress),
    AgenticToolApprovalRequired {
        approval_id: String,
        session_id: String,
        round: usize,
        tool: crate::AgentToolMetadata,
        arguments: serde_json::Value,
    },
    /// Emitted as soon as the runtime reads a file out of the working directory.
    File(crate::files::File),
}
