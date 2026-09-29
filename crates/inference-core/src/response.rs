use std::{
    error::Error,
    fmt::{Debug, Display},
    sync::Arc,
};

use candle_core::Tensor;

pub use inference_protocol::response::*;

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
    ImageGeneration(ImageGenerationResponse),
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

#[derive(Debug, Clone)]
pub enum ResponseOk {
    // Chat
    Done(ChatCompletionResponse),
    Chunk(ChatCompletionChunkResponse),
    // Completion
    CompletionDone(CompletionResponse),
    CompletionChunk(CompletionChunkResponse),
    // Image generation
    ImageGeneration(ImageGenerationResponse),
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
    // Embeddings
    Embeddings {
        embeddings: Vec<f32>,
        prompt_tokens: usize,
        total_tokens: usize,
    },
    // Agentic tool progress
    AgenticToolCallProgress {
        round: usize,
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
    File(crate::files::File),
}

pub enum ResponseErr {
    InternalError(Box<dyn Error + Send + Sync>),
    ValidationError(Box<dyn Error + Send + Sync>),
    ModelError(String, ChatCompletionResponse),
    CompletionModelError(String, CompletionResponse),
}

impl Display for ResponseErr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InternalError(e) | Self::ValidationError(e) => Display::fmt(e, f),
            Self::ModelError(e, x) => f
                .debug_struct("ChatModelError")
                .field("msg", e)
                .field("incomplete_response", x)
                .finish(),
            Self::CompletionModelError(e, x) => f
                .debug_struct("CompletionModelError")
                .field("msg", e)
                .field("incomplete_response", x)
                .finish(),
        }
    }
}

impl Debug for ResponseErr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InternalError(e) | Self::ValidationError(e) => Debug::fmt(e, f),
            Self::ModelError(e, x) => f
                .debug_struct("ChatModelError")
                .field("msg", e)
                .field("incomplete_response", x)
                .finish(),
            Self::CompletionModelError(e, x) => f
                .debug_struct("CompletionModelError")
                .field("msg", e)
                .field("incomplete_response", x)
                .finish(),
        }
    }
}

impl std::error::Error for ResponseErr {}

impl Response {
    /// Convert the response into a result form.
    pub fn as_result(self) -> Result<ResponseOk, Box<ResponseErr>> {
        match self {
            Self::Done(x) => Ok(ResponseOk::Done(x)),
            Self::Chunk(x) => Ok(ResponseOk::Chunk(x)),
            Self::CompletionDone(x) => Ok(ResponseOk::CompletionDone(x)),
            Self::CompletionChunk(x) => Ok(ResponseOk::CompletionChunk(x)),
            Self::InternalError(e) => Err(Box::new(ResponseErr::InternalError(e))),
            Self::ValidationError(e) => Err(Box::new(ResponseErr::ValidationError(e))),
            Self::ModelError(e, x) => Err(Box::new(ResponseErr::ModelError(e, x))),
            Self::CompletionModelError(e, x) => {
                Err(Box::new(ResponseErr::CompletionModelError(e, x)))
            }
            Self::ImageGeneration(x) => Ok(ResponseOk::ImageGeneration(x)),
            Self::Speech {
                pcm,
                rate,
                channels,
            } => Ok(ResponseOk::Speech {
                pcm,
                rate,
                channels,
            }),
            Self::Raw {
                logits_chunks,
                tokens,
            } => Ok(ResponseOk::Raw {
                logits_chunks,
                tokens,
            }),
            Self::Embeddings {
                embeddings,
                prompt_tokens,
                total_tokens,
            } => Ok(ResponseOk::Embeddings {
                embeddings,
                prompt_tokens,
                total_tokens,
            }),
            Self::AgenticToolCallProgress {
                round,
                tool_name,
                phase,
            } => Ok(ResponseOk::AgenticToolCallProgress {
                round,
                tool_name,
                phase,
            }),
            Self::BlockDenoisingProgress(progress) => {
                Ok(ResponseOk::BlockDenoisingProgress(progress))
            }
            Self::AgenticToolApprovalRequired {
                approval_id,
                session_id,
                round,
                tool,
                arguments,
            } => Ok(ResponseOk::AgenticToolApprovalRequired {
                approval_id,
                session_id,
                round,
                tool,
                arguments,
            }),
            Self::File(f) => Ok(ResponseOk::File(f)),
        }
    }
}
