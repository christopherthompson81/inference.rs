use serde::{Deserialize, Serialize};

use crate::tools::ToolCallResponse;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
/// Top-n logprobs element
pub struct TopLogprob {
    pub token: u32,
    pub logprob: f32,
    pub bytes: Option<String>,
}

pub const SYSTEM_FINGERPRINT: &str = "local";

#[derive(Debug, Clone, Serialize)]
/// Chat completion response message.
pub struct ResponseMessage {
    pub content: Option<String>,
    pub role: String,
    pub tool_calls: Option<Vec<ToolCallResponse>>,
    /// Reasoning/analysis content separated from final content.
    /// This contains chain-of-thought reasoning that is not intended for end users.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
/// Delta in content for streaming response.
pub struct Delta {
    pub content: Option<String>,
    pub role: String,
    pub tool_calls: Option<Vec<ToolCallResponse>>,
    /// Reasoning/analysis content delta.
    /// This contains incremental chain-of-thought reasoning.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
/// A logprob with the top logprobs for this token.
pub struct ResponseLogprob {
    pub token: String,
    pub logprob: f32,
    pub bytes: Option<Vec<u8>>,
    pub top_logprobs: Vec<TopLogprob>,
}

#[derive(Debug, Clone, Serialize)]
/// Logprobs per token.
pub struct Logprobs {
    pub content: Option<Vec<ResponseLogprob>>,
}

#[derive(Debug, Clone, Serialize)]
/// Chat completion choice.
pub struct Choice {
    pub finish_reason: String,
    #[serde(skip)]
    pub stop_sequence: Option<String>,
    pub index: usize,
    pub message: ResponseMessage,
    pub logprobs: Option<Logprobs>,
}

#[derive(Debug, Clone, Serialize)]
/// Chat completion streaming chunk choice.
pub struct ChunkChoice {
    pub finish_reason: Option<String>,
    #[serde(skip)]
    pub stop_sequence: Option<String>,
    pub index: usize,
    pub delta: Delta,
    pub logprobs: Option<ResponseLogprob>,
}

#[derive(Debug, Clone, Serialize)]
/// Chat completion streaming chunk choice.
pub struct CompletionChunkChoice {
    pub text: String,
    pub index: usize,
    pub logprobs: Option<ResponseLogprob>,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
/// OpenAI compatible prompt token breakdown.
pub struct PromptTokensDetails {
    /// Prompt tokens served from the prefix cache, not recomputed.
    pub cached_tokens: usize,
}

#[derive(Debug, Clone, Serialize)]
/// OpenAI compatible (superset) usage during a request.
pub struct Usage {
    pub completion_tokens: usize,
    pub prompt_tokens: usize,
    pub total_tokens: usize,
    /// Present when some prompt tokens were served from the prefix cache.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
    pub avg_tok_per_sec: f32,
    pub avg_prompt_tok_per_sec: f32,
    pub avg_compl_tok_per_sec: f32,
    pub total_time_sec: f32,
    pub total_prompt_time_sec: f32,
    pub total_completion_time_sec: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct AgenticToolCallRecord {
    pub round: usize,
    pub name: String,
    pub arguments: String,
    pub result_content: String,
    /// Base64-encoded PNG images.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub result_images_base64: Vec<String>,
    /// Resolve via the response's top-level `files` or `/v1/files/{id}`.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub file_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
/// An OpenAI compatible chat completion response.
pub struct ChatCompletionResponse {
    pub id: String,
    pub choices: Vec<Choice>,
    pub created: u64,
    pub model: String,
    pub system_fingerprint: String,
    pub object: String,
    pub usage: Usage,
    /// Exact LoRA generation used for this response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub adapter_generation: Option<String>,
    /// Ordered record of all tool calls made during the agentic loop.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agentic_tool_calls: Option<Vec<AgenticToolCallRecord>>,
    /// Files surfaced by tool calls. Bodies inline up to the wire-embed cap; larger files are fetch-by-id.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub files: Option<Vec<crate::files::File>>,
    /// Reuse in later requests to keep agentic state across messages.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
/// Chat completion streaming request chunk.
pub struct ChatCompletionChunkResponse {
    pub id: String,
    pub choices: Vec<ChunkChoice>,
    pub created: u128,
    pub model: String,
    pub system_fingerprint: String,
    pub object: String,
    pub usage: Option<Usage>,
    /// Exact LoRA generation used for this chunk.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub adapter_generation: Option<String>,
    /// Set on the final chunk so streaming clients can read it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
/// Completion request choice.
pub struct CompletionChoice {
    pub finish_reason: String,
    pub index: usize,
    pub text: String,
    pub logprobs: Option<Logprobs>,
}

#[derive(Debug, Clone, Serialize)]
/// An OpenAI compatible completion response.
pub struct CompletionResponse {
    pub id: String,
    pub choices: Vec<CompletionChoice>,
    pub created: u64,
    pub model: String,
    pub system_fingerprint: String,
    pub object: String,
    pub usage: Usage,
    /// Exact LoRA generation used for this response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub adapter_generation: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
/// Completion request choice.
pub struct CompletionChunkResponse {
    pub id: String,
    pub choices: Vec<CompletionChunkChoice>,
    pub created: u128,
    pub model: String,
    pub system_fingerprint: String,
    pub object: String,
    /// Exact LoRA generation used for this chunk.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub adapter_generation: Option<String>,
}

#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, Serialize)]
pub struct ImageChoice {
    pub url: Option<String>,
    pub b64_json: Option<String>,
}

#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, Serialize)]
pub struct ImageGenerationResponse {
    pub created: u128,
    pub data: Vec<ImageChoice>,
}

/// Tool-specific structured progress data for agentic tool calls.
#[derive(Debug, Clone)]
pub enum AgenticToolCallData {
    /// Python code execution.
    CodeExecution {
        code: Option<String>,
        stdout: Option<String>,
        stderr: Option<String>,
        exception: Option<String>,
        images: Vec<image::DynamicImage>,
        video_frames: Vec<image::DynamicImage>,
        video_frame_count: Option<usize>,
        working_directory: Option<String>,
        execution_time_ms: Option<u64>,
    },
    /// Web search or content extraction.
    WebSearch {
        query: Option<String>,
        results_count: Option<usize>,
        sources: Vec<String>,
    },
    /// Shell command execution.
    Shell {
        commands: Vec<String>,
        stdout: Option<String>,
        stderr: Option<String>,
        exit_code: Option<i64>,
        status: Option<String>,
        working_directory: Option<String>,
        timed_out: Option<bool>,
    },
    /// User callback, MCP, or HTTP dispatch. Opaque to the engine.
    Custom { arguments: String, content: String },
}

/// Phase of an agentic tool call.
#[derive(Debug, Clone)]
pub enum AgenticToolCallPhase {
    /// Tool call parsed, about to execute.
    Calling(AgenticToolCallData),
    /// Execution complete.
    Complete(AgenticToolCallData),
}

#[derive(Debug, Clone)]
pub struct BlockDenoisingProgress {
    pub index: usize,
    pub step: usize,
    pub total_steps: usize,
    pub tokens: Vec<u32>,
    pub text: String,
    pub finished: bool,
    pub final_block: bool,
}

#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct ServiceUnavailableError(pub String);
