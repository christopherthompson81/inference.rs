//! Engine operations outside the request protocols: requantization, online calibration, agentic sessions and
//! tokenization.

use either::Either;
use futures::future::BoxFuture;
use inference_core::{
    CalibrationAction, CalibrationRequest, CalibrationStatus, DetokenizationRequest,
    InferenceRsError, Request, SerializedSession, TokenizationRequest, parse_isq_value,
};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::api_error::{ApiError, ApiErrorKind};
use crate::types::SharedInferenceRsState;

const SESSION_NOT_FOUND: &str = "session_not_found";
const INVALID_SESSION: &str = "invalid_session";
const INVALID_ISQ: &str = "invalid_isq";
const CALIBRATION_FAILED: &str = "calibration_failed";
const TOKENIZATION_FAILED: &str = "tokenization_failed";

#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct ReIsqRequest {
    /// The ISQ type to requantize to, e.g. `Q4K`; numeric shorthands resolve as they would on the CPU.
    #[schema(example = "Q4K")]
    pub ggml_type: String,
}

/// Answered once the requantization is queued behind the requests already running.
#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct ReIsqResponse {
    pub ggml_type: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct CalibrationApplyRequest {
    /// Also save the collected importance matrix to this `.cimatrix` path; over HTTP, a bare file name written to
    /// the server's working directory.
    #[serde(default)]
    pub save_cimatrix: Option<String>,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct SessionList {
    pub data: Vec<String>,
}

/// Branches a session into a new one, named by the engine, with the source's first `num_turns` turns (0 copies all).
#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct SessionForkRequest {
    pub num_turns: usize,
}

/// The tools the engine's MCP servers provide to the default model; built-in tools aren't listed.
#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct McpToolList {
    #[schema(example = "list")]
    pub object: &'static str,
    pub data: Vec<McpToolObject>,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct McpToolObject {
    pub name: String,
    pub description: Option<String>,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct SessionStored {
    pub id: String,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct SessionDeleted {
    pub id: String,
    /// False when there was no such session.
    pub deleted: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct TokenizeRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub text: String,
    #[serde(default = "default_true")]
    #[schema(default = true)]
    pub add_special_tokens: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct TokenizeResponse {
    pub tokens: Vec<u32>,
}

#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct DetokenizeRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub tokens: Vec<u32>,
    #[serde(default = "default_true")]
    #[schema(default = true)]
    pub skip_special_tokens: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct DetokenizeResponse {
    pub text: String,
}

fn default_true() -> bool {
    true
}

fn engine_error(error: InferenceRsError) -> ApiError {
    ApiError::from_error(&error, ApiErrorKind::Internal)
}

async fn send(
    state: &SharedInferenceRsState,
    model: Option<&str>,
    request: Request,
) -> Result<(), ApiError> {
    let sender = state.get_sender(model).map_err(engine_error)?;
    sender.send(request).await.map_err(|_| ApiError::internal())
}

// The engine answers these on a one-shot channel, a dropped sender meaning it stopped; its errors are about the
// request (no ISQ, nothing collected, no tokenizer), so they keep their message.
async fn answer<T>(
    mut rx: tokio::sync::mpsc::Receiver<anyhow::Result<T>>,
    code: &'static str,
) -> Result<T, ApiError> {
    match rx.recv().await {
        Some(Ok(value)) => Ok(value),
        Some(Err(error)) => Err(ApiError::new(
            ApiErrorKind::InvalidRequest,
            format!("{error:#}"),
            Some(code),
            None,
        )),
        None => Err(ApiError::internal()),
    }
}

pub async fn re_isq(
    state: &SharedInferenceRsState,
    request: ReIsqRequest,
) -> Result<ReIsqResponse, ApiError> {
    let level = parse_isq_value(&request.ggml_type, None).map_err(|error| {
        ApiError::new(
            ApiErrorKind::InvalidRequest,
            error,
            Some(INVALID_ISQ),
            Some("ggml_type"),
        )
    })?;
    send(state, None, Request::ReIsq(level)).await?;
    Ok(ReIsqResponse {
        ggml_type: request.ggml_type,
    })
}

pub fn calibration<'a>(
    state: &'a SharedInferenceRsState,
    action: CalibrationAction,
) -> BoxFuture<'a, Result<CalibrationStatus, ApiError>> {
    Box::pin(calibration_inner(state, action))
}

async fn calibration_inner(
    state: &SharedInferenceRsState,
    action: CalibrationAction,
) -> Result<CalibrationStatus, ApiError> {
    let (tx, rx) = tokio::sync::mpsc::channel(1);
    let request = Request::Calibration(CalibrationRequest {
        action,
        response: tx,
    });
    send(state, None, request).await?;
    answer(rx, CALIBRATION_FAILED).await
}

pub fn list_sessions(
    state: &SharedInferenceRsState,
    owner: Option<&str>,
) -> Result<SessionList, ApiError> {
    let data = state.list_session_ids(None, owner).map_err(engine_error)?;
    Ok(SessionList { data })
}

pub fn export_session(
    state: &SharedInferenceRsState,
    session_id: &str,
    owner: Option<&str>,
) -> Result<SerializedSession, ApiError> {
    state
        .export_session(None, session_id, owner)
        .map_err(engine_error)?
        .ok_or_else(|| {
            ApiError::new(
                ApiErrorKind::NotFound,
                format!("Session '{session_id}' was not found."),
                Some(SESSION_NOT_FOUND),
                Some("session_id"),
            )
        })
}

/// Replaces any session already under `session_id`.
pub fn import_session(
    state: &SharedInferenceRsState,
    session_id: String,
    session: SerializedSession,
    owner: Option<&str>,
) -> Result<(), ApiError> {
    state
        .import_session(None, session_id, session, owner)
        .map_err(|error| match error {
            InferenceRsError::Other(message) => ApiError::new(
                ApiErrorKind::InvalidRequest,
                message,
                Some(INVALID_SESSION),
                None,
            ),
            error => engine_error(error),
        })
}

/// Copies `src_session_id`'s first `num_turns` turns into a new session `session_id`, so the two diverge from there.
pub fn fork_session(
    state: &SharedInferenceRsState,
    src_session_id: &str,
    session_id: String,
    num_turns: usize,
    owner: Option<&str>,
) -> Result<(), ApiError> {
    state
        .fork_session(None, src_session_id, session_id, num_turns, owner)
        .map_err(|error| match error {
            InferenceRsError::Other(message) => ApiError::new(
                ApiErrorKind::InvalidRequest,
                message,
                Some(INVALID_SESSION),
                Some("src_session_id"),
            ),
            error => engine_error(error),
        })
}

pub fn delete_session(
    state: &SharedInferenceRsState,
    session_id: &str,
    owner: Option<&str>,
) -> Result<SessionDeleted, ApiError> {
    let deleted = state
        .delete_session(None, session_id, owner)
        .map_err(engine_error)?;
    Ok(SessionDeleted {
        id: session_id.to_string(),
        deleted,
    })
}

pub async fn tokenize(
    state: &SharedInferenceRsState,
    request: TokenizeRequest,
) -> Result<TokenizeResponse, ApiError> {
    let (tx, rx) = tokio::sync::mpsc::channel(1);
    let tokenize = Request::Tokenize(TokenizationRequest {
        text: Either::Right(request.text),
        tools: None,
        add_generation_prompt: false,
        add_special_tokens: request.add_special_tokens,
        enable_thinking: None,
        reasoning_effort: None,
        response: tx,
    });
    send(state, request.model.as_deref(), tokenize).await?;
    Ok(TokenizeResponse {
        tokens: answer(rx, TOKENIZATION_FAILED).await?,
    })
}

pub async fn detokenize(
    state: &SharedInferenceRsState,
    request: DetokenizeRequest,
) -> Result<DetokenizeResponse, ApiError> {
    let (tx, rx) = tokio::sync::mpsc::channel(1);
    let detokenize = Request::Detokenize(DetokenizationRequest {
        tokens: request.tokens,
        skip_special_tokens: request.skip_special_tokens,
        response: tx,
    });
    send(state, request.model.as_deref(), detokenize).await?;
    Ok(DetokenizeResponse {
        text: answer(rx, TOKENIZATION_FAILED).await?,
    })
}
