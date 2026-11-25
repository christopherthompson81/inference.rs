//! Engine operations outside the request protocols: requantization, online calibration, control vectors, agentic
//! sessions and tokenization.

use either::Either;
use futures::future::BoxFuture;
pub use inference_core::{CalibrationAction, CalibrationStatus, SerializedSession};
use inference_core::{
    CalibrationRequest, ControlVectorRequest, DetokenizationRequest, InferenceRsError,
    RequantizeRequest, Request, TokenizationRequest, parse_isq_value,
};
use serde::Serialize;
use utoipa::ToSchema;

use crate::api_error::{ApiError, ApiErrorKind};
use crate::types::SharedInferenceRsState;

const SESSION_NOT_FOUND: &str = "session_not_found";
const INVALID_SESSION: &str = "invalid_session";
const INVALID_ISQ: &str = "invalid_isq";
const CALIBRATION_FAILED: &str = "calibration_failed";
const REQUANTIZE_FAILED: &str = "requantize_failed";
const CONTROL_VECTORS_FAILED: &str = "control_vectors_failed";
const TOKENIZATION_FAILED: &str = "tokenization_failed";

#[derive(Debug, Clone, inference_json::FromJson, Serialize, ToSchema)]
pub struct ReIsqRequest {
    /// The ISQ type to requantize to, e.g. `Q4K`; numeric shorthands resolve as they would on the CPU.
    #[schema(example = "Q4K")]
    pub ggml_type: String,
    /// The model to requantize; the default model when absent.
    #[serde(default)]
    pub model: Option<String>,
}

/// Answered once the model is requantized; a model not loaded with ISQ is a `requantize_failed` error.
#[derive(Debug, Clone, inference_json::FromJson, Serialize, ToSchema)]
pub struct ReIsqResponse {
    pub ggml_type: String,
}

#[derive(Debug, Clone, inference_json::FromJson, Serialize, ToSchema)]
pub struct CalibrationApplyRequest {
    /// Also save the collected importance matrix to this `.cimatrix` path; over HTTP, a bare file name written to
    /// the server's working directory.
    #[serde(default)]
    pub save_cimatrix: Option<String>,
    /// The model to requantize; the default model when absent.
    #[serde(default)]
    pub model: Option<String>,
}

/// The model a calibration start or status acts on; the default model when absent.
#[derive(Debug, Clone, Default, inference_json::FromJson, Serialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct CalibrationTarget {
    #[serde(default)]
    pub model: Option<String>,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct SessionList {
    pub data: Vec<String>,
}

/// Branches a session into a new one, named by the engine, with the source's first `num_turns` turns (0 copies all).
#[derive(Debug, Clone, inference_json::FromJson, Serialize, ToSchema)]
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

#[derive(Debug, Clone, inference_json::FromJson, Serialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct TokenizeRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub text: String,
    #[serde(default = "default_true")]
    #[schema(default = true)]
    pub add_special_tokens: bool,
}

#[derive(Debug, Clone, inference_json::FromJson, Serialize, ToSchema)]
pub struct TokenizeResponse {
    pub tokens: Vec<u32>,
}

#[derive(Debug, Clone, inference_json::FromJson, Serialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct DetokenizeRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub tokens: Vec<u32>,
    #[serde(default = "default_true")]
    #[schema(default = true)]
    pub skip_special_tokens: bool,
}

#[derive(Debug, Clone, inference_json::FromJson, Serialize, ToSchema)]
pub struct DetokenizeResponse {
    pub text: String,
}

// `default` names the default model, which the engine reaches without a model id.
fn targeted(model: Option<&str>) -> Option<&str> {
    model.filter(|model| *model != crate::lora_routing::DEFAULT_MODEL_ID)
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
    let sender = state.get_sender(targeted(model)).map_err(engine_error)?;
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

fn default_scale() -> f32 {
    1.0
}

/// A control vector file (llama.cpp's GGUF, `direction.<layer>`, or a safetensors file with those names) and the
/// scale its directions are added at.
#[derive(Debug, Clone, PartialEq, inference_json::FromJson, Serialize, ToSchema)]
pub struct ControlVectorFile {
    /// The file's path on the engine's host.
    pub path: String,
    #[serde(default = "default_scale")]
    pub scale: f32,
}

/// A control vector derived on the loaded model from contrasting texts (llama.cpp's `cvector-generator` `mean`
/// method): each positive text pairs with the negative one at its index, and the vector is their layer outputs' mean
/// difference, normalized per layer. Derived once and cached, keyed on the model and the texts.
#[derive(Debug, Clone, PartialEq, inference_json::FromJson, Serialize, ToSchema)]
pub struct DerivedControlVector {
    pub positive: Vec<String>,
    pub negative: Vec<String>,
    #[serde(default = "default_scale")]
    pub scale: f32,
    /// `mean`, the only method so far, when absent.
    #[serde(default)]
    pub method: Option<String>,
}

/// The control vectors a model is steered by: each layer's output gains the files' (and derived vectors') directions
/// for it, summed at their scales, within `layer_range` when one is given. A looped model's repeated layers add theirs
/// on every pass. A derived vector is reported back as its cached file.
#[derive(Debug, Clone, Default, PartialEq, inference_json::FromJson, Serialize, ToSchema)]
pub struct ControlVectors {
    #[serde(default)]
    pub files: Vec<ControlVectorFile>,
    #[serde(default)]
    pub derived: Vec<DerivedControlVector>,
    /// The first and last layer steered (0-based, inclusive); every layer a file names when absent.
    #[serde(default)]
    pub layer_range: Option<[usize; 2]>,
    /// A looped model's (Inkling 2's) loop guidance strength, LoopCD-Hidden (arXiv:2610.02185): the coda's input
    /// pushed away from the first pass's output, `h + w (h - h1)`. About 0.2 to 0.3 for generation; absent, none.
    #[serde(default)]
    pub loop_guidance: Option<f32>,
}

impl ControlVectors {
    pub(crate) fn to_set(&self) -> inference_core::control_vectors::ControlVectorSet {
        inference_core::control_vectors::ControlVectorSet {
            files: self
                .files
                .iter()
                .map(|f| inference_core::control_vectors::ControlVectorFile {
                    path: f.path.clone().into(),
                    scale: f.scale,
                })
                .collect(),
            layers: self.layer_range.map(|[first, last]| (first, last)),
            loop_guidance: self.loop_guidance,
        }
    }

    fn of(set: &inference_core::control_vectors::ControlVectorSet) -> Self {
        Self {
            files: set
                .files
                .iter()
                .map(|f| ControlVectorFile {
                    path: f.path.to_string_lossy().into_owned(),
                    scale: f.scale,
                })
                .collect(),
            derived: Vec::new(),
            layer_range: set.layers.map(|(first, last)| [first, last]),
            loop_guidance: set.loop_guidance,
        }
    }
}

/// Steers a model (the default one unless `model` names another) by `vectors` from its next step; empty `files`
/// stops. Changing the files or their scales on a steered model rewrites its vectors in place, keeping its CUDA graphs.
#[derive(Debug, Clone, inference_json::FromJson, Serialize, ToSchema)]
pub struct SetControlVectorsRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub vectors: ControlVectors,
}

/// Which model's control vectors to read: the default one when absent.
#[derive(Debug, Clone, Default, inference_json::FromJson, Serialize, ToSchema)]
pub struct ControlVectorsQuery {
    #[serde(default)]
    pub model: Option<String>,
}

/// The control vectors a model is steered by after the request.
#[derive(Debug, Clone, inference_json::FromJson, Serialize, ToSchema)]
pub struct ControlVectorsResponse {
    pub vectors: ControlVectors,
}

pub(crate) async fn control_vectors(
    state: &SharedInferenceRsState,
    model: Option<&str>,
    vectors: Option<&ControlVectors>,
) -> Result<ControlVectorsResponse, ApiError> {
    let ask = |set| async move {
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        let request = Request::ControlVectors(ControlVectorRequest { set, response: tx });
        send(state, model, request).await?;
        answer(rx, CONTROL_VECTORS_FAILED).await
    };
    let set = match vectors {
        // derived vectors become their cached files, derived on the model first where the cache lacks them
        Some(vectors) if !vectors.derived.is_empty() => {
            let model_name = ask(None).await?.model;
            let mut resolved = vectors.clone();
            for derived in &vectors.derived {
                let path =
                    crate::control_vector_derive::derived_file(state, model, &model_name, derived)
                        .await?;
                resolved.files.push(ControlVectorFile {
                    path: path.to_string_lossy().into_owned(),
                    scale: derived.scale,
                });
            }
            resolved.derived.clear();
            Some(resolved.to_set())
        }
        Some(vectors) => Some(vectors.to_set()),
        None => None,
    };
    let state = ask(set).await?;
    Ok(ControlVectorsResponse {
        vectors: ControlVectors::of(&state.set),
    })
}

pub(crate) async fn re_isq(
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
    let (tx, rx) = tokio::sync::mpsc::channel(1);
    let request_isq = Request::ReIsq(RequantizeRequest {
        isq: level,
        response: tx,
    });
    send(state, request.model.as_deref(), request_isq).await?;
    answer(rx, REQUANTIZE_FAILED).await?;
    Ok(ReIsqResponse {
        ggml_type: request.ggml_type,
    })
}

pub(crate) fn calibration<'a>(
    state: &'a SharedInferenceRsState,
    action: CalibrationAction,
    model: Option<&'a str>,
) -> BoxFuture<'a, Result<CalibrationStatus, ApiError>> {
    Box::pin(calibration_inner(state, action, model))
}

async fn calibration_inner(
    state: &SharedInferenceRsState,
    action: CalibrationAction,
    model: Option<&str>,
) -> Result<CalibrationStatus, ApiError> {
    let (tx, rx) = tokio::sync::mpsc::channel(1);
    let request = Request::Calibration(CalibrationRequest {
        action,
        response: tx,
    });
    send(state, model, request).await?;
    answer(rx, CALIBRATION_FAILED).await
}

pub(crate) fn list_sessions(
    state: &SharedInferenceRsState,
    owner: Option<&str>,
) -> Result<SessionList, ApiError> {
    let data = state.list_session_ids(None, owner).map_err(engine_error)?;
    Ok(SessionList { data })
}

pub(crate) fn export_session(
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
pub(crate) fn import_session(
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
pub(crate) fn fork_session(
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

pub(crate) fn delete_session(
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

pub(crate) async fn tokenize(
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

pub(crate) async fn detokenize(
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
