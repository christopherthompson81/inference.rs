//! OpenAI embeddings as an engine operation.

use anyhow::{Context, Error as AnyhowError, Result, anyhow};
use base64::{Engine, prelude::BASE64_STANDARD};
use futures::future::join_all;
use inference_core::{
    Constraint, InferenceRs, NormalRequest, Request, RequestMessage, Response, SamplingParams,
};
use tokio::sync::mpsc::Receiver;

use crate::{
    api_error::{ApiError, ApiErrorKind, ModelErrorMessage},
    dispatch::{
        base_process_non_streaming_response, create_response_channel, response_model_id,
        send_request_with_model,
    },
    openai::{
        EmbeddingData, EmbeddingEncodingFormat, EmbeddingInput, EmbeddingRequest,
        EmbeddingResponse, EmbeddingUsage, EmbeddingVector,
    },
    types::SharedInferenceRsState,
    util::validate_model_name,
};

struct EmbeddingWithUsage {
    embedding: Vec<f32>,
    prompt_tokens: usize,
    total_tokens: usize,
}

/// Why an embeddings request failed: the caller's request, or the engine. Kept as `anyhow` so the cause survives.
pub enum EmbeddingError {
    Validation(AnyhowError),
    Internal(AnyhowError),
}

/// Embeds every input of an OpenAI embeddings request.
pub async fn embed(
    state: SharedInferenceRsState,
    oairequest: EmbeddingRequest,
) -> Result<EmbeddingResponse, EmbeddingError> {
    let repr =
        serde_json::to_string(&oairequest).expect("Serialization of embedding request failed.");
    InferenceRs::maybe_log_request(state.clone(), repr);

    if let Err(e) = validate_model_name(&oairequest.model, state.clone()) {
        return validation_error(e);
    }

    if let Some(dimensions) = oairequest.dimensions {
        return validation_error(AnyhowError::new(ApiError::new(
            ApiErrorKind::InvalidRequest,
            format!("Custom embedding dimensions ({dimensions}) are not supported."),
            Some("invalid_dimensions"),
            Some("dimensions"),
        )));
    }

    let inputs = match normalize_inputs(oairequest.input) {
        Ok(inputs) => inputs,
        Err(e) => return validation_error(e),
    };

    if inputs.is_empty() {
        return validation_error(AnyhowError::new(ApiError::new(
            ApiErrorKind::InvalidRequest,
            "input must contain at least one entry.",
            Some("invalid_input"),
            Some("input"),
        )));
    }

    let model_override = if oairequest.model == "default" {
        None
    } else {
        Some(oairequest.model.clone())
    };

    let encoding = oairequest.encoding_format.unwrap_or_default();
    let return_base64 = matches!(encoding, EmbeddingEncodingFormat::Base64);

    let mut data = Vec::with_capacity(inputs.len());
    let mut total_prompt_tokens: usize = 0;
    let mut total_tokens: usize = 0;

    match inputs {
        Inputs::Prompt(prompts) => {
            let futures = prompts.into_iter().map(|prompt| {
                let state = state.clone();
                let model_override = model_override.clone();
                async move {
                    fetch_embedding(
                        state,
                        prompt,
                        model_override.as_deref(),
                        oairequest.truncate_sequence.unwrap_or(false),
                    )
                    .await
                }
            });

            let results = join_all(futures).await;
            for (index, result) in results.into_iter().enumerate() {
                match result {
                    Ok(EmbeddingWithUsage {
                        embedding,
                        prompt_tokens,
                        total_tokens: item_total_tokens,
                    }) => {
                        let embedding = if return_base64 {
                            EmbeddingVector::Base64(encode_embedding_base64(&embedding))
                        } else {
                            EmbeddingVector::Float(embedding)
                        };
                        data.push(EmbeddingData {
                            object: "embedding",
                            embedding,
                            index,
                        });
                        total_prompt_tokens = total_prompt_tokens.saturating_add(prompt_tokens);
                        total_tokens = total_tokens.saturating_add(item_total_tokens);
                    }
                    Err(e) => {
                        InferenceRs::maybe_log_error(state.clone(), e.as_ref());
                        return internal_error(e);
                    }
                }
            }
        }
        Inputs::Tokens(batches) => {
            let futures = batches.into_iter().map(|tokens| {
                let state = state.clone();
                let model_override = model_override.clone();
                async move {
                    fetch_embedding_tokens(
                        state,
                        tokens,
                        model_override.as_deref(),
                        oairequest.truncate_sequence.unwrap_or(false),
                    )
                    .await
                }
            });

            let results = join_all(futures).await;
            for (index, result) in results.into_iter().enumerate() {
                match result {
                    Ok(EmbeddingWithUsage {
                        embedding,
                        prompt_tokens,
                        total_tokens: item_total_tokens,
                    }) => {
                        let embedding = if return_base64 {
                            EmbeddingVector::Base64(encode_embedding_base64(&embedding))
                        } else {
                            EmbeddingVector::Float(embedding)
                        };
                        data.push(EmbeddingData {
                            object: "embedding",
                            embedding,
                            index,
                        });
                        total_prompt_tokens = total_prompt_tokens.saturating_add(prompt_tokens);
                        total_tokens = total_tokens.saturating_add(item_total_tokens);
                    }
                    Err(e) => {
                        InferenceRs::maybe_log_error(state.clone(), e.as_ref());
                        return internal_error(e);
                    }
                }
            }
        }
    }

    let usage = EmbeddingUsage {
        prompt_tokens: saturating_to_u32(total_prompt_tokens),
        total_tokens: saturating_to_u32(total_tokens),
    };

    let response = EmbeddingResponse {
        object: "list",
        data,
        model: response_model_id(&state, oairequest.model.clone(), &oairequest.model)
            .unwrap_or(oairequest.model),
        usage,
    };

    InferenceRs::maybe_log_response(state.clone(), &response);

    Ok(response)
}

enum Inputs {
    Prompt(Vec<String>),
    Tokens(Vec<Vec<u32>>),
}

impl Inputs {
    fn is_empty(&self) -> bool {
        match self {
            Self::Prompt(x) => x.is_empty(),
            Self::Tokens(x) => x.is_empty(),
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Prompt(x) => x.len(),
            Self::Tokens(x) => x.len(),
        }
    }
}

fn normalize_inputs(input: EmbeddingInput) -> Result<Inputs> {
    match input {
        EmbeddingInput::Single(s) => Ok(Inputs::Prompt(vec![s])),
        EmbeddingInput::Multiple(items) => Ok(Inputs::Prompt(items)),
        EmbeddingInput::Tokens(t) => Ok(Inputs::Tokens(vec![t])),
        EmbeddingInput::TokensBatch(batch) => Ok(Inputs::Tokens(batch)),
    }
}

async fn fetch_embedding(
    state: SharedInferenceRsState,
    prompt: String,
    model_id: Option<&str>,
    truncate_sequence: bool,
) -> Result<EmbeddingWithUsage> {
    let (tx, mut rx) = create_response_channel(Some(1));

    let request = Request::Normal(Box::new(NormalRequest {
        id: state.next_request_id(),
        queued_at: None,
        messages: RequestMessage::Embedding { prompt },
        sampling_params: SamplingParams::deterministic(),
        seed: None,
        response: tx,
        return_logprobs: false,
        is_streaming: false,
        suffix: None,
        constraint: Constraint::None,
        tool_choice: None,
        tools: None,
        logits_processors: None,
        return_raw_logits: false,
        web_search_options: None,
        enable_code_execution: false,
        enable_shell: false,
        shell_options: None,
        code_execution_permission: None,
        code_execution_approval_notifier: None,
        agent_permission: None,
        agent_approval_handler: None,
        agent_approval_notifier: None,
        max_tool_rounds: None,
        tool_dispatch_url: None,
        model_id: model_id.map(|m| m.to_string()),
        adapter: None,
        truncate_sequence,
        session_id: None,
        files: None,
        input_files: Vec::new(),
    }));

    send_request_with_model(&state, request, model_id)
        .await
        .context("Failed to dispatch embedding request")?;

    process_embedding_response(&mut rx, state.clone()).await
}

async fn fetch_embedding_tokens(
    state: SharedInferenceRsState,
    tokens: Vec<u32>,
    model_id: Option<&str>,
    truncate_sequence: bool,
) -> Result<EmbeddingWithUsage> {
    let (tx, mut rx) = create_response_channel(Some(1));

    let request = Request::Normal(Box::new(NormalRequest {
        id: state.next_request_id(),
        queued_at: None,
        messages: RequestMessage::EmbeddingTokens { prompt: tokens },
        sampling_params: SamplingParams::deterministic(),
        seed: None,
        response: tx,
        return_logprobs: false,
        is_streaming: false,
        suffix: None,
        constraint: Constraint::None,
        tool_choice: None,
        tools: None,
        logits_processors: None,
        return_raw_logits: false,
        web_search_options: None,
        enable_code_execution: false,
        enable_shell: false,
        shell_options: None,
        code_execution_permission: None,
        code_execution_approval_notifier: None,
        agent_permission: None,
        agent_approval_handler: None,
        agent_approval_notifier: None,
        max_tool_rounds: None,
        tool_dispatch_url: None,
        model_id: model_id.map(|m| m.to_string()),
        adapter: None,
        truncate_sequence,
        session_id: None,
        files: None,
        input_files: Vec::new(),
    }));

    send_request_with_model(&state, request, model_id)
        .await
        .context("Failed to dispatch embedding request")?;

    process_embedding_response(&mut rx, state.clone()).await
}

async fn process_embedding_response(
    rx: &mut Receiver<Response>,
    state: SharedInferenceRsState,
) -> Result<EmbeddingWithUsage> {
    base_process_non_streaming_response(
        rx,
        state.clone(),
        |_, response| match response {
            Response::Embeddings {
                embeddings,
                prompt_tokens,
                total_tokens,
            } => Ok(EmbeddingWithUsage {
                embedding: embeddings,
                prompt_tokens,
                total_tokens,
            }),
            Response::ValidationError(e) => Err(AnyhowError::new(ApiError::from_error(
                e.as_ref(),
                ApiErrorKind::InvalidRequest,
            ))),
            Response::InternalError(e) => {
                InferenceRs::maybe_log_error(state.clone(), e.as_ref());
                Err(anyhow!(e))
            }
            Response::ModelError(msg, _) => {
                InferenceRs::maybe_log_error(state.clone(), &ModelErrorMessage(msg));
                Err(AnyhowError::new(ApiError::model_error()))
            }
            Response::Done(_)
            | Response::Chunk(_)
            | Response::CompletionDone(_)
            | Response::CompletionChunk(_)
            | Response::CompletionModelError(_, _)
            | Response::ImageGeneration(_)
            | Response::Speech { .. }
            | Response::Raw { .. }
            | Response::AgenticToolCallProgress { .. }
            | Response::BlockDenoisingProgress(_)
            | Response::AgenticToolApprovalRequired { .. }
            | Response::File(_) => Err(anyhow!(
                "Received unexpected response type from embedding request."
            )),
        },
        |_, err| Err(anyhow!(err)),
    )
    .await
}

fn validation_error<E>(err: E) -> Result<EmbeddingResponse, EmbeddingError>
where
    E: Into<AnyhowError>,
{
    Err(EmbeddingError::Validation(err.into()))
}

fn internal_error<E>(err: E) -> Result<EmbeddingResponse, EmbeddingError>
where
    E: Into<AnyhowError>,
{
    Err(EmbeddingError::Internal(err.into()))
}

fn encode_embedding_base64(embedding: &[f32]) -> String {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(embedding));
    for value in embedding {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    BASE64_STANDARD.encode(bytes)
}

fn saturating_to_u32(value: usize) -> u32 {
    if value > u32::MAX as usize {
        u32::MAX
    } else {
        value as u32
    }
}
