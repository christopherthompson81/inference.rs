//! OpenAI completions as an engine operation, free of HTTP.

use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

use anyhow::Result;
use inference_core::{
    CompletionChunkResponse, Constraint, InferenceRs, NormalRequest, Request, RequestMessage,
    Response, SamplingParams,
};
use tokio::sync::mpsc::{Receiver, Sender};

use crate::{
    api_error::{boxed_anyhow, ApiError, ApiErrorKind, ModelErrorMessage},
    dispatch::{
        apply_model_override, create_response_channel, request_model_override,
        send_request_with_model,
    },
    engine_chat::{DispatchError, ResponseTap},
    lora_routing::{resolve_lora_adapter_model, DEFAULT_MODEL_ID},
    openai::{CompletionRequest, Grammar},
    sampling::{convert_stop_tokens, get_dry_sampling_params},
    types::SharedInferenceRsState,
    util::validate_model_name,
};

/// A dispatched completion request.
pub struct PreparedCompletion {
    pub rx: Receiver<Response>,
    pub is_streaming: bool,
    /// The model name the caller asked for, when routing resolved it to another id.
    pub model_override: Option<String>,
}

/// Resolves the request's model (LoRA aliases included), parses it and sends it to its model.
pub async fn prepare_completion(
    state: &SharedInferenceRsState,
    mut oairequest: CompletionRequest,
) -> Result<PreparedCompletion, DispatchError> {
    let (tx, rx) = create_response_channel(None);
    let requested_model = oairequest.model.clone();
    resolve_lora_adapter_model(state, &mut oairequest.model, &mut oairequest.adapter)
        .map_err(|error| DispatchError::Validation(Box::new(error)))?;
    let model_override = request_model_override(requested_model, &oairequest.model);
    let model_id = (oairequest.model != DEFAULT_MODEL_ID).then(|| oairequest.model.clone());
    let (request, is_streaming) = parse_request(oairequest, state.clone(), tx)
        .map_err(|error| DispatchError::Validation(boxed_anyhow(error)))?;
    send_request_with_model(state, request, model_id.as_deref())
        .await
        .map_err(|error| DispatchError::Internal(error.into()))?;
    Ok(PreparedCompletion {
        rx,
        is_streaming,
        model_override,
    })
}

/// Waits for a non-streaming completion's final response; errors come back as the core's error responses.
pub async fn collect_completion(
    rx: &mut Receiver<Response>,
    model_override: Option<&str>,
) -> Response {
    loop {
        match rx.recv().await {
            Some(
                Response::AgenticToolCallProgress { .. }
                | Response::BlockDenoisingProgress(_)
                | Response::AgenticToolApprovalRequired { .. }
                | Response::File(_),
            ) => continue,
            Some(mut response) => {
                if let Response::CompletionDone(response)
                | Response::CompletionModelError(_, response) = &mut response
                {
                    apply_model_override(&mut response.model, model_override);
                }
                return response;
            }
            None => {
                return Response::InternalError(
                    anyhow::Error::msg("No response received from the model.").into(),
                )
            }
        }
    }
}

/// Parses and validates a completion request.
///
/// This function transforms an OpenAI-compatible completion request into the
/// request format used by inference.rs.
pub fn parse_request(
    oairequest: CompletionRequest,
    state: Arc<InferenceRs>,
    tx: Sender<Response>,
) -> Result<(Request, bool)> {
    let repr = serde_json::to_string(&oairequest).expect("Serialization of request failed.");
    InferenceRs::maybe_log_request(state.clone(), repr);

    // Validate that the requested model matches the loaded model
    validate_model_name(&oairequest.model, state.clone())?;

    if oairequest.max_tokens == Some(0) {
        anyhow::bail!("max_tokens must be at least 1.");
    }

    let stop_toks = convert_stop_tokens(oairequest.stop_seqs);

    let is_streaming = oairequest.stream.unwrap_or(false);

    let dry_params = get_dry_sampling_params(
        oairequest.dry_multiplier,
        oairequest.dry_sequence_breakers,
        oairequest.dry_base,
        oairequest.dry_allowed_length,
    )?;

    Ok((
        Request::Normal(Box::new(NormalRequest {
            id: state.next_request_id(),
            queued_at: None,
            messages: RequestMessage::Completion {
                text: oairequest.prompt,
                echo_prompt: oairequest.echo_prompt,
                best_of: oairequest.best_of,
            },
            sampling_params: SamplingParams {
                temperature: oairequest.temperature,
                top_k: oairequest.top_k,
                top_p: oairequest.top_p,
                min_p: oairequest.min_p,
                top_n_logprobs: oairequest.logprobs.unwrap_or(1),
                frequency_penalty: oairequest.frequency_penalty,
                presence_penalty: oairequest.presence_penalty,
                repetition_penalty: oairequest.repetition_penalty,
                max_len: oairequest.max_tokens,
                stop_toks,
                ignore_eos: oairequest.ignore_eos,
                logits_bias: oairequest.logit_bias,
                n_choices: oairequest.n_choices,
                dry_params,
            },
            seed: oairequest.seed,
            response: tx,
            return_logprobs: oairequest.logprobs.is_some(),
            is_streaming,
            suffix: oairequest.suffix,
            constraint: match oairequest.grammar {
                Some(Grammar::Regex(regex)) => Constraint::Regex(regex),
                Some(Grammar::Lark(lark)) => Constraint::Lark(lark),
                Some(Grammar::JsonSchema(schema)) => Constraint::JsonSchema(schema),
                Some(Grammar::Llguidance(llguidance)) => Constraint::Llguidance(llguidance),
                None => Constraint::None,
            },
            tool_choice: oairequest.tool_choice,
            tools: oairequest.tools,
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
            model_id: if oairequest.model == DEFAULT_MODEL_ID {
                None
            } else {
                Some(oairequest.model.clone())
            },
            adapter: oairequest.adapter.map(Into::into),
            truncate_sequence: oairequest.truncate_sequence.unwrap_or(false),
            session_id: None,
            files: None,
            input_files: Vec::new(),
        })),
        is_streaming,
    ))
}

/// One event of a streaming completion.
pub enum CompletionStreamEvent {
    Chunk(CompletionChunkResponse),
    /// Terminal: nothing follows an error.
    Error(ApiError),
}

impl CompletionStreamEvent {
    /// `{"event": "chunk" | "error", "data": <payload>}`; an error's payload is the OpenAI error envelope.
    pub fn to_json(&self) -> String {
        let (event, data) = match self {
            Self::Chunk(chunk) => (
                "chunk",
                serde_json::to_value(chunk)
                    .unwrap_or_else(|_| ApiError::internal().to_openai_body()),
            ),
            Self::Error(error) => ("error", error.to_openai_body()),
        };
        serde_json::json!({ "event": event, "data": data }).to_string()
    }
}

/// The events of a streaming completion. Dropping it abandons the request.
pub struct CompletionStream {
    rx: Receiver<Response>,
    state: SharedInferenceRsState,
    model_override: Option<String>,
    tap: Option<ResponseTap>,
    finished: bool,
}

impl CompletionStream {
    pub fn new(
        rx: Receiver<Response>,
        state: SharedInferenceRsState,
        model_override: Option<String>,
        tap: Option<ResponseTap>,
    ) -> Self {
        Self {
            rx,
            state,
            model_override,
            tap,
            finished: false,
        }
    }

    fn map(&mut self, response: Response) -> Option<CompletionStreamEvent> {
        Some(match response {
            Response::CompletionModelError(msg, _) => {
                InferenceRs::maybe_log_error(self.state.clone(), &ModelErrorMessage(msg));
                self.finished = true;
                CompletionStreamEvent::Error(ApiError::model_error())
            }
            Response::ValidationError(e) => {
                self.finished = true;
                CompletionStreamEvent::Error(ApiError::from_error(
                    e.as_ref(),
                    ApiErrorKind::InvalidRequest,
                ))
            }
            Response::InternalError(e) => {
                InferenceRs::maybe_log_error(self.state.clone(), &*e);
                self.finished = true;
                CompletionStreamEvent::Error(ApiError::from_error(
                    e.as_ref(),
                    ApiErrorKind::Internal,
                ))
            }
            Response::CompletionChunk(mut response) => {
                if response.choices.iter().all(|x| x.finish_reason.is_some()) {
                    self.finished = true;
                }
                InferenceRs::maybe_log_response(self.state.clone(), &response);
                apply_model_override(&mut response.model, self.model_override.as_deref());
                CompletionStreamEvent::Chunk(response)
            }
            Response::AgenticToolCallProgress { .. }
            | Response::BlockDenoisingProgress(_)
            | Response::AgenticToolApprovalRequired { .. }
            | Response::File(_) => return None,
            Response::Done(_)
            | Response::CompletionDone(_)
            | Response::Chunk(_)
            | Response::ImageGeneration(_)
            | Response::ModelError(_, _)
            | Response::Speech { .. }
            | Response::Raw { .. }
            | Response::Embeddings { .. } => unreachable!("not a completion stream response"),
        })
    }

    /// The next event, or `None` once the stream has finished.
    pub async fn next_event(&mut self) -> Option<CompletionStreamEvent> {
        futures::StreamExt::next(self).await
    }
}

impl futures::Stream for CompletionStream {
    type Item = CompletionStreamEvent;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            if self.finished {
                return Poll::Ready(None);
            }
            match self.rx.poll_recv(cx) {
                Poll::Ready(Some(response)) => {
                    if let Some(tap) = &self.tap {
                        tap(&response);
                    }
                    if let Some(event) = self.map(response) {
                        return Poll::Ready(Some(event));
                    }
                }
                Poll::Ready(None) => {
                    self.finished = true;
                    return Poll::Ready(Some(CompletionStreamEvent::Error(ApiError::internal())));
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}
