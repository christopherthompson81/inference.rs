//! OpenAI completions as an engine operation, free of HTTP.

use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

use anyhow::Result;
use futures::future::BoxFuture;
use inference_core::{
    CompletionChunkResponse, Constraint, InferenceRs, NormalRequest, Request, RequestCancellation,
    RequestMessage, Response, SamplingParams,
};
use tokio::sync::mpsc::{Receiver, Sender};

use crate::{
    api_error::{ApiError, ApiErrorKind, ModelErrorMessage, boxed_anyhow},
    dispatch::{
        apply_model_override, create_response_channel, response_model_id, send_request_with_model,
    },
    engine_chat::{DispatchError, ResponseTap},
    logits_processors::LogitsProcessors,
    lora_routing::{DEFAULT_MODEL_ID, resolve_lora_adapter_model},
    openai::{CompletionPrompt, CompletionRequest, Grammar},
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
    pub cancellation: RequestCancellation,
}

/// Resolves the request's model (LoRA aliases included), parses it and sends it to its model.
pub(crate) fn prepare_completion<'a>(
    state: &'a SharedInferenceRsState,
    logits_processors: &'a LogitsProcessors,
    oairequest: CompletionRequest,
) -> BoxFuture<'a, Result<PreparedCompletion, DispatchError>> {
    Box::pin(prepare_completion_inner(
        state,
        logits_processors,
        oairequest,
    ))
}

async fn prepare_completion_inner(
    state: &SharedInferenceRsState,
    logits_processors: &LogitsProcessors,
    mut oairequest: CompletionRequest,
) -> Result<PreparedCompletion, DispatchError> {
    let (tx, rx) = create_response_channel(None);
    let requested_model = oairequest.model.clone();
    resolve_lora_adapter_model(state, &mut oairequest.model, &mut oairequest.adapter)
        .map_err(|error| DispatchError::Validation(Box::new(error)))?;
    let model_override = response_model_id(state, requested_model, &oairequest.model);
    let logits_processors = logits_processors
        .resolve(oairequest.logits_processors.as_deref())
        .map_err(|error| DispatchError::Validation(Box::new(error)))?;
    let model_id = (oairequest.model != DEFAULT_MODEL_ID).then(|| oairequest.model.clone());
    let (mut request, is_streaming) = parse_request(oairequest, state.clone(), tx)
        .map_err(|error| DispatchError::Validation(boxed_anyhow(error)))?;
    let cancellation = RequestCancellation::default();
    if let Request::Normal(normal) = &mut request {
        normal.cancellation = Some(cancellation.clone());
        normal.logits_processors = logits_processors;
    }
    send_request_with_model(state, request, model_id.as_deref())
        .await
        .map_err(|error| DispatchError::Internal(error.into()))?;
    Ok(PreparedCompletion {
        rx,
        is_streaming,
        model_override,
        cancellation,
    })
}

/// Waits for a non-streaming completion's final response; errors come back as the core's error responses.
pub fn collect_completion<'a>(
    rx: &'a mut Receiver<Response>,
    model_override: Option<&'a str>,
) -> BoxFuture<'a, Response> {
    Box::pin(collect_completion_inner(rx, model_override))
}

async fn collect_completion_inner(
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
                );
            }
        }
    }
}

/// Parses and validates a completion request.
///
/// This function transforms an OpenAI-compatible completion request into the
/// request format used by inference.rs.
pub(crate) fn parse_request(
    oairequest: CompletionRequest,
    state: Arc<InferenceRs>,
    tx: Sender<Response>,
) -> Result<(Request, bool)> {
    let repr = serde_json::to_string(&oairequest).expect("Serialization of request failed.");
    InferenceRs::maybe_log_request(state.clone(), repr);

    // Validate that the requested model matches the loaded model
    validate_model_name(&oairequest.model, state.clone())?;
    let adapter = oairequest
        .adapter
        .clone()
        .map(crate::lora_routing::core_adapter_selection)
        .transpose()?;

    if oairequest.max_tokens == Some(0) {
        anyhow::bail!("max_tokens must be at least 1.");
    }
    if matches!(oairequest.prompt, CompletionPrompt::Tokens(_))
        && (oairequest.echo_prompt || oairequest.best_of.is_some_and(|n| n > 1))
    {
        anyhow::bail!("echo and best_of need a text prompt, not token ids.");
    }

    let stop_toks = convert_stop_tokens(oairequest.stop_seqs, oairequest.stop_token_ids);

    let is_streaming = oairequest.stream.unwrap_or(false);

    let dry_params = get_dry_sampling_params(
        oairequest.dry_multiplier,
        oairequest.dry_sequence_breakers,
        oairequest.dry_base,
        oairequest.dry_allowed_length,
    )?;

    let messages = match oairequest.prompt {
        CompletionPrompt::Text(text) => RequestMessage::Completion {
            text,
            echo_prompt: oairequest.echo_prompt,
            best_of: oairequest.best_of,
        },
        CompletionPrompt::Tokens(tokens) => RequestMessage::CompletionTokens(tokens),
    };
    let sampling_params = SamplingParams {
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
    };
    Ok((
        Request::Normal(Box::new(NormalRequest {
            seed: oairequest.seed,
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
            model_id: if oairequest.model == DEFAULT_MODEL_ID {
                None
            } else {
                Some(oairequest.model.clone())
            },
            adapter,
            truncate_sequence: oairequest.truncate_sequence.unwrap_or(false),
            ..NormalRequest::new_simple(
                messages,
                sampling_params,
                tx,
                state.next_request_id(),
                oairequest.tools,
                oairequest.tool_choice,
            )
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
    cancellation: Option<RequestCancellation>,
}

impl CompletionStream {
    pub(crate) fn new(
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
            cancellation: None,
        }
    }

    /// Reports each engine response to `tap` as the stream reads it, e.g. for an access log.
    pub fn with_tap(mut self, tap: Option<ResponseTap>) -> Self {
        self.tap = tap;
        self
    }

    /// The request's cancellation, for a caller that cancels from elsewhere, e.g. a signal handler.
    pub fn cancellation(&self) -> Option<RequestCancellation> {
        self.cancellation.clone()
    }

    pub fn with_cancellation(mut self, cancellation: RequestCancellation) -> Self {
        self.cancellation = Some(cancellation);
        self
    }

    /// Ends the request on its next sampled token; the stream still yields its final event, with usage.
    pub fn cancel(&self) {
        if let Some(cancellation) = &self.cancellation {
            cancellation.cancel();
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
            | Response::Transcription(_)
            | Response::VoiceActivity(_)
            | Response::Diarization(_)
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
