//! ## Completions functionality and route handler.

use std::{
    pin::Pin,
    task::{Context, Poll},
    time::Duration,
};

use axum::{
    Extension,
    extract::{Json, State, rejection::JsonRejection},
    response::{
        IntoResponse, Sse,
        sse::{Event, KeepAlive, KeepAliveStream},
    },
};
use inference_core::{CompletionChunkResponse, CompletionResponse, InferenceRs, Response};
use tokio::sync::mpsc::Receiver;

pub use crate::engine_completion::parse_request;
use crate::{
    completion_core::{
        BaseCompletionResponder, handle_completion_error, handle_completion_validation_error,
    },
    engine_chat::DispatchError,
    engine_completion::{
        CompletionStream, CompletionStreamEvent, collect_completion, prepare_completion,
    },
    handler_core::{
        ApiError, ApiErrorHttp, ApiErrorKind, ModelErrorMessage, openai_error_from_error,
        openai_error_response,
    },
    openai::{CompletionChunkResponseBody, CompletionRequest, CompletionResponseBody},
    streaming::{DoneState, StreamOutcomeHandle, get_keep_alive_interval, openai_error_event},
    types::{ExtractedInferenceRsState, OnChunkCallback, OnDoneCallback, SharedInferenceRsState},
};

/// A callback function that processes streaming response chunks before they are sent to the client.
///
/// This hook allows modification of each chunk in the streaming response, enabling features like
/// content filtering, transformation, or logging. The callback receives a chunk and must return
/// a (potentially modified) chunk.
///
/// ### Examples
///
/// ```no_run
/// use inference_server_core::completions::CompletionOnChunkCallback;
///
/// let on_chunk: CompletionOnChunkCallback = Box::new(|mut chunk| {
///     // Log the chunk or modify its content
///     println!("Processing chunk: {:?}", chunk);
///     chunk
/// });
/// ```
pub type CompletionOnChunkCallback = OnChunkCallback<CompletionChunkResponse>;

/// A callback function that is executed when the streaming response completes.
///
/// This hook receives all chunks that were streamed during the response, allowing for
/// post-processing, analytics, or cleanup operations after the stream finishes.
///
/// ### Examples
///
/// ```no_run
/// use inference_server_core::completions::CompletionOnDoneCallback;
///
/// let on_done: CompletionOnDoneCallback = Box::new(|chunks| {
///     println!("Stream completed with {} chunks", chunks.len());
///     // Process all chunks for analytics
/// });
/// ```
pub type CompletionOnDoneCallback = OnDoneCallback<CompletionChunkResponse>;

/// Frames the engine's completion stream as Server-Sent Events, ending with `[DONE]`.
pub struct CompletionStreamer {
    inner: CompletionStream,
    done_state: DoneState,
    on_chunk: Option<CompletionOnChunkCallback>,
    on_done: Option<CompletionOnDoneCallback>,
    chunks: Vec<CompletionChunkResponse>,
}

impl futures::Stream for CompletionStreamer {
    type Item = Result<Event, axum::Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match self.done_state {
            DoneState::SendingDone => {
                self.done_state = DoneState::Done;
                return Poll::Ready(Some(Ok(Event::default().data("[DONE]"))));
            }
            DoneState::Done => {
                if let Some(on_done) = &self.on_done {
                    on_done(&self.chunks);
                }
                return Poll::Ready(None);
            }
            DoneState::Running => (),
        }
        match Pin::new(&mut self.inner).poll_next(cx) {
            Poll::Ready(Some(CompletionStreamEvent::Chunk(mut response))) => {
                if let Some(on_chunk) = &self.on_chunk {
                    response = on_chunk(response);
                }
                if self.on_done.is_some() {
                    self.chunks.push(response.clone());
                }
                Poll::Ready(Some(Event::default().json_data(response)))
            }
            Poll::Ready(Some(CompletionStreamEvent::Error(error))) => {
                Poll::Ready(Some(Ok(openai_error_event(error))))
            }
            // https://platform.openai.com/docs/api-reference/completions/create: the stream ends with `data: [DONE]`
            Poll::Ready(None) => {
                self.done_state = DoneState::Done;
                Poll::Ready(Some(Ok(Event::default().data("[DONE]"))))
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

pub type CompletionResponder =
    BaseCompletionResponder<CompletionResponse, KeepAliveStream<CompletionStreamer>>;

impl IntoResponse for CompletionResponder {
    /// Converts the completion responder into an HTTP response.
    fn into_response(self) -> axum::response::Response {
        match self {
            CompletionResponder::Sse(s) => s.into_response(),
            CompletionResponder::Json(s) => Json(s).into_response(),
            CompletionResponder::InternalError(e) => {
                openai_error_from_error(e.as_ref(), ApiErrorKind::Internal)
            }
            CompletionResponder::ValidationError(e) => {
                openai_error_from_error(e.as_ref(), ApiErrorKind::InvalidRequest)
            }
            CompletionResponder::ModelError(_, _) => openai_error_response(ApiError::model_error()),
        }
    }
}

/// OpenAI-compatible completions endpoint handler.
#[utoipa::path(
    post,
    tag = "inference.rs",
    path = "/v1/completions",
    request_body = CompletionRequest,
    responses((
        status = 200,
        description = "Completion JSON or server-sent event chunks",
        content(
            (CompletionResponseBody = "application/json"),
            (CompletionChunkResponseBody = "text/event-stream")
        )
    ))
)]
pub async fn completions(
    State(state): ExtractedInferenceRsState,
    stream_outcome: Option<Extension<StreamOutcomeHandle>>,
    payload: Result<Json<CompletionRequest>, JsonRejection>,
) -> CompletionResponder {
    let oairequest = match payload {
        Ok(Json(request)) => request,
        Err(error) => {
            return CompletionResponder::ValidationError(Box::new(ApiError::from_json_rejection(
                error,
            )));
        }
    };
    let prepared = match prepare_completion(&state, oairequest).await {
        Ok(prepared) => prepared,
        Err(DispatchError::Validation(e)) => return handle_completion_validation_error(state, e),
        Err(DispatchError::Internal(e)) => return handle_error(state, e),
    };
    if prepared.is_streaming {
        let tap = stream_outcome.map(|Extension(handle)| handle.tap());
        let stream = CompletionStream::new(prepared.rx, state, prepared.model_override, tap);
        CompletionResponder::Sse(sse(stream, None, None))
    } else {
        let mut rx = prepared.rx;
        let response = collect_completion(&mut rx, prepared.model_override.as_deref()).await;
        match_responses(state, response)
    }
}

/// Handle route / generation errors and logging them.
pub fn handle_error(
    state: SharedInferenceRsState,
    e: Box<dyn std::error::Error + Send + Sync + 'static>,
) -> CompletionResponder {
    handle_completion_error(state, e)
}

/// Creates a SSE streamer for chat completions with optional callbacks.
pub fn create_streamer(
    rx: Receiver<Response>,
    state: SharedInferenceRsState,
    on_chunk: Option<CompletionOnChunkCallback>,
    on_done: Option<CompletionOnDoneCallback>,
) -> Sse<KeepAliveStream<CompletionStreamer>> {
    create_streamer_with_outcome(rx, state, on_chunk, on_done, None)
}

/// Like [`create_streamer`], also reporting usage and errors to the access log at stream end.
pub fn create_streamer_with_outcome(
    rx: Receiver<Response>,
    state: SharedInferenceRsState,
    on_chunk: Option<CompletionOnChunkCallback>,
    on_done: Option<CompletionOnDoneCallback>,
    outcome: Option<StreamOutcomeHandle>,
) -> Sse<KeepAliveStream<CompletionStreamer>> {
    let tap = outcome.map(StreamOutcomeHandle::tap);
    sse(
        CompletionStream::new(rx, state, None, tap),
        on_chunk,
        on_done,
    )
}

fn sse(
    inner: CompletionStream,
    on_chunk: Option<CompletionOnChunkCallback>,
    on_done: Option<CompletionOnDoneCallback>,
) -> Sse<KeepAliveStream<CompletionStreamer>> {
    Sse::new(CompletionStreamer {
        inner,
        done_state: DoneState::Running,
        on_chunk,
        on_done,
        chunks: Vec::new(),
    })
    .keep_alive(KeepAlive::new().interval(Duration::from_millis(get_keep_alive_interval())))
}

/// Process non-streaming completion responses.
pub async fn process_non_streaming_response(
    rx: &mut Receiver<Response>,
    state: SharedInferenceRsState,
) -> CompletionResponder {
    match_responses(state, collect_completion(rx, None).await)
}

/// Matches and processes different types of model responses into appropriate completion responses.
pub fn match_responses(state: SharedInferenceRsState, response: Response) -> CompletionResponder {
    match response {
        Response::InternalError(e) => {
            InferenceRs::maybe_log_error(state, &*e);
            CompletionResponder::InternalError(e)
        }
        Response::CompletionModelError(msg, response) => {
            InferenceRs::maybe_log_error(state.clone(), &ModelErrorMessage(msg.to_string()));
            InferenceRs::maybe_log_response(state, &response);
            CompletionResponder::ModelError(msg, response)
        }
        Response::ValidationError(e) => CompletionResponder::ValidationError(e),
        Response::CompletionDone(response) => {
            InferenceRs::maybe_log_response(state, &response);
            CompletionResponder::Json(response)
        }
        Response::CompletionChunk(_) => unreachable!(),
        Response::Chunk(_) => unreachable!(),
        Response::Done(_) => unreachable!(),
        Response::ModelError(_, _) => unreachable!(),
        Response::ImageGeneration(_) => unreachable!(),
        Response::Speech { .. } => unreachable!(),
        Response::Raw { .. } => unreachable!(),
        Response::Embeddings { .. } => unreachable!(),
        Response::AgenticToolCallProgress { .. } => unreachable!(),
        Response::BlockDenoisingProgress(_) => unreachable!(),
        Response::AgenticToolApprovalRequired { .. } => unreachable!(),
        Response::File(_) => unreachable!(),
    }
}
