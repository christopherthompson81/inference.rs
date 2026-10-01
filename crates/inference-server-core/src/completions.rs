//! ## Completions functionality and route handler.

use std::{
    pin::Pin,
    task::{Context, Poll},
    time::Duration,
};

use axum::{
    Extension,
    extract::Json,
    response::{
        IntoResponse, Sse,
        sse::{Event, KeepAlive, KeepAliveStream},
    },
};
use inference_core::{CompletionChunkResponse, CompletionResponse};

use crate::handler_core::{ApiJson, ApiJsonRejection};
#[cfg(test)]
use crate::openai::{CompletionChunkResponseBody, CompletionResponseBody};
use crate::{
    completion_core::BaseCompletionResponder,
    engine_completion::{CompletionStream, CompletionStreamEvent},
    handler_core::openai_error_response,
    openai::CompletionRequest,
    streaming::{DoneState, StreamOutcomeHandle, get_keep_alive_interval, openai_error_event},
    types::{OnChunkCallback, OnDoneCallback, OwnedEngine},
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
            CompletionResponder::Error(error) => openai_error_response(error),
        }
    }
}

/// OpenAI-compatible completions endpoint handler.
#[cfg_attr(test, utoipa::path(
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
))]
pub async fn completions(
    OwnedEngine(engine): OwnedEngine,
    stream_outcome: Option<Extension<StreamOutcomeHandle>>,
    payload: Result<ApiJson<CompletionRequest>, ApiJsonRejection>,
) -> CompletionResponder {
    let request = match payload {
        Ok(ApiJson(request)) => request,
        Err(ApiJsonRejection(error)) => return CompletionResponder::Error(error),
    };
    if request.stream.unwrap_or(false) {
        let tap = stream_outcome.map(|Extension(handle)| handle.tap());
        match engine.completion_stream(request).await {
            Ok(stream) => {
                CompletionResponder::Sse(create_streamer(stream.with_tap(tap), None, None))
            }
            Err(error) => CompletionResponder::Error(error),
        }
    } else {
        match engine.completion(request).await {
            Ok(response) => CompletionResponder::Json(response),
            Err(error) => CompletionResponder::Error(error),
        }
    }
}

/// Frames `stream` (from [`inference_api::Engine::completion_stream`]) as SSE, with optional per-chunk and end hooks.
pub fn create_streamer(
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
