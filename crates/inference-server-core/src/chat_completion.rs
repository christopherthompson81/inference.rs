//! ## Chat Completions functionality and route handler.

use std::{pin::Pin, task::Poll, time::Duration};

use axum::{
    Extension,
    extract::Json,
    response::{
        IntoResponse, Sse,
        sse::{Event, KeepAlive, KeepAliveStream},
    },
};
use inference_api::response::{ChatCompletionChunkResponse, ChatCompletionResponse};

use crate::handler_core::{ApiJson, ApiJsonRejection};
#[cfg(test)]
use crate::openai::{ChatCompletionChunkResponseBody, ChatCompletionResponseBody};
use crate::{
    completion_core::BaseCompletionResponder,
    engine_chat::{ChatStream, ChatStreamEvent},
    handler_core::openai_error_response,
    openai::ChatCompletionRequest,
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
/// use inference_server_core::chat_completion::ChatCompletionOnChunkCallback;
///
/// let on_chunk: ChatCompletionOnChunkCallback = Box::new(|mut chunk| {
///     // Log the chunk or modify its content
///     println!("Processing chunk: {:?}", chunk);
///     chunk
/// });
/// ```
pub type ChatCompletionOnChunkCallback = OnChunkCallback<ChatCompletionChunkResponse>;

/// A callback function that is executed when the streaming response completes.
///
/// This hook receives all chunks that were streamed during the response, allowing for
/// post-processing, analytics, or cleanup operations after the stream finishes.
///
/// ### Examples
///
/// ```no_run
/// use inference_server_core::chat_completion::ChatCompletionOnDoneCallback;
///
/// let on_done: ChatCompletionOnDoneCallback = Box::new(|chunks| {
///     println!("Stream completed with {} chunks", chunks.len());
///     // Process all chunks for analytics
/// });
/// ```
pub type ChatCompletionOnDoneCallback = OnDoneCallback<ChatCompletionChunkResponse>;

/// A streaming response handler.
///
/// It frames the engine's chat stream events as Server-Sent Events, ending with `[DONE]`.
pub struct ChatCompletionStreamer {
    inner: ChatStream,
    done_state: DoneState,
    on_chunk: Option<ChatCompletionOnChunkCallback>,
    on_done: Option<ChatCompletionOnDoneCallback>,
    chunks: Vec<ChatCompletionChunkResponse>,
}

impl ChatCompletionStreamer {
    fn new(
        inner: ChatStream,
        on_chunk: Option<ChatCompletionOnChunkCallback>,
        on_done: Option<ChatCompletionOnDoneCallback>,
    ) -> Self {
        Self {
            inner,
            done_state: DoneState::Running,
            on_chunk,
            on_done,
            chunks: Vec::new(),
        }
    }
}

impl futures::Stream for ChatCompletionStreamer {
    type Item = Result<Event, axum::Error>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        match self.done_state {
            DoneState::SendingDone => {
                // https://platform.openai.com/docs/api-reference/completions/create
                // If true, returns a stream of events that happen during the Run as server-sent events, terminating when the Run enters a terminal state with a data: [DONE] message.
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
            // Only streams that opt in carry these, and HTTP has no event for them.
            Poll::Ready(Some(ChatStreamEvent::BlockDenoisingProgress(_))) => {
                cx.waker().wake_by_ref();
                Poll::Pending
            }
            Poll::Ready(Some(event)) => {
                let name = sse_event_name(&event);
                let sse = match event {
                    ChatStreamEvent::Chunk(mut response) => {
                        if let Some(on_chunk) = &self.on_chunk {
                            response = on_chunk(response);
                        }
                        if self.on_done.is_some() {
                            self.chunks.push(response.clone());
                        }
                        Event::default().json_data(response)
                    }
                    ChatStreamEvent::AgenticToolCallProgress(progress) => {
                        Event::default().json_data(progress.to_json())
                    }
                    ChatStreamEvent::AgenticToolApprovalRequired(approval) => {
                        Event::default().json_data(approval.to_json())
                    }
                    ChatStreamEvent::FileProduced(file) => Event::default().json_data(file),
                    ChatStreamEvent::BlockDenoisingProgress(_) => {
                        unreachable!(
                            "the HTTP route doesn't ask its ChatStream for denoising progress"
                        )
                    }
                    ChatStreamEvent::Error(error) => Ok(openai_error_event(error)),
                };
                Poll::Ready(Some(match name {
                    Some(name) => sse.map(|event| event.event(name)),
                    None => sse,
                }))
            }
            Poll::Ready(None) => {
                self.done_state = DoneState::Done;
                Poll::Ready(Some(Ok(Event::default().data("[DONE]"))))
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

/// Represents different types of chat completion responses.
pub type ChatCompletionResponder =
    BaseCompletionResponder<ChatCompletionResponse, KeepAliveStream<ChatCompletionStreamer>>;

impl IntoResponse for ChatCompletionResponder {
    /// Converts the chat completion responder into an HTTP response.
    fn into_response(self) -> axum::response::Response {
        match self {
            ChatCompletionResponder::Sse(s) => s.into_response(),
            ChatCompletionResponder::Json(s) => Json(s).into_response(),
            ChatCompletionResponder::Error(error) => openai_error_response(error),
        }
    }
}

/// OpenAI-compatible chat completions endpoint handler.
#[cfg_attr(test, utoipa::path(
    post,
    tag = "inference.rs",
    path = "/v1/chat/completions",
    request_body = ChatCompletionRequest,
    responses((
        status = 200,
        description = "Chat completion JSON or server-sent event chunks",
        content(
            (ChatCompletionResponseBody = "application/json"),
            (ChatCompletionChunkResponseBody = "text/event-stream")
        )
    ))
))]
pub async fn chatcompletions(
    OwnedEngine(engine): OwnedEngine,
    stream_outcome: Option<Extension<StreamOutcomeHandle>>,
    payload: Result<ApiJson<ChatCompletionRequest>, ApiJsonRejection>,
) -> ChatCompletionResponder {
    let request = match payload {
        Ok(ApiJson(request)) => request,
        Err(ApiJsonRejection(error)) => return ChatCompletionResponder::Error(error),
    };
    if request.stream.unwrap_or(false) {
        let tap = stream_outcome.map(|Extension(handle)| handle.tap());
        match engine.chat_stream(request, Default::default()).await {
            Ok(stream) => {
                ChatCompletionResponder::Sse(create_streamer(stream.with_tap(tap), None, None))
            }
            Err(error) => ChatCompletionResponder::Error(error),
        }
    } else {
        match engine.chat(request, Default::default()).await {
            Ok(response) => ChatCompletionResponder::Json(response),
            Err(error) => ChatCompletionResponder::Error(error),
        }
    }
}

/// Frames `stream` (from [`inference_api::Engine::chat_stream`]) as SSE, with optional per-chunk and end hooks.
pub fn create_streamer(
    stream: ChatStream,
    on_chunk: Option<ChatCompletionOnChunkCallback>,
    on_done: Option<ChatCompletionOnDoneCallback>,
) -> Sse<KeepAliveStream<ChatCompletionStreamer>> {
    sse(ChatCompletionStreamer::new(stream, on_chunk, on_done))
}

// Chunks and errors go out as plain `data:` lines; the other events are named.
fn sse_event_name(event: &ChatStreamEvent) -> Option<&'static str> {
    match event {
        ChatStreamEvent::AgenticToolCallProgress(_) => Some("agentic_tool_call_progress"),
        ChatStreamEvent::AgenticToolApprovalRequired(_) => Some("agentic_tool_approval_required"),
        ChatStreamEvent::FileProduced(_) => Some("file_produced"),
        ChatStreamEvent::BlockDenoisingProgress(_) => None,
        ChatStreamEvent::Chunk(_) | ChatStreamEvent::Error(_) => None,
    }
}

fn sse(streamer: ChatCompletionStreamer) -> Sse<KeepAliveStream<ChatCompletionStreamer>> {
    Sse::new(streamer)
        .keep_alive(KeepAlive::new().interval(Duration::from_millis(get_keep_alive_interval())))
}
