//! ## Chat Completions functionality and route handler.

use std::{pin::Pin, sync::Arc, task::Poll, time::Duration};

use axum::{
    extract::{rejection::JsonRejection, Json, State},
    response::{
        sse::{Event, KeepAlive, KeepAliveStream},
        IntoResponse, Sse,
    },
    Extension,
};
use inference_core::{ChatCompletionChunkResponse, ChatCompletionResponse, InferenceRs, Response};
use tokio::sync::mpsc::Receiver;

pub use crate::engine_chat::{
    parse_request, serialize_agentic_progress, ChatCompletionParseContext,
};
use crate::handler_core::ApiErrorHttp;
use crate::{
    agentic::AgenticDefaults,
    completion_core::{
        handle_completion_error, handle_completion_validation_error, BaseCompletionResponder,
    },
    engine_chat::{
        collect_chat, ChatDispatchError, ChatEngine, ChatStream, ChatStreamEvent, ResponseTap,
    },
    handler_core::{
        openai_error_from_error, openai_error_response, ApiError, ApiErrorKind, ModelErrorMessage,
    },
    openai::{
        ChatCompletionChunkResponseBody, ChatCompletionRequest, ChatCompletionResponseBody,
        OpenAiToolSurface,
    },
    skills::SkillStore,
    streaming::{get_keep_alive_interval, openai_error_event, DoneState, StreamOutcomeHandle},
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
                    ChatStreamEvent::AgenticToolCallProgress(payload)
                    | ChatStreamEvent::AgenticToolApprovalRequired(payload) => {
                        Event::default().json_data(payload)
                    }
                    ChatStreamEvent::FileProduced(file) => Event::default().json_data(file),
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
            ChatCompletionResponder::InternalError(e) => {
                openai_error_from_error(e.as_ref(), ApiErrorKind::Internal)
            }
            ChatCompletionResponder::ValidationError(e) => {
                openai_error_from_error(e.as_ref(), ApiErrorKind::InvalidRequest)
            }
            ChatCompletionResponder::ModelError(_, _) => {
                openai_error_response(ApiError::model_error())
            }
        }
    }
}

/// OpenAI-compatible chat completions endpoint handler.
#[utoipa::path(
    post,
    tag = "Mistral.rs",
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
)]
pub async fn chatcompletions(
    State(state): ExtractedInferenceRsState,
    Extension(agentic_defaults): Extension<AgenticDefaults>,
    Extension(skill_store): Extension<Arc<SkillStore>>,
    stream_outcome: Option<Extension<StreamOutcomeHandle>>,
    payload: Result<Json<ChatCompletionRequest>, JsonRejection>,
) -> ChatCompletionResponder {
    let oairequest = match payload {
        Ok(Json(request)) => request,
        Err(error) => {
            return ChatCompletionResponder::ValidationError(Box::new(
                ApiError::from_json_rejection(error),
            ));
        }
    };
    let engine = ChatEngine {
        state: state.clone(),
        agentic: agentic_defaults,
        skill_store: Some(skill_store),
    };
    let prepared = match engine
        .prepare(
            oairequest,
            OpenAiToolSurface::ChatCompletions,
            Default::default(),
        )
        .await
    {
        Ok(prepared) => prepared,
        Err(ChatDispatchError::Validation(e)) => {
            return handle_completion_validation_error(state, e)
        }
        Err(ChatDispatchError::Internal(e)) => return handle_error(state, e),
    };

    if prepared.is_streaming {
        let tap = stream_outcome.map(|Extension(handle)| {
            Box::new(move |response: &Response| handle.observe(response)) as ResponseTap
        });
        let stream = ChatStream::new(prepared.rx, state, prepared.model_override, tap);
        ChatCompletionResponder::Sse(sse(ChatCompletionStreamer::new(stream, None, None)))
    } else {
        let mut rx = prepared.rx;
        let response = collect_chat(&mut rx, prepared.model_override.as_deref()).await;
        match_responses(state, response)
    }
}

/// Handle route / generation errors and logging them.
pub fn handle_error(
    state: SharedInferenceRsState,
    e: Box<dyn std::error::Error + Send + Sync + 'static>,
) -> ChatCompletionResponder {
    handle_completion_error(state, e)
}

/// Creates a SSE streamer for chat completions with optional callbacks.
pub fn create_streamer(
    rx: Receiver<Response>,
    state: SharedInferenceRsState,
    on_chunk: Option<ChatCompletionOnChunkCallback>,
    on_done: Option<ChatCompletionOnDoneCallback>,
) -> Sse<KeepAliveStream<ChatCompletionStreamer>> {
    create_streamer_with_outcome(rx, state, on_chunk, on_done, None)
}

/// Like [`create_streamer`], also reporting usage and errors to the access log at stream end.
pub fn create_streamer_with_outcome(
    rx: Receiver<Response>,
    state: SharedInferenceRsState,
    on_chunk: Option<ChatCompletionOnChunkCallback>,
    on_done: Option<ChatCompletionOnDoneCallback>,
    outcome: Option<StreamOutcomeHandle>,
) -> Sse<KeepAliveStream<ChatCompletionStreamer>> {
    let tap = outcome
        .map(|handle| Box::new(move |response: &Response| handle.observe(response)) as ResponseTap);
    let stream = ChatStream::new(rx, state, None, tap);
    sse(ChatCompletionStreamer::new(stream, on_chunk, on_done))
}

// Chunks and errors go out as plain `data:` lines; the other events are named.
fn sse_event_name(event: &ChatStreamEvent) -> Option<&'static str> {
    match event {
        ChatStreamEvent::AgenticToolCallProgress(_) => Some("agentic_tool_call_progress"),
        ChatStreamEvent::AgenticToolApprovalRequired(_) => Some("agentic_tool_approval_required"),
        ChatStreamEvent::FileProduced(_) => Some("file_produced"),
        ChatStreamEvent::Chunk(_) | ChatStreamEvent::Error(_) => None,
    }
}

fn sse(streamer: ChatCompletionStreamer) -> Sse<KeepAliveStream<ChatCompletionStreamer>> {
    Sse::new(streamer)
        .keep_alive(KeepAlive::new().interval(Duration::from_millis(get_keep_alive_interval())))
}

/// Process non-streaming chat completion responses.
pub async fn process_non_streaming_response(
    rx: &mut Receiver<Response>,
    state: SharedInferenceRsState,
) -> ChatCompletionResponder {
    match_responses(state, collect_chat(rx, None).await)
}

/// Matches and processes different types of model responses into appropriate chat completion responses.
pub fn match_responses(
    state: SharedInferenceRsState,
    response: Response,
) -> ChatCompletionResponder {
    match response {
        Response::InternalError(e) => {
            InferenceRs::maybe_log_error(state, &*e);
            ChatCompletionResponder::InternalError(e)
        }
        Response::ModelError(msg, response) => {
            InferenceRs::maybe_log_error(state.clone(), &ModelErrorMessage(msg.to_string()));
            InferenceRs::maybe_log_response(state, &response);
            ChatCompletionResponder::ModelError(msg, response)
        }
        Response::ValidationError(e) => ChatCompletionResponder::ValidationError(e),
        Response::Done(response) => {
            InferenceRs::maybe_log_response(state, &response);
            ChatCompletionResponder::Json(response)
        }
        Response::Chunk(_) => unreachable!(),
        Response::CompletionDone(_) => unreachable!(),
        Response::CompletionModelError(_, _) => unreachable!(),
        Response::CompletionChunk(_) => unreachable!(),
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
