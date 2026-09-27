//! SSE streaming utilities.

use std::{
    env,
    sync::{Arc, Mutex},
    time::Instant,
};

use inference_core::{Response, Usage};

use crate::{
    engine_chat::ResponseTap,
    handler_core::ApiError,
    metrics::{ITL_METRIC, TTFT_METRIC},
    util::sanitize_error_message,
};

/// What a streaming request produced, filled in by the streamer and read once the SSE body ends.
#[derive(Debug, Clone, Default)]
pub struct StreamOutcome {
    pub usage: Option<Usage>,
    /// The error message sent to the client, if the request failed
    pub error: Option<String>,
}

/// Shared between the observability middleware and the streamer that serves the request.
#[derive(Debug, Clone, Default)]
pub struct StreamOutcomeHandle(Arc<Mutex<StreamOutcomeState>>);

#[derive(Debug, Default)]
struct StreamOutcomeState {
    usage: Option<Usage>,
    error: Option<String>,
    latency: Option<StreamLatency>,
    last_token_at: Option<Instant>,
}

/// Set at body-wrap time, before any engine response can be observed.
#[derive(Debug)]
struct StreamLatency {
    labels: [(&'static str, String); 4],
    start: Instant,
}

impl StreamOutcomeHandle {
    /// A tap for an engine stream that records into this handle.
    pub fn tap(self) -> ResponseTap {
        Box::new(move |response: &Response| self.observe(response))
    }

    pub fn snapshot(&self) -> StreamOutcome {
        let state = self.0.lock().expect("stream outcome poisoned");
        StreamOutcome {
            usage: state.usage.clone(),
            error: state.error.clone(),
        }
    }

    /// Enable TTFT/ITL recording with the request's labels and arrival time.
    pub fn set_latency_labels(&self, labels: [(&'static str, String); 4], start: Instant) {
        let mut state = self.0.lock().expect("stream outcome poisoned");
        state.latency = Some(StreamLatency { labels, start });
    }

    /// Record usage, errors, and streaming latency from any engine response as it flows through a streamer.
    pub fn observe(&self, response: &Response) {
        let mut state = self.0.lock().expect("stream outcome poisoned");
        match response {
            Response::Chunk(chunk) => {
                if let Some(usage) = &chunk.usage {
                    state.usage = Some(usage.clone());
                }
            }
            Response::Done(done) => state.usage = Some(done.usage.clone()),
            Response::CompletionDone(done) => state.usage = Some(done.usage.clone()),
            Response::ModelError(msg, _) | Response::CompletionModelError(msg, _) => {
                state.error = Some(msg.clone())
            }
            Response::InternalError(e) | Response::ValidationError(e) => {
                state.error = Some(sanitize_error_message(e.as_ref()))
            }
            _ => {}
        }
        // Only token steps carry decoded output; agentic control events and done markers share the channel
        if matches!(response, Response::Chunk(_) | Response::CompletionChunk(_)) {
            let Some(latency) = &state.latency else {
                return;
            };
            let now = Instant::now();
            if let Some(last) = state.last_token_at {
                metrics::histogram!(ITL_METRIC, &latency.labels)
                    .record(now.duration_since(last).as_secs_f64());
            } else {
                metrics::histogram!(TTFT_METRIC, &latency.labels)
                    .record(now.duration_since(latency.start).as_secs_f64());
            }
            state.last_token_at = Some(now);
        }
    }
}

/// OpenAI-style streaming error event; clients parse `data:` lines as JSON, so a raw text line is invisible to them.
pub(crate) fn openai_error_event(error: ApiError) -> axum::response::sse::Event {
    axum::response::sse::Event::default().data(error.to_openai_body().to_string())
}

/// Default keep-alive interval for Server-Sent Events (SSE) streams in milliseconds.
pub const DEFAULT_KEEP_ALIVE_INTERVAL_MS: u64 = 10_000;

/// Represents the current state of a streaming response.
pub enum DoneState {
    /// The stream is actively processing and sending response chunks
    Running,
    /// The stream has finished processing and is about to send the `[DONE]` message
    SendingDone,
    /// The stream has completed entirely
    Done,
}

/// Gets the keep-alive interval for SSE streams from environment or default.
pub fn get_keep_alive_interval() -> u64 {
    env::var("KEEP_ALIVE_INTERVAL")
        .map(|val| {
            val.parse::<u64>().unwrap_or_else(|e| {
                tracing::warn!("Failed to parse KEEP_ALIVE_INTERVAL: {}. Using default.", e);
                DEFAULT_KEEP_ALIVE_INTERVAL_MS
            })
        })
        .unwrap_or(DEFAULT_KEEP_ALIVE_INTERVAL_MS)
}
