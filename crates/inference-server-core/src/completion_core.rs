//! Core functionality for completions.

use axum::response::Sse;

use crate::handler_core::ApiError;

/// A completion route's answer: an event stream, the whole response, or an error.
#[derive(Debug)]
pub enum BaseCompletionResponder<R, S> {
    Sse(Sse<S>),
    Json(R),
    Error(ApiError),
}
