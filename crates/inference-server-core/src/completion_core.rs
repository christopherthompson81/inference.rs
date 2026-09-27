//! Core functionality for completions.

use std::error::Error;

use axum::response::Sse;
use inference_core::InferenceRs;

use crate::{
    handler_core::{ApiError, ApiErrorKind},
    types::SharedInferenceRsState,
};

/// Generic responder enum for different completion types.
#[derive(Debug)]
pub enum BaseCompletionResponder<R, S> {
    /// Server-Sent Events streaming response
    Sse(Sse<S>),
    /// Complete JSON response for non-streaming requests
    Json(R),
    /// Model error with partial response data
    ModelError(String, R),
    /// Internal server error
    InternalError(Box<dyn Error>),
    /// Request validation error
    ValidationError(Box<dyn Error>),
}

/// Generic function to handle completion errors and logging them.
pub(crate) fn handle_completion_error<R, S>(
    state: SharedInferenceRsState,
    e: Box<dyn std::error::Error + Send + Sync + 'static>,
) -> BaseCompletionResponder<R, S> {
    InferenceRs::maybe_log_error(state, e.as_ref());
    BaseCompletionResponder::InternalError(e)
}

pub(crate) fn handle_completion_validation_error<R, S>(
    state: SharedInferenceRsState,
    e: Box<dyn std::error::Error + Send + Sync + 'static>,
) -> BaseCompletionResponder<R, S> {
    let error = ApiError::from_error(e.as_ref(), ApiErrorKind::InvalidRequest);
    if matches!(
        error.kind,
        ApiErrorKind::Internal | ApiErrorKind::Unavailable | ApiErrorKind::Overloaded
    ) {
        InferenceRs::maybe_log_error(state, e.as_ref());
    }
    BaseCompletionResponder::ValidationError(e)
}
