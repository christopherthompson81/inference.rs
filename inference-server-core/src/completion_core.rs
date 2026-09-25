//! Core functionality for completions.

use std::error::Error;

use anyhow::Result;
use axum::response::Sse;
use inference_core::{DrySamplingParams, InferenceRs, StopTokens as InternalStopTokens};

use crate::{
    handler_core::{ApiError, ApiErrorKind},
    openai::StopTokens,
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

/// Helper function to convert from the OpenAI stop tokens to the inference.rs
/// internal stop tokens.
pub(crate) fn convert_stop_tokens(stop_seqs: Option<StopTokens>) -> Option<InternalStopTokens> {
    match stop_seqs {
        Some(StopTokens::Multi(sequences)) => Some(InternalStopTokens::Seqs(sequences)),
        Some(StopTokens::Single(sequence)) => Some(InternalStopTokens::Seqs(vec![sequence])),
        None => None,
    }
}

/// Helper function to get the dry sampling params.
pub(crate) fn get_dry_sampling_params(
    dry_multiplier: Option<f32>,
    dry_sequence_breakers: Option<Vec<String>>,
    dry_base: Option<f32>,
    dry_allowed_length: Option<usize>,
) -> Result<Option<DrySamplingParams>> {
    match dry_multiplier {
        Some(multiplier) => {
            let params = DrySamplingParams::new_with_defaults(
                multiplier,
                dry_sequence_breakers,
                dry_base,
                dry_allowed_length,
            )?;
            Ok(Some(params))
        }
        None => Ok(None),
    }
}
