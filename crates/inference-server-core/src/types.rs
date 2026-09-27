//! ## Custom types used in inference.rs server core.

use axum::extract::State;
pub use inference_api::types::*;

/// The engine state as an axum extractor.
pub type ExtractedInferenceRsState = State<SharedInferenceRsState>;
