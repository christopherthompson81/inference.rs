//! Errors the SDK returns: the engine's own, or a load that failed.

use inference_api::{EngineLoadError, api_error::ApiError};
use thiserror::Error;

#[derive(Error, Debug)]
#[non_exhaustive]
pub enum Error {
    /// The engine refused or failed a request; the kind says whose fault it was.
    #[error(transparent)]
    Api(#[from] ApiError),

    #[error("model loading failed: {0}")]
    ModelLoad(#[from] EngineLoadError),

    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("fetch failed: {0}")]
    Fetch(String),

    /// A request the SDK could not build.
    #[error("invalid request: {0}")]
    Request(String),

    #[error("{0}")]
    Io(#[from] std::io::Error),

    /// The engine answered without what the call asked for.
    #[error("the engine returned no result")]
    Empty,
}

pub type Result<T> = std::result::Result<T, Error>;
