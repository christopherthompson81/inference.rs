//! OpenAI-compatible embeddings endpoint.

use axum::{
    extract::{Json, State, rejection::JsonRejection},
    response::IntoResponse,
};

use crate::{
    engine_embeddings::{EmbeddingError, embed},
    handler_core::{ApiError, ApiErrorHttp, ApiErrorKind, openai_error_from_error},
    openai::{EmbeddingRequest, EmbeddingResponse},
    types::ExtractedInferenceRsState,
};

pub enum EmbeddingResponder {
    Json(EmbeddingResponse),
    InternalError(anyhow::Error),
    ValidationError(anyhow::Error),
}

impl IntoResponse for EmbeddingResponder {
    fn into_response(self) -> axum::response::Response {
        match self {
            EmbeddingResponder::Json(s) => Json(s).into_response(),
            EmbeddingResponder::InternalError(e) => {
                openai_error_from_error(e.as_ref(), ApiErrorKind::Internal)
            }
            EmbeddingResponder::ValidationError(e) => {
                openai_error_from_error(e.as_ref(), ApiErrorKind::InvalidRequest)
            }
        }
    }
}

#[utoipa::path(
    post,
    tag = "inference.rs",
    path = "/v1/embeddings",
    request_body = EmbeddingRequest,
    responses((status = 200, description = "Embeddings", body = EmbeddingResponse))
)]
pub async fn embeddings(
    State(state): ExtractedInferenceRsState,
    payload: Result<Json<EmbeddingRequest>, JsonRejection>,
) -> EmbeddingResponder {
    let oairequest = match payload {
        Ok(Json(request)) => request,
        Err(error) => {
            return EmbeddingResponder::ValidationError(
                ApiError::from_json_rejection(error).into(),
            );
        }
    };
    match embed(state, oairequest).await {
        Ok(response) => EmbeddingResponder::Json(response),
        Err(EmbeddingError::Validation(e)) => EmbeddingResponder::ValidationError(e),
        Err(EmbeddingError::Internal(e)) => EmbeddingResponder::InternalError(e),
    }
}
