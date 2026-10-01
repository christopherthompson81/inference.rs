//! OpenAI-compatible embeddings endpoint.

use axum::response::Response;

use crate::handler_core::{ApiJson, ApiJsonRejection};
#[cfg(test)]
use crate::openai::EmbeddingResponse;
use crate::{
    handler_core::{json_response, openai_error_response},
    openai::EmbeddingRequest,
    types::OwnedEngine,
};

#[cfg_attr(test, utoipa::path(
    post,
    tag = "inference.rs",
    path = "/v1/embeddings",
    request_body = EmbeddingRequest,
    responses((status = 200, description = "Embeddings", body = EmbeddingResponse))
))]
pub async fn embeddings(
    OwnedEngine(engine): OwnedEngine,
    payload: Result<ApiJson<EmbeddingRequest>, ApiJsonRejection>,
) -> Response {
    match payload {
        Ok(ApiJson(request)) => json_response(engine.embeddings(request).await),
        Err(ApiJsonRejection(error)) => openai_error_response(error),
    }
}
