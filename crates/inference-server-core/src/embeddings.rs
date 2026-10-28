//! OpenAI-compatible embeddings endpoint, reranking with a cross-encoder, and answers from a decision model.

use axum::response::Response;

use crate::handler_core::{ApiJson, ApiJsonRejection};
#[cfg(test)]
use crate::openai::{DecisionResponse, EmbeddingResponse, RerankResponse};
use crate::{
    handler_core::{json_response, openai_error_response},
    openai::{DecisionRequest, EmbeddingRequest, RerankRequest},
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

#[cfg_attr(test, utoipa::path(
    post,
    tag = "inference.rs",
    path = "/v1/rerank",
    request_body = RerankRequest,
    responses((status = 200, description = "Documents ordered by relevance", body = RerankResponse))
))]
pub async fn rerank(
    OwnedEngine(engine): OwnedEngine,
    payload: Result<ApiJson<RerankRequest>, ApiJsonRejection>,
) -> Response {
    match payload {
        Ok(ApiJson(request)) => json_response(engine.rerank(request).await),
        Err(ApiJsonRejection(error)) => openai_error_response(error),
    }
}

#[cfg_attr(test, utoipa::path(
    post,
    tag = "inference.rs",
    path = "/v1/decisions",
    request_body = DecisionRequest,
    responses((status = 200, description = "Each question's answer", body = DecisionResponse))
))]
pub async fn decisions(
    OwnedEngine(engine): OwnedEngine,
    payload: Result<ApiJson<DecisionRequest>, ApiJsonRejection>,
) -> Response {
    match payload {
        Ok(ApiJson(request)) => json_response(engine.decide(request).await),
        Err(ApiJsonRejection(error)) => openai_error_response(error),
    }
}
