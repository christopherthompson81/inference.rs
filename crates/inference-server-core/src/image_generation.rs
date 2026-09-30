//! The image generation route: HTTP framing over the engine's image generation.

use axum::{extract::State, response::Response};

use crate::handler_core::{ApiJson, ApiJsonRejection};
use crate::{
    generation::generate_image,
    handler_core::{json_response, openai_error_response},
    openai::ImageGenerationRequest,
    types::ExtractedInferenceRsState,
};

/// Image generation endpoint handler.
#[cfg_attr(test, utoipa::path(
    post,
    tag = "inference.rs",
    path = "/v1/images/generations",
    request_body = ImageGenerationRequest,
    responses((status = 200, description = "Image generation", body = inference_core::ImageGenerationResponse))
))]
pub async fn image_generation(
    State(state): ExtractedInferenceRsState,
    payload: Result<ApiJson<ImageGenerationRequest>, ApiJsonRejection>,
) -> Response {
    match payload {
        Ok(ApiJson(request)) => json_response(generate_image(&state, request).await),
        Err(ApiJsonRejection(error)) => openai_error_response(error),
    }
}
