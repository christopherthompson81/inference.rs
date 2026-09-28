//! The image generation route: HTTP framing over the engine's image generation.

use axum::{
    extract::{rejection::JsonRejection, Json, State},
    response::Response,
};

use crate::{
    generation::generate_image,
    handler_core::{json_response, openai_error_response, ApiError, ApiErrorHttp},
    openai::ImageGenerationRequest,
    types::ExtractedInferenceRsState,
};

/// Image generation endpoint handler.
#[utoipa::path(
    post,
    tag = "Mistral.rs",
    path = "/v1/images/generations",
    request_body = ImageGenerationRequest,
    responses((status = 200, description = "Image generation", body = inference_core::ImageGenerationResponse))
)]
pub async fn image_generation(
    State(state): ExtractedInferenceRsState,
    payload: Result<Json<ImageGenerationRequest>, JsonRejection>,
) -> Response {
    match payload {
        Ok(Json(request)) => json_response(generate_image(&state, request).await),
        Err(error) => openai_error_response(ApiError::from_json_rejection(error)),
    }
}
