//! The image generation route: HTTP framing over the engine's image generation.

use axum::{
    extract::{OriginalUri, State},
    response::Response,
};

use crate::handler_core::{ApiJson, ApiJsonRejection};
use crate::{
    generation::generate_image,
    handler_core::{json_response, openai_error_response},
    openai::ImageGenerationRequest,
    route_registry::IMAGE_GENERATION_ROUTE,
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
    OriginalUri(uri): OriginalUri,
    payload: Result<ApiJson<ImageGenerationRequest>, ApiJsonRejection>,
) -> Response {
    let request = match payload {
        Ok(ApiJson(request)) => request,
        Err(ApiJsonRejection(error)) => return openai_error_response(error),
    };
    let prefix = router_prefix(uri.path());
    json_response(generate_image(&state, request).await.map(|mut response| {
        for url in response
            .data
            .iter_mut()
            .filter_map(|choice| choice.url.as_mut())
        {
            url.insert_str(0, prefix);
        }
        response
    }))
}

// File-store urls are relative to this router, which a host app may have nested under a prefix.
fn router_prefix(request_path: &str) -> &str {
    request_path
        .strip_suffix(IMAGE_GENERATION_ROUTE.path)
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::router_prefix;

    #[test]
    fn the_prefix_is_whatever_nests_the_route() {
        assert_eq!(router_prefix("/v1/images/generations"), "");
        assert_eq!(
            router_prefix("/api/inference/v1/images/generations"),
            "/api/inference"
        );
    }
}
