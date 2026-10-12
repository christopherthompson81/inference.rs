//! The image routes: HTTP framing over the engine's image generation and editing.

use axum::{
    extract::{FromRequest, Multipart, OriginalUri, Request},
    http::header,
    response::Response,
};
use inference_api::api_error::ApiError;
use inference_api::media_source::{ATTACHMENT_PREFIX, MediaAttachment, MediaAttachments};
use inference_api::response::ImageGenerationResponse;

use crate::form::{Field, invalid, multipart_payload, read_form};
use crate::handler_core::{ApiJson, ApiJsonRejection};
use crate::{
    handler_core::{json_response, openai_error_response},
    openai::{ImageEditRequest, ImageGenerationRequest, ImageReference},
    route_registry::{IMAGE_EDIT_ROUTE, IMAGE_GENERATION_ROUTE},
    types::OwnedEngine,
};

const IMAGE_FIELD: &str = "image";
const MASK_FIELD: &str = "mask";
const SIZE_FIELD: &str = "size";
const WIDTH_FIELD: &str = "width";
const HEIGHT_FIELD: &str = "height";
// OpenAI's `size` lets the model choose, which here is the first image's own size
const AUTO_SIZE: &str = "auto";
const MULTIPART: &str = "multipart/form-data";
const INVALID_CODE: &str = "invalid_image_request";
const EDIT_FIELDS: &[(&str, Field)] = &[
    ("n", Field::Integer),
    (WIDTH_FIELD, Field::Integer),
    (HEIGHT_FIELD, Field::Integer),
    ("steps", Field::Integer),
    ("guidance_scale", Field::Number),
    ("seed", Field::Integer),
];

/// Image generation endpoint handler.
#[cfg_attr(test, utoipa::path(
    post,
    tag = "inference.rs",
    path = "/v1/images/generations",
    request_body = ImageGenerationRequest,
    responses((status = 200, description = "Image generation", body = inference_api::response::ImageGenerationResponse))
))]
pub async fn image_generation(
    OwnedEngine(engine): OwnedEngine,
    OriginalUri(uri): OriginalUri,
    payload: Result<ApiJson<ImageGenerationRequest>, ApiJsonRejection>,
) -> Response {
    let request = match payload {
        Ok(ApiJson(request)) => request,
        Err(ApiJsonRejection(error)) => return openai_error_response(error),
    };
    let prefix = router_prefix(uri.path(), IMAGE_GENERATION_ROUTE.path);
    json_response(
        engine
            .image_generation(request)
            .await
            .map(|r| prefixed(r, prefix)),
    )
}

/// The multipart form `/v1/images/edits` takes; a JSON `ImageEditRequest` names its images by URL instead.
#[cfg(test)]
#[derive(utoipa::ToSchema)]
#[allow(dead_code)]
pub struct ImageEditForm {
    /// The image to edit, or (repeated as `image[]`) the subject references: PNG, JPEG, WebP and the like.
    #[schema(value_type = Vec<String>, format = Binary)]
    image: Vec<Vec<u8>>,
    prompt: String,
    model: Option<String>,
    n: Option<usize>,
    response_format: Option<String>,
    /// `<width>x<height>`, or `auto` (as no size): one image keeps its aspect, several take the first's.
    size: Option<String>,
    /// In place of `size`, as the JSON body takes them.
    width: Option<usize>,
    height: Option<usize>,
    steps: Option<usize>,
    guidance_scale: Option<f64>,
    seed: Option<u64>,
}

/// Image edit endpoint handler: a JSON body, or OpenAI's multipart form with the images as files.
#[cfg_attr(test, utoipa::path(
    post,
    tag = "inference.rs",
    path = "/v1/images/edits",
    request_body(content(
        (ImageEditRequest = "application/json"),
        (inline(ImageEditForm) = "multipart/form-data")
    )),
    responses((status = 200, description = "The edited images", body = inference_api::response::ImageGenerationResponse))
))]
pub async fn image_edit(
    OwnedEngine(engine): OwnedEngine,
    OriginalUri(uri): OriginalUri,
    request: Request,
) -> Response {
    let multipart = request
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.starts_with(MULTIPART));
    let parsed = if multipart {
        edit_form(request).await
    } else {
        match ApiJson::<ImageEditRequest>::from_request(request, &()).await {
            Ok(ApiJson(request)) => Ok((request, MediaAttachments::default())),
            Err(ApiJsonRejection(error)) => Err(error),
        }
    };
    let (request, media) = match parsed {
        Ok(parsed) => parsed,
        Err(error) => return openai_error_response(error),
    };
    let prefix = router_prefix(uri.path(), IMAGE_EDIT_ROUTE.path);
    json_response(
        engine
            .image_edit(request, media)
            .await
            .map(|r| prefixed(r, prefix)),
    )
}

// the form's files become the request's attachments, each named `media://N` in order
async fn edit_form(request: Request) -> Result<(ImageEditRequest, MediaAttachments), ApiError> {
    let multipart = multipart_payload(Multipart::from_request(request, &()).await, INVALID_CODE)?;
    let (mut fields, uploads) = read_form::<serde_json::Map<String, serde_json::Value>>(
        multipart,
        EDIT_FIELDS,
        &[IMAGE_FIELD, MASK_FIELD],
        INVALID_CODE,
    )
    .await?;
    if uploads.iter().any(|u| u.field == MASK_FIELD) {
        return Err(invalid(
            "Masked edits are not supported; omit `mask`.",
            MASK_FIELD,
            INVALID_CODE,
        ));
    }
    if let Some(size) = fields.remove(SIZE_FIELD) {
        if fields.contains_key(WIDTH_FIELD) || fields.contains_key(HEIGHT_FIELD) {
            let message = "Give `size` or `width` and `height`, not both.";
            return Err(invalid(message, SIZE_FIELD, INVALID_CODE));
        }
        let size = size.as_str().unwrap_or_default().to_string();
        if size != AUTO_SIZE {
            let (width, height) = size
                .split_once('x')
                .and_then(|(w, h)| Some((w.parse::<u64>().ok()?, h.parse::<u64>().ok()?)))
                .ok_or_else(|| {
                    invalid(
                        format!("`{size}` is not <width>x<height> or auto"),
                        SIZE_FIELD,
                        INVALID_CODE,
                    )
                })?;
            fields.insert(WIDTH_FIELD.to_string(), width.into());
            fields.insert(HEIGHT_FIELD.to_string(), height.into());
        }
    }
    let images: Vec<ImageReference> = (0..uploads.len())
        .map(|i| ImageReference {
            image_url: format!("{ATTACHMENT_PREFIX}{i}"),
        })
        .collect();
    fields.insert(
        "images".to_string(),
        serde_json::to_value(images).map_err(|e| invalid(e.to_string(), "body", INVALID_CODE))?,
    );
    let request = serde_json::from_value(serde_json::Value::Object(fields))
        .map_err(|e| invalid(e.to_string(), "body", INVALID_CODE))?;
    let media = MediaAttachments::new(
        uploads
            .into_iter()
            .map(|u| MediaAttachment {
                bytes: u.bytes,
                mime_type: u.content_type,
            })
            .collect(),
    );
    Ok((request, media))
}

fn prefixed(mut response: ImageGenerationResponse, prefix: &str) -> ImageGenerationResponse {
    for url in response
        .data
        .iter_mut()
        .filter_map(|choice| choice.url.as_mut())
    {
        url.insert_str(0, prefix);
    }
    response
}

// File-store urls are relative to this router, which a host app may have nested under a prefix.
fn router_prefix<'a>(request_path: &'a str, route: &str) -> &'a str {
    request_path.strip_suffix(route).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::{IMAGE_GENERATION_ROUTE, router_prefix};

    #[test]
    fn the_prefix_is_whatever_nests_the_route() {
        let route = IMAGE_GENERATION_ROUTE.path;
        assert_eq!(router_prefix("/v1/images/generations", route), "");
        assert_eq!(
            router_prefix("/api/inference/v1/images/generations", route),
            "/api/inference"
        );
    }
}
