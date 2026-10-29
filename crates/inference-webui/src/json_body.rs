//! A JSON request body decoded through `FromJson`, answering as axum's `Json` does: 415 without a JSON content type,
//! 400 for malformed JSON, 422 for JSON of the wrong shape.

use axum::{
    body::Bytes,
    extract::{FromRequest, Request},
    http::{HeaderMap, StatusCode, header},
    response::{IntoResponse, Response},
};
use inference_json::FromJson;

const MISSING_CONTENT_TYPE: &str = "Expected request with `Content-Type: application/json`";

pub struct JsonBody<T>(pub T);

impl<S: Send + Sync, T: FromJson> FromRequest<S> for JsonBody<T> {
    type Rejection = Response;

    async fn from_request(request: Request, state: &S) -> Result<Self, Response> {
        if !json_content_type(request.headers()) {
            return Err((StatusCode::UNSUPPORTED_MEDIA_TYPE, MISSING_CONTENT_TYPE).into_response());
        }
        let body = Bytes::from_request(request, state)
            .await
            .map_err(IntoResponse::into_response)?;
        let value = inference_json::parse_slice(&body).map_err(|e| match e.is_syntax() {
            true => {
                let message = format!("Failed to parse the request body as JSON: {e}");
                (StatusCode::BAD_REQUEST, message).into_response()
            }
            false => data_rejection(&e),
        })?;
        T::from_json(&value)
            .map(Self)
            .map_err(|e| data_rejection(&e))
    }
}

fn data_rejection(e: &inference_json::JsonError) -> Response {
    let message = format!("Failed to deserialize the JSON body into the target type: {e}");
    (StatusCode::UNPROCESSABLE_ENTITY, message).into_response()
}

// axum's rule: `application/json` or any `application/*+json`
fn json_content_type(headers: &HeaderMap) -> bool {
    let Some(mime) = headers
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<mime::Mime>().ok())
    else {
        return false;
    };
    mime.type_() == "application"
        && (mime.subtype() == "json" || mime.suffix().is_some_and(|s| s == "json"))
}
