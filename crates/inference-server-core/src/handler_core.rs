//! Core functionality for handlers.

use axum::{
    body::Bytes,
    extract::{FromRequest, Json, Request},
    http::{HeaderMap, StatusCode, header},
    response::IntoResponse,
};
use inference_api::request_body::{INVALID_REQUEST_BODY, JsonRequest, MALFORMED_JSON};

pub(crate) use crate::api_error::{
    ApiError, ApiErrorKind, ModelErrorMessage, SERVICE_UNAVAILABLE_MESSAGE,
};
pub use crate::dispatch::{
    DEFAULT_CHANNEL_BUFFER_SIZE, create_response_channel, send_request, send_request_with_model,
};

const INVALID_CONTENT_TYPE: &str = "invalid_content_type";
const REQUEST_BODY_TOO_LARGE: &str = "request_body_too_large";
const MISSING_JSON_CONTENT_TYPE: &str = "Expected request with `Content-Type: application/json`";

// Rate-limited operations (a busy adapter load) clear within a request's time, so clients retry promptly.
const RETRY_AFTER_SECS: &str = "1";

/// Error message attached to a failed response so the access log can report it.
#[derive(Clone, Debug)]
pub struct ResponseErrorMessage(pub String);

pub(crate) fn openai_error_response(error: ApiError) -> axum::response::Response {
    let mut response = Json(error.to_openai_body()).into_response();
    *response.status_mut() = error.status();
    if error.kind == ApiErrorKind::RateLimited {
        response.headers_mut().insert(
            axum::http::header::RETRY_AFTER,
            axum::http::HeaderValue::from_static(RETRY_AFTER_SECS),
        );
    }
    response
        .extensions_mut()
        .insert(ResponseErrorMessage(error.message));
    response
}

pub(crate) fn json_response<T: serde::Serialize>(
    result: Result<T, ApiError>,
) -> axum::response::Response {
    match result {
        Ok(body) => Json(body).into_response(),
        Err(error) => openai_error_response(error),
    }
}

pub(crate) fn openai_error_from_error(
    error: &(dyn std::error::Error + 'static),
    fallback: ApiErrorKind,
) -> axum::response::Response {
    openai_error_response(ApiError::from_error(error, fallback))
}

/// A JSON body parsed by its type's [`JsonRequest`] impl, whose deserializer is compiled once in inference-api.
#[derive(Debug)]
pub struct ApiJson<T>(pub T);

/// Why an [`ApiJson`] body was refused; handlers unwrap it to answer in their own error format.
pub struct ApiJsonRejection(pub ApiError);

impl IntoResponse for ApiJsonRejection {
    fn into_response(self) -> axum::response::Response {
        openai_error_response(self.0)
    }
}

impl<S: Send + Sync, T: JsonRequest> FromRequest<S> for ApiJson<T> {
    type Rejection = ApiJsonRejection;

    async fn from_request(request: Request, _state: &S) -> Result<Self, ApiJsonRejection> {
        let body = json_body(request).await.map_err(ApiJsonRejection)?;
        T::from_json(&body).map(Self).map_err(ApiJsonRejection)
    }
}

async fn json_body(request: Request) -> Result<Bytes, ApiError> {
    if !json_content_type(request.headers()) {
        return Err(body_rejection(
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            MISSING_JSON_CONTENT_TYPE.to_string(),
        ));
    }
    Bytes::from_request(request, &())
        .await
        .map_err(|rejection| body_rejection(rejection.status(), rejection.body_text()))
}

// Same rule and parser as axum's `Json`: `application/json` or any `application/*+json`.
fn json_content_type(headers: &HeaderMap) -> bool {
    let Some(mime) = headers
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<mime::Mime>().ok())
    else {
        return false;
    };
    mime.type_() == "application"
        && (mime.subtype() == "json" || mime.suffix().is_some_and(|suffix| suffix == "json"))
}

fn body_rejection(status: StatusCode, message: String) -> ApiError {
    let code = match status {
        StatusCode::PAYLOAD_TOO_LARGE => REQUEST_BODY_TOO_LARGE,
        StatusCode::UNSUPPORTED_MEDIA_TYPE => INVALID_CONTENT_TYPE,
        StatusCode::UNPROCESSABLE_ENTITY => INVALID_REQUEST_BODY,
        _ => MALFORMED_JSON,
    };
    let kind = match status {
        StatusCode::PAYLOAD_TOO_LARGE => ApiErrorKind::PayloadTooLarge,
        StatusCode::UNSUPPORTED_MEDIA_TYPE => ApiErrorKind::UnsupportedMediaType,
        _ => ApiErrorKind::InvalidRequest,
    };
    ApiError::new(kind, message, Some(code), None)
}

/// HTTP views of an [`ApiError`]: the status it maps to, and errors that start as an HTTP status.
pub(crate) trait ApiErrorHttp {
    fn from_status(status: StatusCode, message: impl Into<String>) -> Self;
    fn status(&self) -> StatusCode;
}

impl ApiErrorHttp for ApiError {
    fn from_status(status: StatusCode, message: impl Into<String>) -> Self {
        let message = message.into();
        match status {
            StatusCode::BAD_REQUEST | StatusCode::UNPROCESSABLE_ENTITY => {
                Self::invalid_request(message)
            }
            StatusCode::NOT_FOUND => {
                Self::new(ApiErrorKind::NotFound, message, Some("not_found"), None)
            }
            StatusCode::FORBIDDEN => {
                Self::new(ApiErrorKind::Forbidden, message, Some("forbidden"), None)
            }
            StatusCode::CONFLICT => {
                Self::new(ApiErrorKind::Conflict, message, Some("conflict"), None)
            }
            StatusCode::PAYLOAD_TOO_LARGE => Self::new(
                ApiErrorKind::PayloadTooLarge,
                message,
                Some("request_body_too_large"),
                None,
            ),
            StatusCode::UNSUPPORTED_MEDIA_TYPE => Self::new(
                ApiErrorKind::UnsupportedMediaType,
                message,
                Some("invalid_content_type"),
                None,
            ),
            StatusCode::TOO_MANY_REQUESTS => Self::new(
                ApiErrorKind::RateLimited,
                message,
                Some("rate_limit_exceeded"),
                None,
            ),
            StatusCode::SERVICE_UNAVAILABLE => Self::new(
                ApiErrorKind::Unavailable,
                SERVICE_UNAVAILABLE_MESSAGE,
                Some("service_unavailable"),
                None,
            ),
            _ if status.is_server_error() => Self::internal(),
            _ => Self::invalid_request(message),
        }
    }

    fn status(&self) -> StatusCode {
        match self.kind {
            ApiErrorKind::InvalidRequest => StatusCode::BAD_REQUEST,
            ApiErrorKind::NotFound => StatusCode::NOT_FOUND,
            ApiErrorKind::Gone => StatusCode::GONE,
            ApiErrorKind::Unauthorized => StatusCode::UNAUTHORIZED,
            ApiErrorKind::Forbidden => StatusCode::FORBIDDEN,
            ApiErrorKind::Conflict => StatusCode::CONFLICT,
            ApiErrorKind::PayloadTooLarge => StatusCode::PAYLOAD_TOO_LARGE,
            ApiErrorKind::UnsupportedMediaType => StatusCode::UNSUPPORTED_MEDIA_TYPE,
            ApiErrorKind::RateLimited => StatusCode::TOO_MANY_REQUESTS,
            ApiErrorKind::Unavailable | ApiErrorKind::Overloaded => StatusCode::SERVICE_UNAVAILABLE,
            ApiErrorKind::Internal => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api_error::INTERNAL_ERROR_MESSAGE;
    use axum::body::to_bytes;

    const CONTENT_TYPE_PROBES: &[&str] = &[
        "application/json",
        "application/json; charset=utf-8",
        "Application/JSON;charset=utf-8",
        "application/cloudevents+json",
        "application/json;",
        "application/json; =x",
        "text/json",
        "application/jsonl",
        "json",
    ];
    // Axum's default body limit, which `Bytes::from_request` applies without a `DefaultBodyLimit` layer.
    const AXUM_DEFAULT_BODY_LIMIT: usize = 2 * 1024 * 1024;

    fn json_request(content_type: Option<&str>, body: impl Into<axum::body::Body>) -> Request {
        let mut builder = Request::builder();
        if let Some(content_type) = content_type {
            builder = builder.header(header::CONTENT_TYPE, content_type);
        }
        builder.body(body.into()).unwrap()
    }

    #[tokio::test]
    async fn content_types_are_accepted_exactly_as_axum_json_accepts_them() {
        for content_type in CONTENT_TYPE_PROBES.iter().map(|c| Some(*c)).chain([None]) {
            let ours = ApiJson::<inference_api::operations::ReIsqRequest>::from_request(
                json_request(content_type, r#"{"ggml_type":"Q4K"}"#),
                &(),
            )
            .await
            .is_ok();
            let axum = Json::<serde_json::Value>::from_request(
                json_request(content_type, r#"{"ggml_type":"Q4K"}"#),
                &(),
            )
            .await
            .is_ok();
            assert_eq!(ours, axum, "{content_type:?}");
        }
    }

    #[tokio::test]
    async fn oversized_bodies_are_payload_too_large() {
        let body = vec![b' '; AXUM_DEFAULT_BODY_LIMIT + 1];
        let ApiJsonRejection(error) =
            ApiJson::<inference_api::operations::ReIsqRequest>::from_request(
                json_request(Some("application/json"), body),
                &(),
            )
            .await
            .unwrap_err();
        assert_eq!(error.status(), StatusCode::PAYLOAD_TOO_LARGE);
        assert_eq!(error.code.as_deref(), Some(REQUEST_BODY_TOO_LARGE));
    }

    #[test]
    fn every_error_kind_maps_to_its_status() {
        for (kind, status) in [
            (ApiErrorKind::InvalidRequest, StatusCode::BAD_REQUEST),
            (ApiErrorKind::NotFound, StatusCode::NOT_FOUND),
            (ApiErrorKind::Gone, StatusCode::GONE),
            (ApiErrorKind::Unauthorized, StatusCode::UNAUTHORIZED),
            (ApiErrorKind::Forbidden, StatusCode::FORBIDDEN),
            (ApiErrorKind::Conflict, StatusCode::CONFLICT),
            (ApiErrorKind::PayloadTooLarge, StatusCode::PAYLOAD_TOO_LARGE),
            (
                ApiErrorKind::UnsupportedMediaType,
                StatusCode::UNSUPPORTED_MEDIA_TYPE,
            ),
            (ApiErrorKind::RateLimited, StatusCode::TOO_MANY_REQUESTS),
            (ApiErrorKind::Unavailable, StatusCode::SERVICE_UNAVAILABLE),
            (ApiErrorKind::Overloaded, StatusCode::SERVICE_UNAVAILABLE),
            (ApiErrorKind::Internal, StatusCode::INTERNAL_SERVER_ERROR),
        ] {
            assert_eq!(
                ApiError::new(kind, "x", None, None).status(),
                status,
                "{kind:?}"
            );
        }
    }

    #[tokio::test]
    async fn serializes_openai_error_envelope() {
        let response = openai_error_response(ApiError::new(
            ApiErrorKind::NotFound,
            "model `missing` was not found",
            Some("model_not_found"),
            Some("model"),
        ));
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["code"], "model_not_found");
        assert_eq!(body["error"]["param"], "model");
        assert_eq!(body["error"]["message"], "model `missing` was not found");
    }

    #[test]
    fn rate_limited_errors_ask_clients_to_retry() {
        let response = openai_error_response(ApiError::new(
            ApiErrorKind::RateLimited,
            "another LoRA adapter load is already in progress",
            Some("lora_load_busy"),
            None,
        ));
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(
            response.headers().get(axum::http::header::RETRY_AFTER),
            Some(&axum::http::HeaderValue::from_static(RETRY_AFTER_SECS))
        );
    }

    #[tokio::test]
    async fn replaces_server_error_details_with_stable_message() {
        let response = openai_error_response(ApiError::from_status(
            StatusCode::INTERNAL_SERVER_ERROR,
            "database password leaked",
        ));
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["error"]["message"], INTERNAL_ERROR_MESSAGE);
        assert_eq!(body["error"]["type"], "server_error");
        assert_eq!(body["error"]["code"], "internal_error");
    }
}
