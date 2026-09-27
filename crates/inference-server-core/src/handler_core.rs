//! Core functionality for handlers.

use axum::{
    extract::{rejection::JsonRejection, Json},
    http::StatusCode,
    response::IntoResponse,
};

pub(crate) use crate::api_error::{
    ApiError, ApiErrorKind, ModelErrorMessage, SERVICE_UNAVAILABLE_MESSAGE,
};
pub use crate::dispatch::{
    create_response_channel, send_request, send_request_with_model, DEFAULT_CHANNEL_BUFFER_SIZE,
};

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

/// HTTP views of an [`ApiError`]: the status it maps to, and errors that start as an HTTP status or body rejection.
pub(crate) trait ApiErrorHttp {
    fn from_status(status: StatusCode, message: impl Into<String>) -> Self;
    fn from_json_rejection(error: JsonRejection) -> Self;
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

    fn from_json_rejection(error: JsonRejection) -> Self {
        let status = error.status();
        let code = match status {
            StatusCode::PAYLOAD_TOO_LARGE => "request_body_too_large",
            StatusCode::UNSUPPORTED_MEDIA_TYPE => "invalid_content_type",
            StatusCode::UNPROCESSABLE_ENTITY => "invalid_request_body",
            _ => "malformed_json",
        };
        let kind = match status {
            StatusCode::PAYLOAD_TOO_LARGE => ApiErrorKind::PayloadTooLarge,
            StatusCode::UNSUPPORTED_MEDIA_TYPE => ApiErrorKind::UnsupportedMediaType,
            _ => ApiErrorKind::InvalidRequest,
        };
        Self::new(kind, error.body_text(), Some(code), None)
    }

    fn status(&self) -> StatusCode {
        match self.kind {
            ApiErrorKind::InvalidRequest => StatusCode::BAD_REQUEST,
            ApiErrorKind::NotFound => StatusCode::NOT_FOUND,
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

    #[test]
    fn every_error_kind_maps_to_its_status() {
        for (kind, status) in [
            (ApiErrorKind::InvalidRequest, StatusCode::BAD_REQUEST),
            (ApiErrorKind::NotFound, StatusCode::NOT_FOUND),
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
