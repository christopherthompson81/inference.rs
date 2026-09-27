//! Core functionality for handlers.

use axum::{
    extract::{rejection::JsonRejection, Json},
    http::StatusCode,
    response::IntoResponse,
};
use inference_core::Response;
use tokio::sync::mpsc::Receiver;

pub(crate) use crate::api_error::{
    ApiError, ApiErrorKind, ModelErrorMessage, INTERNAL_ERROR_MESSAGE, SERVICE_UNAVAILABLE_MESSAGE,
};
pub(crate) use crate::dispatch::{apply_model_override, request_model_override};
pub use crate::dispatch::{
    create_response_channel, send_request, send_request_with_model, DEFAULT_CHANNEL_BUFFER_SIZE,
};
use crate::types::SharedInferenceRsState;

/// Error message attached to a failed response so the access log can report it.
#[derive(Clone, Debug)]
pub struct ResponseErrorMessage(pub String);

pub(crate) fn openai_error_response(error: ApiError) -> axum::response::Response {
    let mut response = Json(error.to_openai_body()).into_response();
    *response.status_mut() = error.status();
    response
        .extensions_mut()
        .insert(ResponseErrorMessage(error.message));
    response
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
            ApiErrorKind::Conflict => StatusCode::CONFLICT,
            ApiErrorKind::PayloadTooLarge => StatusCode::PAYLOAD_TOO_LARGE,
            ApiErrorKind::UnsupportedMediaType => StatusCode::UNSUPPORTED_MEDIA_TYPE,
            ApiErrorKind::RateLimited => StatusCode::TOO_MANY_REQUESTS,
            ApiErrorKind::Unavailable | ApiErrorKind::Overloaded => StatusCode::SERVICE_UNAVAILABLE,
            ApiErrorKind::Internal => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}

/// Generic function to process non-streaming responses.
pub(crate) async fn base_process_non_streaming_response<R, M, E>(
    rx: &mut Receiver<Response>,
    state: SharedInferenceRsState,
    match_fn: M,
    error_handler: E,
) -> R
where
    M: FnOnce(SharedInferenceRsState, Response) -> R,
    E: FnOnce(SharedInferenceRsState, Box<dyn std::error::Error + Send + Sync + 'static>) -> R,
{
    loop {
        match rx.recv().await {
            Some(Response::AgenticToolCallProgress { .. }) => continue,
            Some(Response::BlockDenoisingProgress(_)) => continue,
            Some(Response::AgenticToolApprovalRequired { .. }) => continue,
            Some(Response::File(_)) => continue,
            Some(response) => return match_fn(state, response),
            None => {
                let error = anyhow::Error::msg("No response received from the model.");
                return error_handler(state, error.into());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::to_bytes;

    #[test]
    fn every_error_kind_maps_to_its_status() {
        for (kind, status) in [
            (ApiErrorKind::InvalidRequest, StatusCode::BAD_REQUEST),
            (ApiErrorKind::NotFound, StatusCode::NOT_FOUND),
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
