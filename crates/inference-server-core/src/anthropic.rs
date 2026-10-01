//! Anthropic-compatible Messages API over HTTP; the operation itself lives in `inference_api::anthropic`.

use std::{pin::Pin, task::Poll, time::Duration};

use axum::{
    Extension,
    extract::Json,
    http,
    response::{
        IntoResponse, Sse,
        sse::{Event, KeepAlive, KeepAliveStream},
    },
};
use futures::Stream;
use tokio::time::{Instant, Interval, MissedTickBehavior, interval_at};

pub use crate::anthropic_api::{
    AnthropicContainer, AnthropicContentBlock, AnthropicCountTokensResponse, AnthropicError,
    AnthropicErrorBody, AnthropicImageSource, AnthropicJsonOutputFormat, AnthropicMessage,
    AnthropicMessageContent, AnthropicMessageResponse, AnthropicMessagesRequest,
    AnthropicOutputConfig, AnthropicResponseContentBlock, AnthropicSkillReference, AnthropicSystem,
    AnthropicThinking, AnthropicTool, AnthropicToolChoice, AnthropicUsage,
    AnthropicWebSearchUserLocation,
};
use crate::anthropic_api::{AnthropicStream, AnthropicStreamEvent, anthropic_error_body};
use crate::handler_core::{ApiJson, ApiJsonRejection};
use crate::{
    handler_core::{ApiError, ApiErrorKind, ResponseErrorMessage},
    streaming::{StreamOutcomeHandle, get_keep_alive_interval},
    types::OwnedEngine,
};

const ANTHROPIC_OVERLOADED_STATUS: u16 = 529;

/// Frames the engine's Anthropic stream as SSE, adding `ping` events while the model is quiet.
pub struct AnthropicStreamer {
    inner: AnthropicStream,
    ping: Interval,
}

impl AnthropicStreamer {
    fn new(inner: AnthropicStream) -> Self {
        let ping_interval = Duration::from_millis(get_keep_alive_interval());
        let mut ping = interval_at(Instant::now() + ping_interval, ping_interval);
        ping.set_missed_tick_behavior(MissedTickBehavior::Delay);
        Self { inner, ping }
    }
}

impl Stream for AnthropicStreamer {
    type Item = Result<Event, axum::Error>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        let event = match Pin::new(&mut self.inner).poll_next(cx) {
            Poll::Ready(Some(event)) => event,
            Poll::Ready(None) => return Poll::Ready(None),
            Poll::Pending => {
                if Pin::new(&mut self.ping).poll_tick(cx).is_ready() {
                    AnthropicStreamEvent {
                        name: "ping",
                        payload: serde_json::json!({"type": "ping"}),
                    }
                } else {
                    return Poll::Pending;
                }
            }
        };
        Poll::Ready(Some(
            Event::default().event(event.name).json_data(event.payload),
        ))
    }
}

pub type AnthropicMessagesSse = Sse<KeepAliveStream<AnthropicStreamer>>;

pub enum AnthropicMessagesResponder {
    Sse(AnthropicMessagesSse),
    Json(AnthropicMessageResponse),
    Error(ApiError),
}

pub enum AnthropicCountTokensResponder {
    Json(AnthropicCountTokensResponse),
    Error(ApiError),
}

impl IntoResponse for AnthropicMessagesResponder {
    fn into_response(self) -> axum::response::Response {
        match self {
            AnthropicMessagesResponder::Sse(s) => s.into_response(),
            AnthropicMessagesResponder::Json(s) => Json(s).into_response(),
            AnthropicMessagesResponder::Error(error) => anthropic_error_response(error),
        }
    }
}

impl IntoResponse for AnthropicCountTokensResponder {
    fn into_response(self) -> axum::response::Response {
        match self {
            AnthropicCountTokensResponder::Json(s) => Json(s).into_response(),
            AnthropicCountTokensResponder::Error(error) => anthropic_error_response(error),
        }
    }
}

fn anthropic_error_status(kind: ApiErrorKind) -> http::StatusCode {
    match kind {
        ApiErrorKind::InvalidRequest | ApiErrorKind::UnsupportedMediaType => {
            http::StatusCode::BAD_REQUEST
        }
        ApiErrorKind::NotFound => http::StatusCode::NOT_FOUND,
        ApiErrorKind::Gone => http::StatusCode::GONE,
        ApiErrorKind::Unauthorized => http::StatusCode::UNAUTHORIZED,
        ApiErrorKind::Forbidden => http::StatusCode::FORBIDDEN,
        ApiErrorKind::Conflict => http::StatusCode::CONFLICT,
        ApiErrorKind::PayloadTooLarge => http::StatusCode::PAYLOAD_TOO_LARGE,
        ApiErrorKind::RateLimited => http::StatusCode::TOO_MANY_REQUESTS,
        ApiErrorKind::Unavailable | ApiErrorKind::Internal => {
            http::StatusCode::INTERNAL_SERVER_ERROR
        }
        ApiErrorKind::Overloaded => http::StatusCode::from_u16(ANTHROPIC_OVERLOADED_STATUS)
            .expect("Anthropic overloaded status must be valid"),
    }
}

fn anthropic_json_rejection(mut error: ApiError) -> ApiError {
    if error.kind == ApiErrorKind::UnsupportedMediaType {
        error.kind = ApiErrorKind::InvalidRequest;
    }
    error
}

pub(crate) fn anthropic_error_response(error: ApiError) -> axum::response::Response {
    let status = anthropic_error_status(error.kind);
    let mut response = (status, Json(anthropic_error_body(&error))).into_response();
    response
        .extensions_mut()
        .insert(ResponseErrorMessage(error.message));
    response
}

#[cfg_attr(test, utoipa::path(
    post,
    tag = "inference.rs",
    path = "/v1/messages",
    request_body = AnthropicMessagesRequest,
    responses((status = 200, description = "Anthropic messages", body = AnthropicMessageResponse))
))]
pub async fn anthropic_messages(
    OwnedEngine(engine): OwnedEngine,
    stream_outcome: Option<Extension<StreamOutcomeHandle>>,
    payload: Result<ApiJson<AnthropicMessagesRequest>, ApiJsonRejection>,
) -> AnthropicMessagesResponder {
    let request = match payload {
        Ok(ApiJson(request)) => request,
        Err(ApiJsonRejection(error)) => {
            return AnthropicMessagesResponder::Error(anthropic_json_rejection(error));
        }
    };
    if request.stream.unwrap_or(false) {
        let tap = stream_outcome.map(|Extension(handle)| handle.tap());
        match engine.anthropic_messages_stream(request).await {
            Ok(stream) => {
                let streamer = AnthropicStreamer::new(stream.with_tap(tap));
                AnthropicMessagesResponder::Sse(Sse::new(streamer).keep_alive(
                    KeepAlive::new().interval(Duration::from_millis(get_keep_alive_interval())),
                ))
            }
            Err(error) => AnthropicMessagesResponder::Error(error),
        }
    } else {
        match engine.anthropic_messages(request).await {
            Ok(response) => AnthropicMessagesResponder::Json(response),
            Err(error) => AnthropicMessagesResponder::Error(error),
        }
    }
}

#[cfg_attr(test, utoipa::path(
    post,
    tag = "inference.rs",
    path = "/v1/messages/count_tokens",
    request_body = AnthropicMessagesRequest,
    responses((status = 200, description = "Anthropic message token count", body = AnthropicCountTokensResponse))
))]
pub async fn anthropic_count_tokens(
    OwnedEngine(engine): OwnedEngine,
    payload: Result<ApiJson<AnthropicMessagesRequest>, ApiJsonRejection>,
) -> AnthropicCountTokensResponder {
    let request = match payload {
        Ok(ApiJson(request)) => request,
        Err(ApiJsonRejection(error)) => {
            return AnthropicCountTokensResponder::Error(anthropic_json_rejection(error));
        }
    };
    match engine.count_tokens(request).await {
        Ok(count) => AnthropicCountTokensResponder::Json(count),
        Err(error) => AnthropicCountTokensResponder::Error(error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api_error::INTERNAL_ERROR_MESSAGE;
    use axum::{
        body::{Body, to_bytes},
        extract::FromRequest,
        http::Request as HttpRequest,
    };
    use inference_core::InferenceRsError;
    use serde_json::Value;

    async fn error_body(response: axum::response::Response) -> Value {
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        serde_json::from_slice(&body).unwrap()
    }

    #[tokio::test]
    async fn maps_anthropic_error_statuses_and_types() {
        let cases = [
            (
                ApiErrorKind::InvalidRequest,
                http::StatusCode::BAD_REQUEST,
                "invalid_request_error",
            ),
            (
                ApiErrorKind::UnsupportedMediaType,
                http::StatusCode::BAD_REQUEST,
                "invalid_request_error",
            ),
            (
                ApiErrorKind::NotFound,
                http::StatusCode::NOT_FOUND,
                "not_found_error",
            ),
            (
                ApiErrorKind::Forbidden,
                http::StatusCode::FORBIDDEN,
                "permission_error",
            ),
            (
                ApiErrorKind::Conflict,
                http::StatusCode::CONFLICT,
                "conflict_error",
            ),
            (
                ApiErrorKind::PayloadTooLarge,
                http::StatusCode::PAYLOAD_TOO_LARGE,
                "request_too_large",
            ),
            (
                ApiErrorKind::RateLimited,
                http::StatusCode::TOO_MANY_REQUESTS,
                "rate_limit_error",
            ),
            (
                ApiErrorKind::Unavailable,
                http::StatusCode::INTERNAL_SERVER_ERROR,
                "api_error",
            ),
            (
                ApiErrorKind::Overloaded,
                http::StatusCode::from_u16(ANTHROPIC_OVERLOADED_STATUS).unwrap(),
                "overloaded_error",
            ),
            (
                ApiErrorKind::Internal,
                http::StatusCode::INTERNAL_SERVER_ERROR,
                "api_error",
            ),
        ];

        for (kind, status, error_type) in cases {
            let response = anthropic_error_response(ApiError::new(kind, "message", None, None));
            assert_eq!(response.status(), status);
            let body = error_body(response).await;
            assert_eq!(body["type"], "error");
            assert_eq!(body["error"]["type"], error_type);
        }
    }

    #[tokio::test]
    async fn typed_model_not_found_is_a_404_from_any_responder() {
        let validation =
            anyhow::Error::new(InferenceRsError::ModelNotFound("missing-model".to_string()))
                .context("request validation failed");
        let internal = InferenceRsError::ModelNotFound("missing-model".to_string());
        let responses = [
            AnthropicMessagesResponder::Error(ApiError::from_error(
                validation.as_ref(),
                ApiErrorKind::InvalidRequest,
            )),
            AnthropicMessagesResponder::Error(ApiError::from_error(
                &internal,
                ApiErrorKind::Internal,
            )),
        ];

        for response in responses {
            let response = response.into_response();
            assert_eq!(response.status(), http::StatusCode::NOT_FOUND);
            let body = error_body(response).await;
            assert_eq!(body["error"]["type"], "not_found_error");
        }
    }

    #[tokio::test]
    async fn reloading_model_uses_anthropic_conflict_error() {
        let reloading = InferenceRsError::ModelReloading("model".to_string());
        let response = AnthropicMessagesResponder::Error(ApiError::from_error(
            &reloading,
            ApiErrorKind::InvalidRequest,
        ))
        .into_response();

        assert_eq!(response.status(), http::StatusCode::CONFLICT);
        let body = error_body(response).await;
        assert_eq!(body["error"]["type"], "conflict_error");
        assert_eq!(body["error"]["message"], "model `model` is being reloaded");
    }

    #[tokio::test]
    async fn internal_and_model_errors_do_not_expose_details() {
        let internal = std::io::Error::other("private internal detail");
        let response = AnthropicMessagesResponder::Error(ApiError::from_error(
            &internal,
            ApiErrorKind::Internal,
        ))
        .into_response();
        assert_eq!(response.status(), http::StatusCode::INTERNAL_SERVER_ERROR);
        let body = error_body(response).await;
        assert_eq!(body["error"]["message"], INTERNAL_ERROR_MESSAGE);
        assert!(!body.to_string().contains("private internal detail"));

        let response = AnthropicMessagesResponder::Error(ApiError::model_error()).into_response();
        let body = error_body(response).await;
        assert_eq!(
            body["error"]["message"],
            crate::api_error::MODEL_ERROR_MESSAGE
        );
    }

    #[tokio::test]
    async fn json_rejections_use_anthropic_errors() {
        let malformed = HttpRequest::builder()
            .header(http::header::CONTENT_TYPE, "application/json")
            .body(Body::from("{"))
            .unwrap();
        let rejection = ApiJson::<AnthropicMessagesRequest>::from_request(malformed, &())
            .await
            .unwrap_err();
        let response = anthropic_error_response(anthropic_json_rejection(rejection.0));
        assert_eq!(response.status(), http::StatusCode::BAD_REQUEST);
        let body = error_body(response).await;
        assert_eq!(body["error"]["type"], "invalid_request_error");

        let wrong_type = HttpRequest::builder()
            .header(http::header::CONTENT_TYPE, "application/json")
            .body(Body::from(r#"{"messages":"not-an-array"}"#))
            .unwrap();
        let rejection = ApiJson::<AnthropicMessagesRequest>::from_request(wrong_type, &())
            .await
            .unwrap_err();
        let response = anthropic_error_response(anthropic_json_rejection(rejection.0));
        assert_eq!(response.status(), http::StatusCode::BAD_REQUEST);
        let body = error_body(response).await;
        assert_eq!(body["error"]["type"], "invalid_request_error");

        let missing_content_type = HttpRequest::builder()
            .body(Body::from(r#"{"messages":[]}"#))
            .unwrap();
        let rejection =
            ApiJson::<AnthropicMessagesRequest>::from_request(missing_content_type, &())
                .await
                .unwrap_err();
        let response = anthropic_error_response(anthropic_json_rejection(rejection.0));
        assert_eq!(response.status(), http::StatusCode::BAD_REQUEST);
        let body = error_body(response).await;
        assert_eq!(body["error"]["type"], "invalid_request_error");
    }
}
