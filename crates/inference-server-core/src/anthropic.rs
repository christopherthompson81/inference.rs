//! Anthropic-compatible Messages API over HTTP; the operation itself lives in `inference_api::anthropic`.

use std::{pin::Pin, sync::Arc, task::Poll, time::Duration};

use axum::{
    Extension,
    extract::{Json, State},
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
use crate::anthropic_api::{
    AnthropicStream, AnthropicStreamEvent, MessagesFailure, anthropic_error_body, collect_messages,
    count_tokens, prepare_messages,
};
use crate::handler_core::{ApiJson, ApiJsonRejection};
use crate::{
    agentic::AgenticDefaults,
    engine_chat::{ChatEngine, DispatchError},
    handler_core::{ApiError, ApiErrorKind, ResponseErrorMessage},
    skills::SkillStore,
    streaming::{StreamOutcomeHandle, get_keep_alive_interval},
    types::ExtractedInferenceRsState,
};

type BoxError = Box<dyn std::error::Error + Send + Sync + 'static>;

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
    InternalError(BoxError),
    ValidationError(BoxError),
    ModelError(String),
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
            AnthropicMessagesResponder::InternalError(e) => {
                anthropic_error_response(ApiError::from_error(e.as_ref(), ApiErrorKind::Internal))
            }
            AnthropicMessagesResponder::ValidationError(e) => anthropic_error_response(
                ApiError::from_error(e.as_ref(), ApiErrorKind::InvalidRequest),
            ),
            AnthropicMessagesResponder::ModelError(_) => {
                anthropic_error_response(ApiError::model_error())
            }
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
    State(state): ExtractedInferenceRsState,
    Extension(agentic_defaults): Extension<AgenticDefaults>,
    Extension(skill_store): Extension<Arc<SkillStore>>,
    stream_outcome: Option<Extension<StreamOutcomeHandle>>,
    payload: Result<ApiJson<AnthropicMessagesRequest>, ApiJsonRejection>,
) -> AnthropicMessagesResponder {
    let request = match payload {
        Ok(ApiJson(request)) => request,
        Err(ApiJsonRejection(error)) => {
            return AnthropicMessagesResponder::ValidationError(Box::new(
                anthropic_json_rejection(error),
            ));
        }
    };
    let engine = ChatEngine {
        state: state.clone(),
        agentic: agentic_defaults,
        skill_store: Some(skill_store),
    };
    let prepared = match prepare_messages(&engine, request).await {
        Ok(prepared) => prepared,
        Err(DispatchError::Validation(e)) => return AnthropicMessagesResponder::ValidationError(e),
        Err(DispatchError::Internal(e)) => return AnthropicMessagesResponder::InternalError(e),
    };
    if prepared.chat.is_streaming {
        let tap = stream_outcome.map(|Extension(handle)| handle.tap());
        let streamer = AnthropicStreamer::new(AnthropicStream::new(prepared, state, tap));
        AnthropicMessagesResponder::Sse(Sse::new(streamer).keep_alive(
            KeepAlive::new().interval(Duration::from_millis(get_keep_alive_interval())),
        ))
    } else {
        let (omit_thinking, override_) = (prepared.omit_thinking, prepared.chat.model_override);
        let mut rx = prepared.chat.rx;
        match collect_messages(&mut rx, state, override_.as_deref(), omit_thinking).await {
            Ok(response) => AnthropicMessagesResponder::Json(response),
            Err(MessagesFailure::Validation(e)) => AnthropicMessagesResponder::ValidationError(e),
            Err(MessagesFailure::Internal(e)) => AnthropicMessagesResponder::InternalError(e),
            Err(MessagesFailure::Model(msg)) => AnthropicMessagesResponder::ModelError(msg),
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
    State(state): ExtractedInferenceRsState,
    payload: Result<ApiJson<AnthropicMessagesRequest>, ApiJsonRejection>,
) -> AnthropicCountTokensResponder {
    let request = match payload {
        Ok(ApiJson(request)) => request,
        Err(ApiJsonRejection(error)) => {
            return AnthropicCountTokensResponder::Error(anthropic_json_rejection(error));
        }
    };
    match count_tokens(&state, request).await {
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
        let responses = [
            AnthropicMessagesResponder::ValidationError(validation.into()),
            AnthropicMessagesResponder::InternalError(Box::new(InferenceRsError::ModelNotFound(
                "missing-model".to_string(),
            ))),
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
        let response = AnthropicMessagesResponder::ValidationError(Box::new(
            InferenceRsError::ModelReloading("model".to_string()),
        ))
        .into_response();

        assert_eq!(response.status(), http::StatusCode::CONFLICT);
        let body = error_body(response).await;
        assert_eq!(body["error"]["type"], "conflict_error");
        assert_eq!(body["error"]["message"], "model `model` is being reloaded");
    }

    #[tokio::test]
    async fn internal_and_model_errors_do_not_expose_details() {
        let response = AnthropicMessagesResponder::InternalError(Box::new(std::io::Error::other(
            "private internal detail",
        )))
        .into_response();
        assert_eq!(response.status(), http::StatusCode::INTERNAL_SERVER_ERROR);
        let body = error_body(response).await;
        assert_eq!(body["error"]["message"], INTERNAL_ERROR_MESSAGE);
        assert!(!body.to_string().contains("private internal detail"));

        let response = AnthropicMessagesResponder::ModelError("private model detail".to_string())
            .into_response();
        let body = error_body(response).await;
        assert_eq!(
            body["error"]["message"],
            crate::api_error::MODEL_ERROR_MESSAGE
        );
        assert!(!body.to_string().contains("private model detail"));
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
