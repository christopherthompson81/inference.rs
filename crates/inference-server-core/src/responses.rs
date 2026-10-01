//! The OpenResponses routes: HTTP framing over the engine's Responses operations.

use std::{pin::Pin, sync::Arc, task::Poll, time::Duration};

use axum::{
    Extension,
    extract::{Json, Path, State},
    response::{
        IntoResponse, Sse,
        sse::{Event, KeepAlive, KeepAliveStream},
    },
};
use futures::Stream;

use crate::handler_core::{ApiJson, ApiJsonRejection};
pub use crate::responses_api::{
    OpenResponsesCreateRequest, OpenResponsesStreamEvent, ResponseDeleted,
};
use crate::{
    agentic::AgenticDefaults,
    engine_chat::ChatEngine,
    handler_core::{ApiError, openai_error_response},
    responses_api::{
        OpenResponsesStreamer, ResponsesStreamItem, cancel_response as cancel, collect_response,
        delete_response as delete, get_response as get, prepare_response, spawn_background,
    },
    responses_types::resource::ResponseResource,
    skills::SkillStore,
    streaming::{StreamOutcomeHandle, get_keep_alive_interval},
    types::ExtractedInferenceRsState,
};

/// Frames the engine's Responses stream as Server-Sent Events, ending with `[DONE]`.
pub struct ResponsesSse {
    inner: OpenResponsesStreamer,
    done: bool,
}

impl Stream for ResponsesSse {
    type Item = Result<Event, axum::Error>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        if self.done {
            return Poll::Ready(None);
        }
        match Pin::new(&mut self.inner).poll_next(cx) {
            Poll::Ready(Some(item)) => {
                let event = Event::default().event(item.name());
                Poll::Ready(Some(match item {
                    ResponsesStreamItem::Event(event_data) => event.json_data(event_data),
                    ResponsesStreamItem::AgenticToolCallProgress(value)
                    | ResponsesStreamItem::AgenticToolApprovalRequired(value) => {
                        event.json_data(value)
                    }
                    ResponsesStreamItem::FileProduced(file) => event.json_data(file),
                }))
            }
            Poll::Ready(None) => {
                self.done = true;
                Poll::Ready(Some(Ok(Event::default().data("[DONE]"))))
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

pub enum OpenResponsesResponder {
    Sse(Box<Sse<KeepAliveStream<ResponsesSse>>>),
    Json(Box<ResponseResource>),
    Error(ApiError),
}

impl IntoResponse for OpenResponsesResponder {
    fn into_response(self) -> axum::response::Response {
        match self {
            Self::Sse(sse) => (*sse).into_response(),
            Self::Json(response) => Json(*response).into_response(),
            Self::Error(error) => openai_error_response(error),
        }
    }
}

/// Create response endpoint - OpenResponses API
#[cfg_attr(test, utoipa::path(
    post,
    tag = "inference.rs",
    path = "/v1/responses",
    request_body = OpenResponsesCreateRequest,
    responses((
        status = 200,
        description = "Response resource or server-sent response events",
        content(
            (ResponseResource = "application/json"),
            (OpenResponsesStreamEvent = "text/event-stream")
        )
    ))
))]
pub async fn create_response(
    State(state): ExtractedInferenceRsState,
    Extension(agentic_defaults): Extension<AgenticDefaults>,
    Extension(skill_store): Extension<Arc<SkillStore>>,
    stream_outcome: Option<Extension<StreamOutcomeHandle>>,
    payload: Result<ApiJson<OpenResponsesCreateRequest>, ApiJsonRejection>,
) -> OpenResponsesResponder {
    let request = match payload {
        Ok(ApiJson(request)) => request,
        Err(ApiJsonRejection(error)) => return OpenResponsesResponder::Error(error),
    };
    let chat = ChatEngine {
        state: state.clone(),
        agentic: agentic_defaults,
        skill_store: Some(skill_store),
    };
    let prepared = match prepare_response(&chat, request).await {
        Ok(prepared) => prepared,
        Err(error) => return OpenResponsesResponder::Error(error.into_api_error(state)),
    };
    if prepared.background {
        return OpenResponsesResponder::Json(Box::new(spawn_background(prepared, state)));
    }
    if prepared.stream {
        let tap = stream_outcome.map(|Extension(handle)| handle.tap());
        let sse = ResponsesSse {
            inner: OpenResponsesStreamer::new(prepared, state, tap),
            done: false,
        };
        return OpenResponsesResponder::Sse(Box::new(Sse::new(sse).keep_alive(
            KeepAlive::new().interval(Duration::from_millis(get_keep_alive_interval())),
        )));
    }
    match collect_response(prepared, &state).await {
        Ok(response) => OpenResponsesResponder::Json(Box::new(response)),
        Err(error) => OpenResponsesResponder::Error(error),
    }
}

fn resource_response(result: Result<ResponseResource, ApiError>) -> axum::response::Response {
    match result {
        Ok(response) => Json(response).into_response(),
        Err(error) => openai_error_response(error),
    }
}

/// Get response by ID endpoint
#[cfg_attr(test, utoipa::path(
    get,
    tag = "inference.rs",
    path = "/v1/responses/{response_id}",
    params(("response_id" = String, Path, description = "The ID of the response to retrieve")),
    responses((status = 200, description = "Response object", body = ResponseResource))
))]
pub async fn get_response(
    State(state): ExtractedInferenceRsState,
    Path(response_id): Path<String>,
) -> impl IntoResponse {
    resource_response(get(&state, &response_id))
}

/// Delete response by ID endpoint
#[cfg_attr(test, utoipa::path(
    delete,
    tag = "inference.rs",
    path = "/v1/responses/{response_id}",
    params(("response_id" = String, Path, description = "The ID of the response to delete")),
    responses((status = 200, description = "Response deleted", body = ResponseDeleted))
))]
pub async fn delete_response(
    State(state): ExtractedInferenceRsState,
    Path(response_id): Path<String>,
) -> impl IntoResponse {
    match delete(&state, &response_id) {
        Ok(deleted) => Json(deleted).into_response(),
        Err(error) => openai_error_response(error),
    }
}

/// Cancel response endpoint
#[cfg_attr(test, utoipa::path(
    post,
    tag = "inference.rs",
    path = "/v1/responses/{response_id}/cancel",
    params(("response_id" = String, Path, description = "The ID of the response to cancel")),
    responses((status = 200, description = "Response cancelled", body = ResponseResource))
))]
pub async fn cancel_response(
    State(state): ExtractedInferenceRsState,
    Path(response_id): Path<String>,
) -> impl IntoResponse {
    resource_response(cancel(&state, &response_id))
}

#[cfg(test)]
mod tests {
    use axum::http::StatusCode;
    use serde_json::{Value, json};

    use super::*;
    use crate::handler_core::ApiErrorKind;

    async fn body(responder: OpenResponsesResponder) -> (StatusCode, Value) {
        let response = responder.into_response();
        let status = response.status();
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        (status, serde_json::from_slice(&body).unwrap())
    }

    #[tokio::test]
    async fn typed_errors_keep_their_status() {
        let error = ApiError::new(
            ApiErrorKind::NotFound,
            "The requested model does not exist.",
            Some("model_not_found"),
            Some("model"),
        );
        let responder = OpenResponsesResponder::Error(error);

        assert_eq!(
            body(responder).await,
            (
                StatusCode::NOT_FOUND,
                json!({
                    "error": {
                        "message": "The requested model does not exist.",
                        "type": "invalid_request_error",
                        "param": "model",
                        "code": "model_not_found"
                    }
                })
            )
        );
    }

    #[tokio::test]
    async fn model_errors_hide_the_internal_message() {
        assert_eq!(
            body(OpenResponsesResponder::Error(ApiError::model_error())).await,
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                json!({
                    "error": {
                        "message": "The model failed to process the request.",
                        "type": "server_error",
                        "param": null,
                        "code": "model_error"
                    }
                })
            )
        );
    }
}
