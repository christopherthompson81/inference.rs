//! Routes over [`ModelManager`]: the inventory, loads with their progress, unloads, downloads and deletions, which are
//! an admin's, and the served models and layout detection, which are anyone's.

use std::convert::Infallible;
use std::sync::Arc;

use axum::{
    Extension,
    response::{
        IntoResponse, Response,
        sse::{Event, KeepAlive, Sse},
    },
};
use futures::{Stream, StreamExt};
use inference_api::engine::ProgressEvent;
use inference_api::model_manager::{
    LayoutDetectRequest, ManagedDeleteRequest, ManagedDownloadRequest, ManagedLoadRequest,
    ModelManager,
};
use inference_api::models::ModelOperationRequest;
use serde_json::json;
use tokio::sync::mpsc;

use crate::handler_core::{ApiJson, ApiJsonRejection, json_response, openai_error_response};

/// The SSE event carrying a load's progress: `{"stage", "done", "total", "unit"}`.
pub const PROGRESS_EVENT: &str = "progress";
/// The SSE event ending a load that succeeded, carrying the loaded model.
pub const LOADED_EVENT: &str = "loaded";
/// The SSE event ending a load or download that failed, carrying the error envelope.
pub const ERROR_EVENT: &str = "error";

fn sse(events: impl Stream<Item = Event> + Send + 'static) -> Response {
    Sse::new(events.map(Ok::<_, Infallible>))
        .keep_alive(KeepAlive::default())
        .into_response()
}

fn event(name: &str, data: &serde_json::Value) -> Event {
    Event::default().event(name).data(data.to_string())
}

#[cfg_attr(test, utoipa::path(
    get,
    tag = "inference.rs",
    path = "/v1/manager/inventory",
    description = "An admin's: the models in the server's model folders (`--model-folder`) and its Hugging Face cache.",
    responses(
        (status = 200, description = "The models found, the cache's repositories, and folders that could not be read"),
        (status = 403, description = "Not an admin"),
    )
))]
pub async fn inventory(Extension(manager): Extension<Arc<ModelManager>>) -> Response {
    match tokio::task::spawn_blocking(move || manager.inventory()).await {
        Ok(scan) => axum::Json(scan).into_response(),
        Err(_) => openai_error_response(inference_api::api_error::ApiError::internal()),
    }
}

#[cfg_attr(test, utoipa::path(
    get,
    tag = "inference.rs",
    path = "/v1/manager/models",
    responses((status = 200, description = "The served models: their kind, status, and where each was loaded from",
        body = inference_api::model_manager::ManagedModels))
))]
pub async fn managed_models(Extension(manager): Extension<Arc<ModelManager>>) -> Response {
    json_response(manager.models())
}

#[cfg_attr(test, utoipa::path(
    post,
    tag = "inference.rs",
    path = "/v1/manager/load",
    description = "An admin's. Loads an inventory entry or a Hugging Face repository as `model_id`, unloading what held \
        that id. The response is an event stream: `progress` events, then `loaded` (the model) or `error`.",
    request_body = ManagedLoadRequest,
    responses(
        (status = 200, description = "`text/event-stream` of the load's progress and outcome"),
        (status = 400, description = "Invalid request"),
        (status = 403, description = "Not an admin"),
    )
))]
pub async fn load(
    Extension(manager): Extension<Arc<ModelManager>>,
    payload: Result<ApiJson<ManagedLoadRequest>, ApiJsonRejection>,
) -> Response {
    let request = match payload {
        Ok(ApiJson(request)) => request,
        Err(ApiJsonRejection(error)) => return openai_error_response(error),
    };
    let (events, receiver) = mpsc::unbounded_channel();
    let progress = events.clone();
    let sink = Arc::new(move |report: &ProgressEvent<'_>| {
        let _ = progress.send(event(PROGRESS_EVENT, &json!(report)));
    });
    // a client that goes away leaves the load running: it ends with the model loaded or the error logged
    tokio::spawn(async move {
        let last = match manager.load(request, sink).await {
            Ok(model) => event(LOADED_EVENT, &json!(model)),
            Err(error) => {
                tracing::warn!("model load failed: {}", error.message);
                event(ERROR_EVENT, &error.to_openai_body())
            }
        };
        let _ = events.send(last);
    });
    sse(futures::stream::unfold(
        receiver,
        |mut receiver| async move { receiver.recv().await.map(|event| (event, receiver)) },
    ))
}

#[cfg_attr(test, utoipa::path(
    post,
    tag = "inference.rs",
    path = "/v1/manager/unload",
    description = "An admin's: stops serving a model, or a layout model, and frees it.",
    request_body = ModelOperationRequest,
    responses(
        (status = 204, description = "Unloaded"),
        (status = 403, description = "Not an admin"),
        (status = 404, description = "Not served"),
    )
))]
pub async fn unload(
    Extension(manager): Extension<Arc<ModelManager>>,
    payload: Result<ApiJson<ModelOperationRequest>, ApiJsonRejection>,
) -> Response {
    let request = match payload {
        Ok(ApiJson(request)) => request,
        Err(ApiJsonRejection(error)) => return openai_error_response(error),
    };
    match manager.unload(request).await {
        Ok(()) => axum::http::StatusCode::NO_CONTENT.into_response(),
        Err(error) => openai_error_response(error),
    }
}

#[cfg_attr(test, utoipa::path(
    post,
    tag = "inference.rs",
    path = "/v1/manager/download",
    description = "An admin's. Downloads from the Hub into the server's cache, with the server's token. The response \
        is an event stream: `files`, `progress` (`{\"file\", \"done\", \"total\"}` in bytes), then `done`, `canceled` \
        or `error`. Closing it stops the download.",
    request_body = ManagedDownloadRequest,
    responses(
        (status = 200, description = "`text/event-stream` of the download"),
        (status = 403, description = "Not an admin"),
    )
))]
pub async fn download(
    Extension(manager): Extension<Arc<ModelManager>>,
    payload: Result<ApiJson<ManagedDownloadRequest>, ApiJsonRejection>,
) -> Response {
    let request = match payload {
        Ok(ApiJson(request)) => request,
        Err(ApiJsonRejection(error)) => return openai_error_response(error),
    };
    let events = match manager.download(request) {
        Ok(events) => events,
        Err(error) => return openai_error_response(error),
    };
    // each item is `{"event", "data"}`, as the C ABI's download stream gives it
    sse(events.map(|item| {
        let item: serde_json::Value = serde_json::from_str(&item).unwrap_or_default();
        let name = item["event"].as_str().unwrap_or(ERROR_EVENT).to_string();
        event(&name, &item["data"])
    }))
}

#[cfg_attr(test, utoipa::path(
    post,
    tag = "inference.rs",
    path = "/v1/manager/delete",
    description = "An admin's: deletes a repository from the server's Hugging Face cache, or a GGUF from a model folder; \
        never a loaded model.",
    request_body = ManagedDeleteRequest,
    responses(
        (status = 200, description = "Deleted", body = inference_api::model_manager::ManagedDeletion),
        (status = 403, description = "Not an admin"),
        (status = 409, description = "The model is loaded"),
    )
))]
pub async fn delete(
    Extension(manager): Extension<Arc<ModelManager>>,
    payload: Result<ApiJson<ManagedDeleteRequest>, ApiJsonRejection>,
) -> Response {
    match payload {
        Ok(ApiJson(request)) => {
            let deleted = tokio::task::spawn_blocking(move || manager.delete(request)).await;
            match deleted {
                Ok(result) => json_response(result),
                Err(_) => openai_error_response(inference_api::api_error::ApiError::internal()),
            }
        }
        Err(ApiJsonRejection(error)) => openai_error_response(error),
    }
}

#[cfg_attr(test, utoipa::path(
    post,
    tag = "inference.rs",
    path = "/v1/layout/detect",
    description = "The layout regions of a page, in reading order, from a layout model `/v1/manager/load` loaded.",
    request_body = LayoutDetectRequest,
    responses(
        (status = 200, description = "The page's regions", body = inference_api::model_manager::LayoutDetectResponse),
        (status = 404, description = "No such layout model"),
    )
))]
pub async fn detect_layout(
    Extension(manager): Extension<Arc<ModelManager>>,
    payload: Result<ApiJson<LayoutDetectRequest>, ApiJsonRejection>,
) -> Response {
    match payload {
        Ok(ApiJson(request)) => json_response(manager.detect_layout(request).await),
        Err(ApiJsonRejection(error)) => openai_error_response(error),
    }
}
