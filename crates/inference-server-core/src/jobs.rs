//! Jobs over HTTP: submit a world generation or reconstruction with its files, follow its stages as server-sent events,
//! and fetch its previews and outputs.

use std::convert::Infallible;

use axum::{
    extract::{Multipart, Path, multipart::MultipartRejection},
    http::{StatusCode, header},
    response::{
        IntoResponse, Response,
        sse::{Event, KeepAlive, Sse},
    },
};
use futures::stream::{self, Stream, StreamExt};
use inference_api::jobs::{JobEvent, JobInput, JobRequest};
use tokio::sync::broadcast::error::RecvError;
use tower::ServiceExt;

use crate::{
    handler_core::{ApiError, ApiErrorHttp, ApiErrorKind, json_response, openai_error_response},
    types::OwnedEngine,
};

const REQUEST_FIELD: &str = "request";
const FILE_FIELD: &str = "file";
// what a job file is served as, by extension; never markup or script
const FILE_TYPES: &[(&str, &str)] = &[
    ("png", "image/png"),
    ("jpg", "image/jpeg"),
    ("jpeg", "image/jpeg"),
    ("webp", "image/webp"),
    ("json", "application/json"),
    ("ply", "application/octet-stream"),
    ("spz", "application/octet-stream"),
    ("npy", "application/octet-stream"),
];

fn multipart_error(error: axum::extract::multipart::MultipartError) -> ApiError {
    ApiError::from_status(error.status(), error.body_text())
}

#[cfg_attr(test, utoipa::path(
    post,
    tag = "inference.rs",
    path = "/v1/jobs",
    request_body(content_type = "multipart/form-data", description = "`request`: the job as JSON; `file`: its inputs, repeated"),
    responses(
        (status = 200, description = "The queued job", body = inference_api::jobs::JobRecord),
        (status = 400, description = "Invalid job, or what it needs is not configured"),
    )
))]
pub async fn submit_job(
    OwnedEngine(engine): OwnedEngine,
    payload: Result<Multipart, MultipartRejection>,
) -> Response {
    let mut multipart = match payload {
        Ok(multipart) => multipart,
        Err(error) => {
            return openai_error_response(ApiError::from_status(error.status(), error.body_text()));
        }
    };
    let parsed = async {
        let mut request = None;
        let mut inputs = Vec::new();
        while let Some(field) = multipart.next_field().await.map_err(multipart_error)? {
            match field.name().unwrap_or_default() {
                REQUEST_FIELD => {
                    let text = field.text().await.map_err(multipart_error)?;
                    request = Some(serde_json::from_str::<JobRequest>(&text).map_err(|error| {
                        ApiError::new(
                            ApiErrorKind::InvalidRequest,
                            format!("`request` is not a job: {error}"),
                            Some("invalid_job"),
                            Some(REQUEST_FIELD),
                        )
                    })?);
                }
                FILE_FIELD => {
                    let name = field.file_name().unwrap_or_default().to_string();
                    let data = field.bytes().await.map_err(multipart_error)?.to_vec();
                    inputs.push(JobInput::Bytes { name, data });
                }
                _ => {}
            }
        }
        let request = request.ok_or_else(|| {
            ApiError::new(
                ApiErrorKind::InvalidRequest,
                "A job needs a multipart field `request` holding it as JSON.",
                Some("missing_required_parameter"),
                Some(REQUEST_FIELD),
            )
        })?;
        Ok::<_, ApiError>((request, inputs))
    }
    .await;
    match parsed {
        Ok((request, inputs)) => json_response(engine.submit_job(request, inputs)),
        Err(error) => openai_error_response(error),
    }
}

#[cfg_attr(test, utoipa::path(
    get,
    tag = "inference.rs",
    path = "/v1/jobs",
    responses((status = 200, description = "This owner's jobs, newest first", body = inference_api::jobs::JobList))
))]
pub async fn list_jobs(OwnedEngine(engine): OwnedEngine) -> Response {
    json_response(Ok::<_, ApiError>(engine.jobs()))
}

#[cfg_attr(test, utoipa::path(
    get,
    tag = "inference.rs",
    path = "/v1/jobs/{id}",
    params(("id" = String, Path, description = "Job ID")),
    responses(
        (status = 200, description = "The job", body = inference_api::jobs::JobRecord),
        (status = 404, description = "No such job"),
    )
))]
pub async fn get_job(OwnedEngine(engine): OwnedEngine, Path(id): Path<String>) -> Response {
    json_response(engine.job(&id))
}

#[cfg_attr(test, utoipa::path(
    delete,
    tag = "inference.rs",
    path = "/v1/jobs/{id}",
    params(("id" = String, Path, description = "Job ID")),
    responses(
        (status = 204, description = "The job and its folder removed"),
        (status = 404, description = "No such job"),
        (status = 409, description = "The job has not finished"),
    )
))]
pub async fn delete_job(OwnedEngine(engine): OwnedEngine, Path(id): Path<String>) -> Response {
    match engine.delete_job(&id) {
        Ok(_) => StatusCode::NO_CONTENT.into_response(),
        Err(error) => openai_error_response(error),
    }
}

#[cfg_attr(test, utoipa::path(
    post,
    tag = "inference.rs",
    path = "/v1/jobs/{id}/cancel",
    params(("id" = String, Path, description = "Job ID")),
    responses(
        (status = 200, description = "The job, asked to stop", body = inference_api::jobs::JobRecord),
        (status = 404, description = "No such job"),
    )
))]
pub async fn cancel_job(OwnedEngine(engine): OwnedEngine, Path(id): Path<String>) -> Response {
    json_response(engine.cancel_job(&id))
}

#[cfg_attr(test, utoipa::path(
    get,
    tag = "inference.rs",
    path = "/v1/jobs/{id}/events",
    params(("id" = String, Path, description = "Job ID")),
    responses(
        (status = 200, description = "Server-sent events: `record` (the job as it stands) first, then each change (`JobEvent`), until it finishes", content_type = "text/event-stream"),
        (status = 404, description = "No such job"),
    )
))]
pub async fn job_events(OwnedEngine(engine): OwnedEngine, Path(id): Path<String>) -> Response {
    let (record, events) = match engine.job_events(&id) {
        Ok(subscribed) => subscribed,
        Err(error) => return openai_error_response(error),
    };
    let first = Event::default()
        .event("record")
        .json_data(&record)
        .unwrap_or_default();
    let done = record.status.is_finished();
    let changes = stream::unfold((events, done), |(mut events, done)| async move {
        if done {
            return None;
        }
        match events.recv().await {
            Ok(event) => {
                let finished = matches!(event, JobEvent::Finished { .. });
                let data = Event::default().json_data(&event).unwrap_or_default();
                Some((data, (events, finished)))
            }
            // a slow reader missed changes; it can fetch the record again
            Err(RecvError::Lagged(_)) => {
                let data = Event::default().event("lagged").data("");
                Some((data, (events, false)))
            }
            Err(RecvError::Closed) => None,
        }
    });
    let stream: std::pin::Pin<Box<dyn Stream<Item = Result<Event, Infallible>> + Send>> =
        Box::pin(stream::once(async { first }).chain(changes).map(Ok));
    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

#[cfg_attr(test, utoipa::path(
    get,
    tag = "inference.rs",
    path = "/v1/jobs/{id}/files/{file}",
    params(
        ("id" = String, Path, description = "Job ID"),
        ("file" = String, Path, description = "A preview's or output's path in the job's folder"),
    ),
    responses(
        (status = 200, description = "The file"),
        (status = 404, description = "No such job or file"),
    )
))]
pub async fn job_file(
    OwnedEngine(engine): OwnedEngine,
    Path((id, file)): Path<(String, String)>,
    request: axum::extract::Request,
) -> Response {
    let path = match engine.job_file(&id, &file) {
        Ok(path) => path,
        Err(error) => return openai_error_response(error),
    };
    let extension = path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
        .unwrap_or_default();
    let Some((_, content_type)) = FILE_TYPES.iter().find(|(e, _)| *e == extension) else {
        return openai_error_response(ApiError::new(
            ApiErrorKind::NotFound,
            format!("job `{id}` serves no `{extension}` files"),
            Some("job_file_not_found"),
            Some("file"),
        ));
    };
    let mime = content_type
        .parse::<mime::Mime>()
        .unwrap_or(mime::APPLICATION_OCTET_STREAM);
    match tower_http::services::ServeFile::new_with_mime(&path, &mime)
        .oneshot(request)
        .await
    {
        Ok(response) => {
            let mut response = response.into_response();
            response.headers_mut().insert(
                header::X_CONTENT_TYPE_OPTIONS,
                header::HeaderValue::from_static("nosniff"),
            );
            response
        }
        Err(infallible) => match infallible {},
    }
}
