//! ## General inference.rs server route handlers.

#[cfg(test)]
use crate::openai::ModelObjects;
use axum::extract::{Json, Path};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
#[cfg(test)]
use inference_api::operations::ReIsqResponse;
use inference_api::operations::{CalibrationApplyRequest, ReIsqRequest};
use inference_api::system::TuneModelRequest;
use inference_core::{CalibrationAction, SerializedSession};

use crate::handler_core::{ApiJson, ApiJsonRejection};
pub use crate::models_api::ModelOperationRequest;
#[cfg(test)]
pub use crate::models_api::{ModelStatus, ModelStatusResponse};
use crate::{
    handler_core::{ApiError, json_response, openai_error_response},
    system,
    types::OwnedEngine,
};

#[cfg_attr(test, utoipa::path(
  get,
  tag = "inference.rs",
  path = "/v1/models",
  responses(
    (status = 200, description = "Served model info", body = ModelObjects),
    (status = 500, description = "Failed to inspect the model registry")
  )
))]
pub async fn models(OwnedEngine(engine): OwnedEngine) -> Response {
    json_response(engine.models())
}

#[cfg_attr(test, utoipa::path(
  get,
  tag = "inference.rs",
  path = "/v1/models/cache_stats",
  responses(
    (status = 200, description = "Each loaded model's cumulative prefix- and encoder-cache counters",
     body = inference_api::models::CacheStats),
    (status = 500, description = "Failed to inspect the model registry")
  )
))]
pub async fn model_cache_stats(OwnedEngine(engine): OwnedEngine) -> Response {
    json_response(engine.cache_stats())
}

#[cfg_attr(test, utoipa::path(
  get,
  tag = "inference.rs",
  path = "/health",
  responses((status = 200, description = "Server is healthy"))
))]
pub async fn health() -> &'static str {
    "OK"
}

#[cfg_attr(test, utoipa::path(
  get,
  tag = "inference.rs",
  path = "/v1/system/info",
  responses((status = 200, description = "Host, device, and build information"))
))]
pub async fn system_info() -> Json<inference_selection::SystemInfo> {
    Json(system::system_info())
}

#[cfg_attr(test, utoipa::path(
  post,
  tag = "inference.rs",
  path = "/v1/system/doctor",
  responses((status = 200, description = "Environment diagnostics report"))
))]
pub async fn system_doctor() -> Json<inference_selection::DoctorReport> {
    Json(system::system_doctor())
}

#[cfg_attr(test, utoipa::path(
  post,
  tag = "inference.rs",
  path = "/re_isq",
  request_body = ReIsqRequest,
  responses(
    (status = 200, description = "Requantization queued for a model that was loaded with ISQ.", body = ReIsqResponse),
    (status = 400, description = "Invalid ISQ type"),
    (status = 500, description = "Failed to dispatch the ISQ request")
  )
))]
pub async fn re_isq(
    OwnedEngine(engine): OwnedEngine,
    payload: Result<ApiJson<ReIsqRequest>, ApiJsonRejection>,
) -> Response {
    let request = match payload {
        Ok(ApiJson(request)) => request,
        Err(ApiJsonRejection(error)) => return openai_error_response(error),
    };
    json_response(engine.re_isq(request).await)
}

// remote clients only get a bare file name so the write can't leave the working directory
fn http_save_cimatrix_path(name: &str) -> Result<std::path::PathBuf, ApiError> {
    let path = std::path::Path::new(name);
    let bare = !name.contains(['/', '\\'])
        && matches!(
            path.components().collect::<Vec<_>>().as_slice(),
            [std::path::Component::Normal(_)]
        );
    if !bare {
        return Err(ApiError::invalid_request(format!(
            "`save_cimatrix` must be a bare file name, got `{name}`"
        )));
    }
    Ok(path.to_path_buf())
}

#[cfg_attr(test, utoipa::path(
  post,
  tag = "inference.rs",
  path = "/calibration/start",
  responses((status = 200, description = "Begin collecting activation statistics from live traffic.", body = inference_core::CalibrationStatus))
))]
pub async fn calibration_start(OwnedEngine(engine): OwnedEngine) -> Response {
    json_response(engine.calibration(CalibrationAction::Start).await)
}

#[cfg_attr(test, utoipa::path(
  get,
  tag = "inference.rs",
  path = "/calibration/status",
  responses((status = 200, description = "Per-layer calibration collection progress.", body = inference_core::CalibrationStatus))
))]
pub async fn calibration_status(OwnedEngine(engine): OwnedEngine) -> Response {
    json_response(engine.calibration(CalibrationAction::Status).await)
}

#[cfg_attr(test, utoipa::path(
  post,
  tag = "inference.rs",
  path = "/calibration/apply",
  request_body = CalibrationApplyRequest,
  responses((status = 200, description = "Requantize with collected statistics and hot-swap the layers.", body = inference_core::CalibrationStatus))
))]
pub async fn calibration_apply(
    OwnedEngine(engine): OwnedEngine,
    payload: Result<ApiJson<CalibrationApplyRequest>, ApiJsonRejection>,
) -> Response {
    let request = match payload {
        Ok(ApiJson(request)) => request,
        Err(ApiJsonRejection(error)) => return openai_error_response(error),
    };
    let save_cimatrix = match request
        .save_cimatrix
        .as_deref()
        .map(http_save_cimatrix_path)
        .transpose()
    {
        Ok(path) => path,
        Err(error) => return openai_error_response(error),
    };
    json_response(
        engine
            .calibration(CalibrationAction::Apply { save_cimatrix })
            .await,
    )
}

fn model_operation_request(
    payload: Result<ApiJson<ModelOperationRequest>, ApiJsonRejection>,
) -> Result<ModelOperationRequest, ApiError> {
    payload
        .map(|ApiJson(request)| request)
        .map_err(|ApiJsonRejection(error)| error)
}

#[cfg_attr(test, utoipa::path(
  post,
  tag = "inference.rs",
  path = "/v1/models/unload",
  request_body = ModelOperationRequest,
  responses(
    (status = 200, description = "Model unloaded or already unloaded", body = ModelStatusResponse),
    (status = 400, description = "Invalid request or model cannot be unloaded"),
    (status = 404, description = "Model not found"),
    (status = 409, description = "Model state conflicts with the operation"),
    (status = 413, description = "Request body is too large"),
    (status = 415, description = "Request content type is not JSON"),
    (status = 500, description = "Model registry failure")
  )
))]
pub async fn unload_model(
    OwnedEngine(engine): OwnedEngine,
    payload: Result<ApiJson<ModelOperationRequest>, ApiJsonRejection>,
) -> Response {
    match model_operation_request(payload) {
        Ok(request) => json_response(engine.unload_model(request)),
        Err(error) => openai_error_response(error),
    }
}

#[cfg_attr(test, utoipa::path(
  post,
  tag = "inference.rs",
  path = "/v1/models/reload",
  request_body = ModelOperationRequest,
  responses(
    (status = 200, description = "Model reloaded or already loaded", body = ModelStatusResponse),
    (status = 400, description = "Invalid request or model cannot be reloaded"),
    (status = 404, description = "Model not found"),
    (status = 409, description = "Model state conflicts with the operation"),
    (status = 413, description = "Request body is too large"),
    (status = 415, description = "Request content type is not JSON"),
    (status = 500, description = "Model reload failure")
  )
))]
pub async fn reload_model(
    OwnedEngine(engine): OwnedEngine,
    payload: Result<ApiJson<ModelOperationRequest>, ApiJsonRejection>,
) -> Response {
    match model_operation_request(payload) {
        Ok(request) => json_response(engine.reload_model(request).await),
        Err(error) => openai_error_response(error),
    }
}

#[cfg_attr(test, utoipa::path(
  post,
  tag = "inference.rs",
  path = "/v1/models/status",
  request_body = ModelOperationRequest,
  responses(
    (status = 200, description = "Model status", body = ModelStatusResponse),
    (status = 400, description = "Invalid request"),
    (status = 404, description = "Model not found"),
    (status = 413, description = "Request body is too large"),
    (status = 415, description = "Request content type is not JSON"),
    (status = 500, description = "Model registry failure")
  )
))]
pub async fn get_model_status(
    OwnedEngine(engine): OwnedEngine,
    payload: Result<ApiJson<ModelOperationRequest>, ApiJsonRejection>,
) -> Response {
    match model_operation_request(payload) {
        Ok(request) => json_response(engine.model_status(request)),
        Err(error) => openai_error_response(error),
    }
}

#[cfg_attr(test, utoipa::path(
  post,
  tag = "inference.rs",
  path = "/v1/models/tune",
  request_body = TuneModelRequest,
  responses(
    (status = 200, description = "Auto-tune result with recommended settings"),
    (status = 400, description = "Invalid tuning request"),
    (status = 413, description = "Request body is too large"),
    (status = 415, description = "Request content type is not JSON"),
    (status = 500, description = "Tuning failed")
  )
))]
pub async fn tune_model(payload: Result<ApiJson<TuneModelRequest>, ApiJsonRejection>) -> Response {
    let request = match payload {
        Ok(ApiJson(request)) => request,
        Err(ApiJsonRejection(error)) => return openai_error_response(error),
    };
    match system::tune_model(request) {
        Ok(result) => Json(result).into_response(),
        Err(error) => openai_error_response(error),
    }
}

/// GET `/v1/sessions/{session_id}`. 404 if the session doesn't exist.
#[cfg_attr(test, utoipa::path(
    get,
    tag = "inference.rs",
    path = "/v1/sessions/{session_id}",
    params(("session_id" = String, Path, description = "Session ID to export")),
    responses(
        (status = 200, description = "Serialized agentic session", body = SerializedSession),
        (status = 404, description = "Session not found"),
    )
))]
pub async fn get_session(
    OwnedEngine(engine): OwnedEngine,
    Path(session_id): Path<String>,
) -> Response {
    json_response(engine.session(&session_id))
}

/// PUT `/v1/sessions/{session_id}`. Replaces any existing session.
#[cfg_attr(test, utoipa::path(
    put,
    tag = "inference.rs",
    path = "/v1/sessions/{session_id}",
    params(("session_id" = String, Path, description = "Session ID to import as")),
    request_body = SerializedSession,
    responses(
        (status = 200, description = "Session imported"),
        (status = 400, description = "Invalid session payload"),
    )
))]
pub async fn put_session(
    OwnedEngine(engine): OwnedEngine,
    Path(session_id): Path<String>,
    payload: Result<ApiJson<SerializedSession>, ApiJsonRejection>,
) -> Response {
    let session = match payload {
        Ok(ApiJson(session)) => session,
        Err(ApiJsonRejection(error)) => return openai_error_response(error),
    };
    match engine.put_session(&session_id, session) {
        Ok(_) => StatusCode::OK.into_response(),
        Err(error) => openai_error_response(error),
    }
}

/// DELETE `/v1/sessions/{session_id}`. Idempotent: returns 200 either way.
#[cfg_attr(test, utoipa::path(
    delete,
    tag = "inference.rs",
    path = "/v1/sessions/{session_id}",
    params(("session_id" = String, Path, description = "Session ID to delete")),
    responses((status = 200, description = "Session deleted (or did not exist)"))
))]
pub async fn delete_session(
    OwnedEngine(engine): OwnedEngine,
    Path(session_id): Path<String>,
) -> Response {
    match engine.delete_session(&session_id) {
        Ok(_) => StatusCode::OK.into_response(),
        Err(error) => openai_error_response(error),
    }
}

#[cfg(test)]
mod tests {
    use axum::{body::Body, extract::FromRequest, http::Request as HttpRequest};

    use super::*;
    use crate::handler_core::ApiErrorHttp;

    #[test]
    fn http_save_cimatrix_accepts_only_bare_file_names() {
        assert!(http_save_cimatrix_path("traffic.cimatrix").is_ok());
        for name in [
            "/tmp/pwned.cimatrix",
            "../../etc/escaped.cimatrix",
            "subdir/traffic.cimatrix",
            "subdir\\traffic.cimatrix",
            "./traffic.cimatrix",
            "..",
            ".",
            "",
        ] {
            assert!(http_save_cimatrix_path(name).is_err(), "{name:?}");
        }
    }

    #[tokio::test]
    async fn lifecycle_json_rejections_use_openai_statuses() {
        let request = HttpRequest::builder()
            .header(axum::http::header::CONTENT_TYPE, "application/json")
            .body(Body::from("{}"))
            .unwrap();
        let rejection = ApiJson::<ModelOperationRequest>::from_request(request, &())
            .await
            .unwrap_err();
        let error = model_operation_request(Err(rejection)).unwrap_err();

        assert_eq!(error.status(), StatusCode::BAD_REQUEST);
        assert_eq!(error.code.as_deref(), Some("invalid_request_body"));
    }
}
