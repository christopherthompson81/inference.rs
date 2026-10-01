//! ## General inference.rs server route handlers.

#[cfg(test)]
use crate::openai::ModelObjects;
use axum::Extension;
use axum::extract::{Json, Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
#[cfg(test)]
use inference_api::operations::ReIsqResponse;
use inference_api::operations::{self, CalibrationApplyRequest, ReIsqRequest};
use inference_api::request_body::{JsonRequest, parse_json};
use inference_core::{
    AutoDeviceMapParams, CalibrationAction, InferenceRs, ModelDType, SerializedSession,
    TokenSource, parse_isq_value,
};
use inference_selection::{AutoTuneRequest, ModelSelected, TuneProfile, auto_tune};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::auth::Owner;
use crate::handler_core::{ApiJson, ApiJsonRejection};
pub use crate::models_api::ModelOperationRequest;
#[cfg(test)]
pub use crate::models_api::{ModelStatus, ModelStatusResponse};
use crate::{
    handler_core::{ApiError, ApiErrorKind, json_response, openai_error_response},
    models_api::{
        cache_stats, list_models, model_status as status, reload_model as reload,
        unload_model as unload,
    },
    system,
    types::ExtractedInferenceRsState,
};

#[derive(Debug, Clone, Copy, Deserialize, Serialize, ToSchema)]
#[serde(rename_all = "kebab-case")]
pub enum TuneProfileRequest {
    Quality,
    Balanced,
    Fast,
}

impl From<TuneProfileRequest> for TuneProfile {
    fn from(value: TuneProfileRequest) -> Self {
        match value {
            TuneProfileRequest::Quality => TuneProfile::Quality,
            TuneProfileRequest::Balanced => TuneProfile::Balanced,
            TuneProfileRequest::Fast => TuneProfile::Fast,
        }
    }
}

#[cfg_attr(test, utoipa::path(
  get,
  tag = "inference.rs",
  path = "/v1/models",
  responses(
    (status = 200, description = "Served model info", body = ModelObjects),
    (status = 500, description = "Failed to inspect the model registry")
  )
))]
pub async fn models(State(state): ExtractedInferenceRsState) -> Response {
    json_response(list_models(&state))
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
pub async fn model_cache_stats(State(state): ExtractedInferenceRsState) -> Response {
    json_response(cache_stats(&state))
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
    State(state): ExtractedInferenceRsState,
    payload: Result<ApiJson<ReIsqRequest>, ApiJsonRejection>,
) -> Response {
    let request = match payload {
        Ok(ApiJson(request)) => request,
        Err(ApiJsonRejection(error)) => return openai_error_response(error),
    };
    InferenceRs::maybe_log_request(state.clone(), format!("Re ISQ: {:?}", request.ggml_type));
    json_response(operations::re_isq(&state, request).await)
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
pub async fn calibration_start(State(state): ExtractedInferenceRsState) -> Response {
    InferenceRs::maybe_log_request(state.clone(), "Calibration start".to_string());
    json_response(operations::calibration(&state, CalibrationAction::Start).await)
}

#[cfg_attr(test, utoipa::path(
  get,
  tag = "inference.rs",
  path = "/calibration/status",
  responses((status = 200, description = "Per-layer calibration collection progress.", body = inference_core::CalibrationStatus))
))]
pub async fn calibration_status(State(state): ExtractedInferenceRsState) -> Response {
    json_response(operations::calibration(&state, CalibrationAction::Status).await)
}

#[cfg_attr(test, utoipa::path(
  post,
  tag = "inference.rs",
  path = "/calibration/apply",
  request_body = CalibrationApplyRequest,
  responses((status = 200, description = "Requantize with collected statistics and hot-swap the layers.", body = inference_core::CalibrationStatus))
))]
pub async fn calibration_apply(
    State(state): ExtractedInferenceRsState,
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
    InferenceRs::maybe_log_request(state.clone(), "Calibration apply".to_string());
    json_response(operations::calibration(&state, CalibrationAction::Apply { save_cimatrix }).await)
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
    State(state): ExtractedInferenceRsState,
    payload: Result<ApiJson<ModelOperationRequest>, ApiJsonRejection>,
) -> Response {
    match model_operation_request(payload) {
        Ok(request) => json_response(unload(&state, request)),
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
    State(state): ExtractedInferenceRsState,
    payload: Result<ApiJson<ModelOperationRequest>, ApiJsonRejection>,
) -> Response {
    match model_operation_request(payload) {
        Ok(request) => json_response(reload(&state, request).await),
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
    State(state): ExtractedInferenceRsState,
    payload: Result<ApiJson<ModelOperationRequest>, ApiJsonRejection>,
) -> Response {
    match model_operation_request(payload) {
        Ok(request) => json_response(status(&state, request)),
        Err(error) => openai_error_response(error),
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct TuneModelRequest {
    #[schema(example = "meta-llama/Llama-3.2-3B-Instruct")]
    pub model_id: String,
    /// Optional model dtype (auto, f16, bf16, etc)
    #[serde(default)]
    pub dtype: Option<String>,
    /// Optional max sequence length for tuning
    #[serde(default)]
    pub max_seq_len: Option<usize>,
    /// Optional max batch size for tuning
    #[serde(default)]
    pub max_batch_size: Option<usize>,
    /// Optional max num images (multimodal)
    #[serde(default)]
    pub max_num_images: Option<usize>,
    /// Optional max image length (multimodal)
    #[serde(default)]
    pub max_image_length: Option<usize>,
    /// Optional tuning profile
    #[serde(default)]
    pub profile: Option<TuneProfileRequest>,
    /// Optional fixed ISQ level to test (e.g., Q4K)
    #[serde(default)]
    pub requested_isq: Option<String>,
    /// Optional HF token source
    #[serde(default)]
    pub token_source: Option<String>,
    /// Optional HF revision
    #[serde(default)]
    pub hf_revision: Option<String>,
    /// Force CPU-only tuning
    #[serde(default)]
    pub cpu: Option<bool>,
}

impl JsonRequest for TuneModelRequest {
    fn from_json(body: &[u8]) -> Result<Self, ApiError> {
        parse_json(body)
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
    let token_source = match request.token_source {
        Some(value) => match value.parse() {
            Ok(token_source) => token_source,
            Err(error) => {
                return openai_error_response(ApiError::new(
                    ApiErrorKind::InvalidRequest,
                    format!("Invalid token_source: {error}"),
                    Some("invalid_token_source"),
                    Some("token_source"),
                ));
            }
        },
        None => TokenSource::CacheToken,
    };

    let dtype = match request
        .dtype
        .as_deref()
        .unwrap_or("auto")
        .parse::<ModelDType>()
    {
        Ok(dtype) => dtype,
        Err(error) => {
            return openai_error_response(ApiError::new(
                ApiErrorKind::InvalidRequest,
                format!("Invalid dtype: {error}"),
                Some("invalid_dtype"),
                Some("dtype"),
            ));
        }
    };

    let max_seq_len = request
        .max_seq_len
        .unwrap_or(AutoDeviceMapParams::DEFAULT_MAX_SEQ_LEN);
    let max_batch_size = request
        .max_batch_size
        .unwrap_or(AutoDeviceMapParams::DEFAULT_MAX_BATCH_SIZE);

    let model_selected = ModelSelected::Run {
        model_id: request.model_id.clone(),
        quant: None,
        tokenizer_json: None,
        dtype,
        topology: None,
        organization: None,
        write_uqff: None,
        from_uqff: None,
        imatrix: None,
        calibration_file: None,
        max_edge: None,
        max_seq_len,
        max_batch_size,
        max_num_images: request.max_num_images,
        max_image_length: request.max_image_length,
        hf_cache_path: None,
        matformer_config_path: None,
        matformer_slice_name: None,
    };

    let requested_isq = match request.requested_isq {
        Some(value) => match parse_isq_value(&value, None) {
            Ok(value) => Some(value),
            Err(error) => {
                return openai_error_response(ApiError::new(
                    ApiErrorKind::InvalidRequest,
                    format!("Invalid isq value: {error}"),
                    Some("invalid_isq"),
                    Some("requested_isq"),
                ));
            }
        },
        None => None,
    };

    let tune_request = AutoTuneRequest {
        model: model_selected,
        token_source,
        hf_revision: request.hf_revision,
        force_cpu: request.cpu.unwrap_or(false),
        profile: request
            .profile
            .map(Into::into)
            .unwrap_or(TuneProfile::Balanced),
        requested_isq,
    };

    match auto_tune(tune_request) {
        Ok(result) => Json(result).into_response(),
        Err(error) => {
            tracing::error!(%error, "model auto-tuning failed");
            openai_error_response(ApiError::internal())
        }
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
    State(state): ExtractedInferenceRsState,
    Extension(owner): Extension<Owner>,
    Path(session_id): Path<String>,
) -> Response {
    json_response(operations::export_session(
        &state,
        &session_id,
        owner.as_deref(),
    ))
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
    State(state): ExtractedInferenceRsState,
    Extension(owner): Extension<Owner>,
    Path(session_id): Path<String>,
    payload: Result<ApiJson<SerializedSession>, ApiJsonRejection>,
) -> Response {
    let session = match payload {
        Ok(ApiJson(session)) => session,
        Err(ApiJsonRejection(error)) => return openai_error_response(error),
    };
    match operations::import_session(&state, session_id, session, owner.as_deref()) {
        Ok(()) => StatusCode::OK.into_response(),
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
    State(state): ExtractedInferenceRsState,
    Extension(owner): Extension<Owner>,
    Path(session_id): Path<String>,
) -> Response {
    match operations::delete_session(&state, &session_id, owner.as_deref()) {
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
