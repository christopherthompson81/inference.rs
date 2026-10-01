//! The LoRA adapter routes: HTTP framing over the engine's adapter management.

use axum::{
    extract::{Query, rejection::QueryRejection},
    response::Response,
};

use crate::handler_core::{ApiJson, ApiJsonRejection};
pub use crate::lora_adapters_api::{
    ALLOW_RUNTIME_LORA_UPDATING_ENV, LORA_ADAPTER_ROOT_ENV, ListLoraAdaptersQuery,
    LoadLoraAdapterRequest, LoraAdapterApiConfig, LoraAdapterListResponse, LoraAdapterObject,
    LoraResidentGenerationObject, UnloadLoraAdapterRequest, runtime_lora_updates_enabled,
};
use crate::{
    handler_core::{ApiError, ApiErrorKind, json_response, openai_error_response},
    types::OwnedEngine,
};

const INVALID_QUERY: &str = "invalid_query";

#[cfg_attr(test, utoipa::path(
    post,
    tag = "LoRA adapters",
    path = "/v1/load_lora_adapter",
    description = "Registered only when runtime LoRA mutation is enabled. CLI servers enable it with INFERENCE_RS_ALLOW_RUNTIME_LORA_UPDATING.",
    request_body(
        content = LoadLoraAdapterRequest,
        example = json!({"lora_name": "production", "lora_path": "/srv/adapters/production"})
    ),
    responses(
        (status = 200, description = "LoRA adapter loaded", body = LoraAdapterObject),
        (status = 400, description = "Invalid request"),
        (status = 403, description = "Adapter path is not allowed"),
        (status = 404, description = "Model or adapter path was not found"),
        (status = 409, description = "LoRA runtime is unavailable or at capacity"),
        (status = 413, description = "Adapter input files exceed the configured safety limit"),
        (status = 415, description = "Request content type is not JSON"),
        (status = 429, description = "Another adapter load is already in progress"),
        (status = 503, description = "Adapter storage or model device is unavailable"),
        (status = 500, description = "Adapter loading task failed")
    )
))]
pub(crate) async fn load_lora_adapter(
    OwnedEngine(engine): OwnedEngine,
    payload: Result<ApiJson<LoadLoraAdapterRequest>, ApiJsonRejection>,
) -> Response {
    match payload {
        Ok(ApiJson(request)) => json_response(engine.load_lora_adapter(request).await),
        Err(ApiJsonRejection(error)) => openai_error_response(error),
    }
}

#[cfg_attr(test, utoipa::path(
    post,
    tag = "LoRA adapters",
    path = "/v1/unload_lora_adapter",
    description = "Registered only when runtime LoRA mutation is enabled. CLI servers enable it with INFERENCE_RS_ALLOW_RUNTIME_LORA_UPDATING.",
    request_body(
        content = UnloadLoraAdapterRequest,
        example = json!({"lora_name": "production"})
    ),
    responses(
        (status = 200, description = "LoRA adapter unloaded", body = LoraAdapterObject),
        (status = 400, description = "Invalid request"),
        (status = 404, description = "Model or adapter was not found"),
        (status = 409, description = "LoRA runtime is unavailable"),
        (status = 413, description = "Request body exceeds the configured limit"),
        (status = 415, description = "Request content type is not JSON"),
        (status = 500, description = "Internal server error")
    )
))]
pub(crate) async fn unload_lora_adapter(
    OwnedEngine(engine): OwnedEngine,
    payload: Result<ApiJson<UnloadLoraAdapterRequest>, ApiJsonRejection>,
) -> Response {
    match payload {
        Ok(ApiJson(request)) => json_response(engine.unload_lora_adapter(request).await),
        Err(ApiJsonRejection(error)) => openai_error_response(error),
    }
}

#[cfg_attr(test, utoipa::path(
    get,
    tag = "LoRA adapters",
    path = "/v1/lora_adapters",
    description = "Always registered for adapter discovery and status, even when runtime LoRA mutation is disabled.",
    params(ListLoraAdaptersQuery),
    responses(
        (status = 200, description = "Loaded LoRA adapters", body = LoraAdapterListResponse),
        (status = 400, description = "Invalid query"),
        (status = 404, description = "Model was not found"),
        (status = 409, description = "LoRA runtime is unavailable"),
        (status = 500, description = "Internal server error")
    )
))]
pub(crate) async fn list_lora_adapters(
    OwnedEngine(engine): OwnedEngine,
    payload: Result<Query<ListLoraAdaptersQuery>, QueryRejection>,
) -> Response {
    match payload {
        Ok(Query(query)) => json_response(engine.lora_adapters(query).await),
        Err(error) => openai_error_response(ApiError::new(
            ApiErrorKind::InvalidRequest,
            error.body_text(),
            Some(INVALID_QUERY),
            None,
        )),
    }
}
