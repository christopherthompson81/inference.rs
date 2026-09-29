//! The models an engine serves: listing them, and unloading, reloading and inspecting one.

use inference_core::{InferenceRsError, ModelStatus as CoreModelStatus};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::{
    api_error::{ApiError, ApiErrorKind},
    lora_routing::{DEFAULT_MODEL_ID, list_lora_adapter_models},
    openai::{ModelObject, ModelObjects},
    types::SharedInferenceRsState,
};

const MODEL_OBJECT: &str = "model";
const MODEL_LIST_OBJECT: &str = "list";
const MODEL_OWNER: &str = "local";

/// The body of an unload, reload or status request.
#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct ModelOperationRequest {
    #[schema(example = "my-model")]
    pub model_id: String,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, ToSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ModelStatus {
    Loaded,
    Unloaded,
    Reloading,
}

impl From<CoreModelStatus> for ModelStatus {
    fn from(status: CoreModelStatus) -> Self {
        match status {
            CoreModelStatus::Loaded => Self::Loaded,
            CoreModelStatus::Unloaded => Self::Unloaded,
            CoreModelStatus::Reloading => Self::Reloading,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct ModelStatusResponse {
    #[schema(example = "my-model")]
    pub model_id: String,
    pub status: ModelStatus,
}

fn core_error(error: InferenceRsError) -> ApiError {
    ApiError::from_error(&error, ApiErrorKind::Internal)
}

fn model_object(state: &SharedInferenceRsState, id: String) -> ModelObject {
    ModelObject {
        root: Some(id.clone()),
        id,
        object: MODEL_OBJECT,
        created: state.get_creation_time(),
        owned_by: MODEL_OWNER,
        parent: None,
        adapter_generation: None,
        status: None,
        tools_available: None,
        mcp_tools_count: None,
        mcp_servers_connected: None,
        max_model_len: None,
    }
}

/// Every served model, preceded by the `default` alias and followed by each loaded LoRA adapter as its own model.
pub fn list_models(state: &SharedInferenceRsState) -> Result<ModelObjects, ApiError> {
    let models_with_status = state.list_models_with_status().map_err(core_error)?;
    let mut data = Vec::new();
    if !models_with_status.is_empty() {
        data.push(model_object(state, DEFAULT_MODEL_ID.to_string()));
    }
    for (model_id, status) in models_with_status {
        let mut object = model_object(state, model_id.clone());
        if status == CoreModelStatus::Loaded {
            // Each lookup takes the engine lock again, so a model unloading meanwhile leaves these unset rather than
            // failing the list.
            let tools_count = state.get_tools_count(Some(&model_id)).unwrap_or_default();
            let has_mcp = state.has_mcp_client(Some(&model_id)).unwrap_or_default();
            if has_mcp || tools_count > 0 {
                object.tools_available = Some(tools_count > 0);
                object.mcp_tools_count = Some(tools_count);
                object.mcp_servers_connected = Some(1);
            }
            object.max_model_len = state.max_sequence_length(Some(&model_id)).ok().flatten();
        }
        object.status = Some(status.to_string());
        data.push(object);
    }
    for adapter_model in list_lora_adapter_models(state).map_err(core_error)? {
        let mut object = model_object(state, adapter_model.id);
        object.root = Some(adapter_model.adapter.alias.clone());
        object.parent = Some(adapter_model.parent);
        object.adapter_generation = Some(adapter_model.adapter.generation.to_string());
        object.status = Some(CoreModelStatus::Loaded.to_string());
        data.push(object);
    }
    Ok(ModelObjects {
        object: MODEL_LIST_OBJECT,
        data,
    })
}

/// Unloads a model; one that is already unloaded is not an error.
pub fn unload_model(
    state: &SharedInferenceRsState,
    request: ModelOperationRequest,
) -> Result<ModelStatusResponse, ApiError> {
    let result = state.unload_model(&request.model_id);
    unload_result(request.model_id, result)
}

/// Reloads an unloaded model; one that is already loaded is not an error.
pub async fn reload_model(
    state: &SharedInferenceRsState,
    request: ModelOperationRequest,
) -> Result<ModelStatusResponse, ApiError> {
    let result = state.reload_model(&request.model_id).await;
    reload_result(request.model_id, result)
}

pub fn model_status(
    state: &SharedInferenceRsState,
    request: ModelOperationRequest,
) -> Result<ModelStatusResponse, ApiError> {
    let result = state.get_model_status(&request.model_id);
    status_result(request.model_id, result)
}

fn unload_result(
    model_id: String,
    result: Result<(), InferenceRsError>,
) -> Result<ModelStatusResponse, ApiError> {
    match result {
        Ok(()) | Err(InferenceRsError::ModelAlreadyUnloaded(_)) => Ok(ModelStatusResponse {
            model_id,
            status: ModelStatus::Unloaded,
        }),
        Err(error) => Err(core_error(error)),
    }
}

fn reload_result(
    model_id: String,
    result: Result<(), InferenceRsError>,
) -> Result<ModelStatusResponse, ApiError> {
    match result {
        Ok(()) | Err(InferenceRsError::ModelAlreadyLoaded(_)) => Ok(ModelStatusResponse {
            model_id,
            status: ModelStatus::Loaded,
        }),
        Err(error) => Err(core_error(error)),
    }
}

fn status_result(
    model_id: String,
    result: Result<Option<CoreModelStatus>, InferenceRsError>,
) -> Result<ModelStatusResponse, ApiError> {
    match result.map_err(core_error)? {
        Some(status) => Ok(ModelStatusResponse {
            model_id,
            status: status.into(),
        }),
        None => Err(core_error(InferenceRsError::ModelNotFound(model_id))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kind(result: Result<ModelStatusResponse, ApiError>) -> Result<ModelStatus, ApiErrorKind> {
        result
            .map(|response| response.status)
            .map_err(|error| error.kind)
    }

    #[test]
    fn unload_results_use_operation_kinds() {
        let model = || "model".to_string();
        assert_eq!(
            kind(unload_result(model(), Ok(()))),
            Ok(ModelStatus::Unloaded)
        );
        assert_eq!(
            kind(unload_result(
                model(),
                Err(InferenceRsError::ModelAlreadyUnloaded(model()))
            )),
            Ok(ModelStatus::Unloaded)
        );
        assert_eq!(
            kind(unload_result(
                model(),
                Err(InferenceRsError::ModelNotFound(model()))
            )),
            Err(ApiErrorKind::NotFound)
        );
        assert_eq!(
            kind(unload_result(
                model(),
                Err(InferenceRsError::NoLoaderConfig(model()))
            )),
            Err(ApiErrorKind::InvalidRequest)
        );
        assert_eq!(
            kind(unload_result(
                model(),
                Err(InferenceRsError::ModelReloading(model()))
            )),
            Err(ApiErrorKind::Conflict)
        );
        assert_eq!(
            kind(unload_result(
                model(),
                Err(InferenceRsError::EnginePoisoned)
            )),
            Err(ApiErrorKind::Internal)
        );
    }

    #[test]
    fn reload_and_status_results_preserve_idempotency() {
        let model = || "model".to_string();
        assert_eq!(
            kind(reload_result(
                model(),
                Err(InferenceRsError::ModelAlreadyLoaded(model()))
            )),
            Ok(ModelStatus::Loaded)
        );
        assert_eq!(
            kind(reload_result(
                model(),
                Err(InferenceRsError::ModelReloading(model()))
            )),
            Err(ApiErrorKind::Conflict)
        );
        assert_eq!(
            kind(status_result(model(), Ok(None))),
            Err(ApiErrorKind::NotFound)
        );
        assert_eq!(
            kind(status_result(model(), Ok(Some(CoreModelStatus::Reloading)))),
            Ok(ModelStatus::Reloading)
        );
    }

    #[test]
    fn internal_model_errors_do_not_expose_details() {
        let error = reload_result(
            "model".to_string(),
            Err(InferenceRsError::ReloadFailed(
                "private failure".to_string(),
            )),
        )
        .unwrap_err();
        assert_eq!(error.kind, ApiErrorKind::Internal);
        assert!(!error.message.contains("private failure"), "{error:?}");
    }
}
