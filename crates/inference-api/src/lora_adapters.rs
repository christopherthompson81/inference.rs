//! Runtime LoRA adapter management: listing adapters, and loading and unloading them from a guarded filesystem root.

use std::{
    fs::File,
    path::{Path, PathBuf},
};

use anyhow::{Context, bail};
use futures::future::BoxFuture;
use inference_core::{
    LoraAdapterFiles, LoraAdapterInfo, LoraAdapterLoadPolicy, MAX_LORA_ALIAS_BYTES,
};
use serde::{Deserialize, Serialize};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use utoipa::{IntoParams, ToSchema};

use crate::{
    api_error::{ApiError, ApiErrorKind, INTERNAL_ERROR_MESSAGE},
    lora_routing::DEFAULT_MODEL_ID,
    types::SharedInferenceRsState,
};

pub const ALLOW_RUNTIME_LORA_UPDATING_ENV: &str = "INFERENCE_RS_ALLOW_RUNTIME_LORA_UPDATING";
pub const LORA_ADAPTER_ROOT_ENV: &str = "INFERENCE_RS_LORA_ADAPTER_ROOT";

const LORA_ADAPTER_OBJECT: &str = "lora_adapter";
const LORA_ADAPTER_LIST_OBJECT: &str = "list";
const LORA_CONFIG_FILE: &str = "adapter_config.json";
const LORA_WEIGHTS_FILE: &str = "adapter_model.safetensors";
const MAX_CONCURRENT_LORA_LOADS: usize = 1;

/// Whether runtime LoRA loading and unloading are allowed, and the filesystem root adapters must load from.
#[derive(Clone, Debug)]
pub struct LoraAdapterApiConfig {
    enabled: bool,
    allowed_root: Option<PathBuf>,
    load_gate: std::sync::Arc<Semaphore>,
}

impl Default for LoraAdapterApiConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            allowed_root: None,
            load_gate: std::sync::Arc::new(Semaphore::new(MAX_CONCURRENT_LORA_LOADS)),
        }
    }
}

impl LoraAdapterApiConfig {
    pub fn from_env() -> Self {
        Self {
            enabled: runtime_lora_updates_enabled(),
            allowed_root: std::env::var_os(LORA_ADAPTER_ROOT_ENV)
                .filter(|value| !value.is_empty())
                .map(PathBuf::from),
            ..Self::default()
        }
    }

    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    pub fn with_allowed_root(mut self, root: impl Into<PathBuf>) -> Self {
        self.allowed_root = Some(root.into());
        self
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    pub fn allowed_root(&self) -> Option<&Path> {
        self.allowed_root.as_deref()
    }

    pub fn prepare(mut self) -> anyhow::Result<Self> {
        if !self.enabled {
            return Ok(self);
        }
        if let Some(root) = &self.allowed_root {
            let root = root.canonicalize().with_context(|| {
                format!("failed to resolve LoRA adapter root `{}`", root.display())
            })?;
            let metadata = root.metadata().with_context(|| {
                format!("failed to inspect LoRA adapter root `{}`", root.display())
            })?;
            if !metadata.is_dir() {
                bail!("LoRA adapter root `{}` is not a directory", root.display());
            }
            self.allowed_root = Some(root);
        }
        Ok(self)
    }

    fn open_adapter_files(&self, path: &Path) -> Result<OpenedAdapterFiles, ApiError> {
        let path = if path.is_relative() {
            self.allowed_root()
                .map(|root| root.join(path))
                .unwrap_or_else(|| path.to_path_buf())
        } else {
            path.to_path_buf()
        };
        let description = format!("adapter directory `{}`", path.display());
        let adapter_dir = path
            .canonicalize()
            .map_err(|error| adapter_filesystem_error(&description, error))?;
        // Checked before anything else about the path, so a caller cannot tell what exists outside the root.
        self.ensure_allowed(&adapter_dir, "adapter directory")?;
        let metadata = adapter_dir
            .metadata()
            .map_err(|error| adapter_filesystem_error(&description, error))?;
        if !metadata.is_dir() {
            return Err(lora_error(
                ApiErrorKind::InvalidRequest,
                "invalid_adapter_path",
                format!("{description} is not a directory"),
            ));
        }
        let (config_path, config) = self.open_adapter_file(&adapter_dir, LORA_CONFIG_FILE)?;
        let (weights_path, weights) = self.open_adapter_file(&adapter_dir, LORA_WEIGHTS_FILE)?;
        Ok(OpenedAdapterFiles {
            source: adapter_dir.display().to_string(),
            config_path,
            weights_path,
            config,
            weights,
        })
    }

    fn open_adapter_file(
        &self,
        adapter_dir: &Path,
        filename: &str,
    ) -> Result<(PathBuf, File), ApiError> {
        let path = adapter_dir.join(filename);
        let description = format!("adapter file `{filename}`");
        let path = path
            .canonicalize()
            .map_err(|error| adapter_filesystem_error(&description, error))?;
        self.ensure_allowed(&path, filename)?;
        let metadata = path
            .metadata()
            .map_err(|error| adapter_filesystem_error(&description, error))?;
        if !metadata.is_file() {
            return Err(lora_error(
                ApiErrorKind::InvalidRequest,
                "invalid_adapter_file",
                format!("adapter file `{filename}` is not a regular file"),
            ));
        }
        let file =
            File::open(&path).map_err(|error| adapter_filesystem_error(&description, error))?;
        let verified_path = path
            .canonicalize()
            .map_err(|error| adapter_filesystem_error(&description, error))?;
        self.ensure_allowed(&verified_path, filename)?;
        verify_open_file(&file, &verified_path, filename)?;
        Ok((verified_path, file))
    }

    fn ensure_allowed(&self, path: &Path, description: &str) -> Result<(), ApiError> {
        if self
            .allowed_root()
            .is_some_and(|root| !path.starts_with(root))
        {
            return Err(lora_error(
                ApiErrorKind::Forbidden,
                "adapter_path_forbidden",
                format!("{description} resolves outside the configured LoRA adapter root"),
            ));
        }
        Ok(())
    }

    fn try_begin_load(&self) -> Result<OwnedSemaphorePermit, ApiError> {
        self.load_gate.clone().try_acquire_owned().map_err(|_| {
            lora_error(
                ApiErrorKind::RateLimited,
                "lora_load_busy",
                "another LoRA adapter load is already in progress",
            )
        })
    }
}

#[derive(Debug)]
struct OpenedAdapterFiles {
    source: String,
    config_path: PathBuf,
    weights_path: PathBuf,
    config: File,
    weights: File,
}

impl OpenedAdapterFiles {
    fn into_runtime_files(self) -> LoraAdapterFiles {
        LoraAdapterFiles::new(
            self.source,
            self.config_path,
            self.config,
            self.weights_path,
            self.weights,
        )
    }
}

fn adapter_filesystem_error(description: &str, error: std::io::Error) -> ApiError {
    let (kind, code) = match error.kind() {
        std::io::ErrorKind::NotFound => (ApiErrorKind::NotFound, "adapter_path_not_found"),
        std::io::ErrorKind::PermissionDenied => (ApiErrorKind::Forbidden, "adapter_path_forbidden"),
        std::io::ErrorKind::NotADirectory | std::io::ErrorKind::IsADirectory => {
            (ApiErrorKind::InvalidRequest, "invalid_adapter_path")
        }
        std::io::ErrorKind::InvalidData | std::io::ErrorKind::InvalidInput => {
            (ApiErrorKind::InvalidRequest, "invalid_adapter_path")
        }
        _ => (ApiErrorKind::Internal, "internal_error"),
    };
    lora_error(
        kind,
        code,
        format!("failed to access {description}: {error}"),
    )
}

// Engine-side failures keep their code but not their detail, which can name server paths.
fn lora_error(kind: ApiErrorKind, code: &str, message: impl Into<String>) -> ApiError {
    let message = if kind == ApiErrorKind::Internal {
        INTERNAL_ERROR_MESSAGE.to_string()
    } else {
        message.into()
    };
    ApiError::new(kind, message, Some(code), None)
}

fn adapter_task_error(action: &str, error: tokio::task::JoinError) -> ApiError {
    lora_error(
        ApiErrorKind::Internal,
        "lora_load_task_failed",
        format!("{action} task failed: {error}"),
    )
}

#[cfg_attr(not(unix), allow(unused_variables))]
fn verify_open_file(file: &File, path: &Path, filename: &str) -> Result<(), ApiError> {
    let opened = file
        .metadata()
        .map_err(|error| adapter_filesystem_error(&format!("adapter file `{filename}`"), error))?;
    if !opened.is_file() {
        return Err(lora_error(
            ApiErrorKind::InvalidRequest,
            "invalid_adapter_file",
            format!("adapter file `{filename}` is not a regular file"),
        ));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;

        let current = path.metadata().map_err(|error| {
            adapter_filesystem_error(&format!("adapter file `{filename}`"), error)
        })?;
        if opened.dev() != current.dev() || opened.ino() != current.ino() {
            return Err(lora_error(
                ApiErrorKind::Conflict,
                "adapter_file_changed",
                format!("adapter file `{filename}` changed while it was being opened"),
            ));
        }
    }
    Ok(())
}

fn normalize_model_id(model: Option<String>) -> Option<String> {
    model.and_then(|model| {
        let model = model.trim();
        (!model.is_empty() && model != DEFAULT_MODEL_ID).then(|| model.to_string())
    })
}

pub fn runtime_lora_updates_enabled() -> bool {
    std::env::var(ALLOW_RUNTIME_LORA_UPDATING_ENV)
        .ok()
        .is_some_and(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
}

#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct LoadLoraAdapterRequest {
    /// Request-facing adapter alias.
    #[schema(example = "production")]
    pub lora_name: String,
    /// Local server filesystem directory containing PEFT safetensors files.
    #[schema(example = "/srv/adapters/production-v2")]
    pub lora_path: String,
    /// Atomically replace an existing alias. Defaults to false, matching vLLM.
    #[serde(default, alias = "replace")]
    #[schema(default = false, example = false)]
    pub load_inplace: bool,
    /// Replace only if the alias still points at this generation.
    #[serde(default)]
    #[schema(value_type = Option<String>, example = "5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a")]
    pub expected_generation: Option<inference_core::AdapterGenerationId>,
    #[serde(default)]
    #[schema(ignore)]
    is_3d_lora_weight: bool,
    #[serde(default)]
    pub model: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct UnloadLoraAdapterRequest {
    pub lora_name: String,
    /// Remove only if the alias still points at this generation.
    #[serde(default)]
    #[schema(value_type = Option<String>, example = "5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a")]
    pub expected_generation: Option<inference_core::AdapterGenerationId>,
    /// Accepted for vLLM request compatibility; aliases are authoritative in inference.rs.
    #[serde(default)]
    pub lora_int_id: Option<u64>,
    #[serde(default)]
    pub model: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize, IntoParams)]
#[into_params(parameter_in = Query)]
pub struct ListLoraAdaptersQuery {
    #[serde(default)]
    pub model: Option<String>,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct LoraAdapterObject {
    pub id: String,
    pub object: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,
    pub generation: String,
    pub rank: usize,
    pub bytes: u64,
}

impl LoraAdapterObject {
    fn from_info(info: LoraAdapterInfo, expose_source: bool) -> Self {
        Self {
            id: info.alias,
            object: LORA_ADAPTER_OBJECT.to_string(),
            source: expose_source.then_some(info.source),
            revision: info.revision,
            generation: info.generation.to_string(),
            rank: info.rank,
            bytes: info.bytes,
        }
    }
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct LoraAdapterListResponse {
    pub object: String,
    pub data: Vec<LoraAdapterObject>,
    pub generations: Vec<LoraResidentGenerationObject>,
    pub resident_generations: usize,
    pub retired_generations: usize,
    pub resident_bytes: u64,
    pub max_adapters: usize,
    pub max_rank: usize,
    pub max_bytes: u64,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct LoraResidentGenerationObject {
    pub generation: String,
    pub aliases: Vec<String>,
    pub rank: usize,
    pub bytes: u64,
    pub retired: bool,
    pub active_leases: usize,
}

impl From<inference_core::LoraResidentGenerationInfo> for LoraResidentGenerationObject {
    fn from(info: inference_core::LoraResidentGenerationInfo) -> Self {
        Self {
            generation: info.generation.to_string(),
            aliases: info.aliases,
            rank: info.rank,
            bytes: info.bytes,
            retired: info.retired,
            active_leases: info.active_leases,
        }
    }
}

fn validate_alias(alias: &str) -> Result<(), ApiError> {
    if alias.trim().is_empty() {
        return Err(lora_error(
            ApiErrorKind::InvalidRequest,
            "invalid_lora_name",
            "lora_name must not be empty",
        ));
    }
    if alias.trim().len() > MAX_LORA_ALIAS_BYTES {
        return Err(lora_error(
            ApiErrorKind::InvalidRequest,
            "invalid_lora_name",
            format!("lora_name must not exceed {MAX_LORA_ALIAS_BYTES} bytes"),
        ));
    }
    Ok(())
}

fn lora_load_policy(request: &LoadLoraAdapterRequest) -> Result<LoraAdapterLoadPolicy, ApiError> {
    if request.expected_generation.is_some() && !request.load_inplace {
        return Err(lora_error(
            ApiErrorKind::InvalidRequest,
            "invalid_lora_load_policy",
            "expected_generation requires load_inplace=true",
        ));
    }
    Ok(match (request.load_inplace, request.expected_generation) {
        (false, None) => LoraAdapterLoadPolicy::Create,
        (true, None) => LoraAdapterLoadPolicy::Upsert,
        (true, Some(generation)) => LoraAdapterLoadPolicy::CompareAndSwap(generation),
        (false, Some(_)) => unreachable!(),
    })
}

fn ensure_updates_enabled(config: &LoraAdapterApiConfig) -> Result<(), ApiError> {
    if config.enabled() {
        return Ok(());
    }
    Err(lora_error(
        ApiErrorKind::Forbidden,
        "lora_updates_disabled",
        "runtime LoRA adapter loading and unloading are disabled",
    ))
}

fn core_error(error: inference_core::InferenceRsError) -> ApiError {
    ApiError::from_error(&error, ApiErrorKind::Internal)
}

/// Loads a PEFT adapter directory under `lora_name`; one load runs at a time and a second is rejected, not queued.
pub fn load_adapter<'a>(
    state: &'a SharedInferenceRsState,
    config: &'a LoraAdapterApiConfig,
    request: LoadLoraAdapterRequest,
) -> BoxFuture<'a, Result<LoraAdapterObject, ApiError>> {
    Box::pin(load_adapter_inner(state, config, request))
}

async fn load_adapter_inner(
    state: &SharedInferenceRsState,
    config: &LoraAdapterApiConfig,
    request: LoadLoraAdapterRequest,
) -> Result<LoraAdapterObject, ApiError> {
    ensure_updates_enabled(config)?;
    validate_alias(&request.lora_name)?;
    let policy = lora_load_policy(&request)?;
    let LoadLoraAdapterRequest {
        lora_name,
        lora_path,
        model,
        ..
    } = request;
    let load_permit = config.try_begin_load()?;
    let adapter_path = PathBuf::from(lora_path);
    let model = normalize_model_id(model);
    let (state, config) = (state.clone(), config.clone());
    // Its own task, so an abandoned caller cannot cancel a load halfway through.
    let operation = tokio::spawn(async move {
        let _load_permit = load_permit;
        let files = tokio::task::spawn_blocking(move || config.open_adapter_files(&adapter_path))
            .await
            .map_err(|error| adapter_task_error("adapter file validation", error))??;
        state
            .load_lora_adapter_files_with_policy(
                model.as_deref(),
                lora_name,
                files.into_runtime_files(),
                policy,
            )
            .await
            .map_err(core_error)
    });
    let info = operation
        .await
        .map_err(|error| adapter_task_error("adapter load", error))??;
    Ok(LoraAdapterObject::from_info(info, true))
}

pub fn unload_adapter<'a>(
    state: &'a SharedInferenceRsState,
    config: &'a LoraAdapterApiConfig,
    request: UnloadLoraAdapterRequest,
) -> BoxFuture<'a, Result<LoraAdapterObject, ApiError>> {
    Box::pin(unload_adapter_inner(state, config, request))
}

async fn unload_adapter_inner(
    state: &SharedInferenceRsState,
    config: &LoraAdapterApiConfig,
    request: UnloadLoraAdapterRequest,
) -> Result<LoraAdapterObject, ApiError> {
    ensure_updates_enabled(config)?;
    let UnloadLoraAdapterRequest {
        lora_name,
        expected_generation,
        lora_int_id: _,
        model,
    } = request;
    validate_alias(&lora_name)?;
    let model = normalize_model_id(model);
    let info = state
        .unload_lora_adapter_if_generation(model.as_deref(), &lora_name, expected_generation)
        .await
        .map_err(core_error)?;
    Ok(LoraAdapterObject::from_info(info, true))
}

/// The loaded adapters and the runtime's capacity; adapter sources are shown only when updates are enabled.
pub fn list_adapters<'a>(
    state: &'a SharedInferenceRsState,
    config: &'a LoraAdapterApiConfig,
    query: ListLoraAdaptersQuery,
) -> BoxFuture<'a, Result<LoraAdapterListResponse, ApiError>> {
    Box::pin(list_adapters_inner(state, config, query))
}

async fn list_adapters_inner(
    state: &SharedInferenceRsState,
    config: &LoraAdapterApiConfig,
    query: ListLoraAdaptersQuery,
) -> Result<LoraAdapterListResponse, ApiError> {
    let model = normalize_model_id(query.model);
    let status = state
        .lora_adapter_status(model.as_deref())
        .await
        .map_err(core_error)?;
    let data = status
        .adapters
        .into_iter()
        .map(|info| LoraAdapterObject::from_info(info, config.enabled()))
        .collect();
    let generations = status.generations.into_iter().map(Into::into).collect();
    Ok(LoraAdapterListResponse {
        object: LORA_ADAPTER_LIST_OBJECT.to_string(),
        data,
        generations,
        resident_generations: status.resident_generations,
        retired_generations: status.retired_generations,
        resident_bytes: status.resident_bytes,
        max_adapters: status.limits.max_adapters,
        max_rank: status.limits.max_rank,
        max_bytes: status.limits.max_bytes,
    })
}

#[cfg(test)]
mod tests {
    use std::io::Read;

    use inference_core::{InferenceRsError, LoraAdapterError};

    use super::*;

    fn lora_core_error(error: LoraAdapterError) -> ApiError {
        core_error(InferenceRsError::LoraAdapter(error))
    }

    #[test]
    fn adapter_objects_expose_their_source_only_when_allowed() {
        let info = LoraAdapterInfo {
            alias: "math".to_string(),
            source: "source".to_string(),
            revision: None,
            generation: inference_core::AdapterGenerationId::from_bytes([3; 32]),
            rank: 8,
            bytes: 16,
        };
        assert!(
            LoraAdapterObject::from_info(info.clone(), false)
                .source
                .is_none()
        );
        assert_eq!(
            LoraAdapterObject::from_info(info, true).source.as_deref(),
            Some("source")
        );
    }

    #[test]
    fn adapter_root_blocks_sibling_paths() {
        let temp = tempfile::tempdir().unwrap();
        let allowed = temp.path().join("allowed");
        let sibling = temp.path().join("sibling");
        std::fs::create_dir(&allowed).unwrap();
        std::fs::create_dir(&sibling).unwrap();
        std::fs::File::create(allowed.join(LORA_CONFIG_FILE)).unwrap();
        std::fs::File::create(allowed.join(LORA_WEIGHTS_FILE)).unwrap();
        let config = LoraAdapterApiConfig::default()
            .with_enabled(true)
            .with_allowed_root(&allowed)
            .prepare()
            .unwrap();

        assert!(config.open_adapter_files(&allowed).is_ok());
        assert_eq!(
            config.open_adapter_files(&sibling).unwrap_err().kind,
            ApiErrorKind::Forbidden
        );
    }

    #[cfg(unix)]
    #[test]
    fn adapter_root_blocks_files_resolving_outside_root() {
        use std::os::unix::fs::symlink;

        let temp = tempfile::tempdir().unwrap();
        let allowed = temp.path().join("allowed");
        let adapter = allowed.join("adapter");
        let outside_config = temp.path().join("adapter_config.json");
        std::fs::create_dir(&allowed).unwrap();
        std::fs::create_dir(&adapter).unwrap();
        std::fs::File::create(&outside_config).unwrap();
        std::fs::File::create(adapter.join(LORA_WEIGHTS_FILE)).unwrap();
        symlink(outside_config, adapter.join(LORA_CONFIG_FILE)).unwrap();
        let config = LoraAdapterApiConfig::default()
            .with_enabled(true)
            .with_allowed_root(allowed)
            .prepare()
            .unwrap();

        assert_eq!(
            config.open_adapter_files(&adapter).unwrap_err().kind,
            ApiErrorKind::Forbidden
        );
    }

    #[test]
    fn adapter_loading_keeps_the_validated_file_handles() {
        let temp = tempfile::tempdir().unwrap();
        let adapter = temp.path().join("adapter");
        std::fs::create_dir(&adapter).unwrap();
        std::fs::write(adapter.join(LORA_CONFIG_FILE), b"original-config").unwrap();
        std::fs::write(adapter.join(LORA_WEIGHTS_FILE), b"original-weights").unwrap();
        let config = LoraAdapterApiConfig::default()
            .with_enabled(true)
            .with_allowed_root(temp.path())
            .prepare()
            .unwrap();
        let mut opened = config.open_adapter_files(&adapter).unwrap();

        std::fs::rename(
            adapter.join(LORA_CONFIG_FILE),
            adapter.join("old-adapter-config.json"),
        )
        .unwrap();
        std::fs::write(adapter.join(LORA_CONFIG_FILE), b"replacement-config").unwrap();
        let mut contents = String::new();
        opened.config.read_to_string(&mut contents).unwrap();
        assert_eq!(contents, "original-config");
    }

    #[test]
    fn relative_adapter_paths_resolve_under_the_allowed_root() {
        let temp = tempfile::tempdir().unwrap();
        let adapter = temp.path().join("production");
        std::fs::create_dir(&adapter).unwrap();
        std::fs::File::create(adapter.join(LORA_CONFIG_FILE)).unwrap();
        std::fs::File::create(adapter.join(LORA_WEIGHTS_FILE)).unwrap();
        let config = LoraAdapterApiConfig::default()
            .with_enabled(true)
            .with_allowed_root(temp.path())
            .prepare()
            .unwrap();

        assert!(config.open_adapter_files(Path::new("production")).is_ok());
    }

    #[test]
    fn lifecycle_requests_accept_vllm_fields_and_reject_typos() {
        let request: LoadLoraAdapterRequest = serde_json::from_value(serde_json::json!({
            "lora_name": "production",
            "lora_path": "production-v2",
            "load_inplace": true,
            "expected_generation": "0707070707070707070707070707070707070707070707070707070707070707",
            "is_3d_lora_weight": true
        }))
        .unwrap();
        assert!(request.load_inplace);
        assert!(request.is_3d_lora_weight);
        assert_eq!(
            lora_load_policy(&request).unwrap(),
            LoraAdapterLoadPolicy::CompareAndSwap(inference_core::AdapterGenerationId::from_bytes(
                [7; 32]
            ))
        );
        assert_eq!(
            request.expected_generation,
            Some(inference_core::AdapterGenerationId::from_bytes([7; 32]))
        );
        assert!(
            serde_json::from_value::<LoadLoraAdapterRequest>(serde_json::json!({
                "lora_name": "production",
                "lora_path": "production-v2",
                "load_inpalce": true
            }))
            .is_err()
        );
    }

    #[test]
    fn load_admission_is_shared_and_non_queueing() {
        let config = LoraAdapterApiConfig::default();
        let permit = config.try_begin_load().unwrap();
        assert_eq!(
            config.clone().try_begin_load().unwrap_err().kind,
            ApiErrorKind::RateLimited
        );
        drop(permit);
        assert!(config.try_begin_load().is_ok());
    }

    #[test]
    fn default_and_empty_model_ids_select_the_default_model() {
        assert_eq!(normalize_model_id(None), None);
        assert_eq!(normalize_model_id(Some(String::new())), None);
        assert_eq!(normalize_model_id(Some(" default ".to_string())), None);
        assert_eq!(
            normalize_model_id(Some(" model ".to_string())),
            Some("model".to_string())
        );
    }

    #[test]
    fn core_errors_map_to_stable_kinds() {
        assert_eq!(
            core_error(InferenceRsError::ModelNotFound("model".to_string())).kind,
            ApiErrorKind::NotFound
        );
        assert_eq!(
            lora_core_error(LoraAdapterError::InvalidAlias).kind,
            ApiErrorKind::InvalidRequest
        );
        assert_eq!(
            lora_core_error(LoraAdapterError::AdapterLimit { max: 1 }).kind,
            ApiErrorKind::Conflict
        );
        assert_eq!(
            lora_core_error(LoraAdapterError::AliasLimit { max: 1 }).kind,
            ApiErrorKind::Conflict
        );
        assert_eq!(
            lora_core_error(LoraAdapterError::AliasTooLong {
                bytes: MAX_LORA_ALIAS_BYTES + 1,
                max: MAX_LORA_ALIAS_BYTES,
            })
            .kind,
            ApiErrorKind::InvalidRequest
        );
        assert_eq!(
            lora_core_error(LoraAdapterError::FileTooLarge {
                path: PathBuf::from("adapter_model.safetensors"),
                bytes: 2,
                max: 1,
            })
            .kind,
            ApiErrorKind::PayloadTooLarge
        );
        assert_eq!(
            lora_core_error(LoraAdapterError::LoadBusy).kind,
            ApiErrorKind::RateLimited
        );
        assert_eq!(
            lora_core_error(LoraAdapterError::AlreadyLoaded {
                alias: "production".to_string(),
                generation: inference_core::AdapterGenerationId::from_bytes([1; 32]),
            })
            .kind,
            ApiErrorKind::Conflict
        );
        assert_eq!(
            adapter_filesystem_error(
                "adapter file `adapter_model.safetensors`",
                std::io::Error::other("disk failure"),
            )
            .kind,
            ApiErrorKind::Internal
        );
        assert_eq!(
            adapter_filesystem_error(
                "adapter directory `/missing`",
                std::io::Error::from(std::io::ErrorKind::NotFound),
            )
            .kind,
            ApiErrorKind::NotFound
        );
        assert_eq!(
            adapter_filesystem_error(
                "adapter directory `/forbidden`",
                std::io::Error::from(std::io::ErrorKind::PermissionDenied),
            )
            .kind,
            ApiErrorKind::Forbidden
        );
        assert_eq!(
            adapter_filesystem_error(
                "adapter file `adapter_model.safetensors`",
                std::io::Error::from(std::io::ErrorKind::NotADirectory),
            )
            .kind,
            ApiErrorKind::InvalidRequest
        );
    }
}
