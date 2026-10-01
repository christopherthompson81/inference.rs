//! Engine-independent facts about the host: devices and build, diagnostics, the Hugging Face cache, auto-tuning.

use inference_core::{AutoDeviceMapParams, ModelDType, TokenSource, parse_isq_value};
pub use inference_core::{hf_hub_cache_dir, hf_token_path};
pub use inference_selection::{
    AutoTuneRequest, AutoTuneResult, DoctorReport, DoctorStatus, FitStatus, QualityTier,
    SystemInfo, TuneProfile, auto_tune,
};
use inference_selection::{ModelSelected, collect_system_info, run_doctor};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::api_error::{ApiError, ApiErrorKind};
use crate::request_body::{JsonRequest, parse_json};

pub fn system_info() -> SystemInfo {
    collect_system_info()
}

pub fn system_doctor() -> DoctorReport {
    run_doctor()
}

pub fn system_info_json() -> Result<String, ApiError> {
    serde_json::to_string(&system_info()).map_err(|_| ApiError::internal())
}

pub fn system_doctor_json() -> Result<String, ApiError> {
    serde_json::to_string(&system_doctor()).map_err(|_| ApiError::internal())
}

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

fn invalid_field(message: String, code: &'static str, param: &'static str) -> ApiError {
    ApiError::new(
        ApiErrorKind::InvalidRequest,
        message,
        Some(code),
        Some(param),
    )
}

/// Picks the quantization and settings that fit `request`'s model on this machine, without loading it.
pub fn tune_model(request: TuneModelRequest) -> Result<AutoTuneResult, ApiError> {
    let token_source = match request.token_source {
        Some(value) => value.parse().map_err(|error| {
            invalid_field(
                format!("Invalid token_source: {error}"),
                "invalid_token_source",
                "token_source",
            )
        })?,
        None => TokenSource::CacheToken,
    };
    let dtype = request
        .dtype
        .as_deref()
        .unwrap_or("auto")
        .parse::<ModelDType>()
        .map_err(|error| {
            invalid_field(format!("Invalid dtype: {error}"), "invalid_dtype", "dtype")
        })?;
    let requested_isq = request
        .requested_isq
        .map(|value| {
            parse_isq_value(&value, None).map_err(|error| {
                invalid_field(
                    format!("Invalid isq value: {error}"),
                    "invalid_isq",
                    "requested_isq",
                )
            })
        })
        .transpose()?;
    let model = ModelSelected::Run {
        model_id: request.model_id,
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
        max_seq_len: request
            .max_seq_len
            .unwrap_or(AutoDeviceMapParams::DEFAULT_MAX_SEQ_LEN),
        max_batch_size: request
            .max_batch_size
            .unwrap_or(AutoDeviceMapParams::DEFAULT_MAX_BATCH_SIZE),
        max_num_images: request.max_num_images,
        max_image_length: request.max_image_length,
        hf_cache_path: None,
        matformer_config_path: None,
        matformer_slice_name: None,
    };
    auto_tune(AutoTuneRequest {
        model,
        token_source,
        hf_revision: request.hf_revision,
        force_cpu: request.cpu.unwrap_or(false),
        profile: request
            .profile
            .map(Into::into)
            .unwrap_or(TuneProfile::Balanced),
        requested_isq,
    })
    .map_err(|error| {
        tracing::error!(%error, "model auto-tuning failed");
        ApiError::internal()
    })
}
