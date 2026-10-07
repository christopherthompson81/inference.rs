pub(crate) mod auto_device_map;
mod checkpoint_inventory;
mod diffusion_loaders;
mod embedding_loaders;
mod multimodal_loaders;
mod normal_loaders;
pub use crate::device_map::AutoDeviceMapParams;
pub(crate) use checkpoint_inventory::{checkpoint_device_map_sizes, checkpoint_runtime_size};
use inference_nn::loaders::NonMappedSubModel;
pub use inference_nn::loaders::{AutoDeviceMapQuantization, DeviceMappedModelLoader};
pub(crate) use inference_nn::loaders::{QK_ROPE_LAYOUT_CONFIG_KEY, qk_rope_layout_from_config};

use std::{
    fmt::{self, Debug},
    path::PathBuf,
    str::FromStr,
    sync::Arc,
};

use anyhow::Result;
use as_any::AsAny;
use inference_quant::{IsqType, QuantizedConfig};
use inference_tensor::{DType, Device};
use serde::Deserialize;
use tokio::sync::Mutex;

#[cfg(feature = "models-gemma")]
pub use normal_loaders::GemmaLoader;
pub(crate) use normal_loaders::NormalLoaderTypeExt;
#[cfg(feature = "models-qwen")]
pub use normal_loaders::Qwen2Loader;
#[cfg(feature = "models-other")]
pub use normal_loaders::Starcoder2Loader;
pub use normal_loaders::{
    AutoNormalLoader, NormalLoaderType, NormalLoadingMetadata, NormalModel, NormalModelLoader,
};
#[cfg(feature = "models-llama")]
pub use normal_loaders::{LlamaLoader, MistralLoader, MixtralLoader};
#[cfg(feature = "models-phi")]
pub use normal_loaders::{Phi2Loader, Phi3Loader};

#[cfg(feature = "models-phi")]
pub use multimodal_loaders::Phi3VLoader;
pub use multimodal_loaders::{
    AutoMultimodalLoader, MultimodalLoaderType, MultimodalModel, MultimodalModelLoader,
};
#[cfg(feature = "models-llama")]
pub use multimodal_loaders::{Idefics2Loader, LLaVALoader, LLaVANextLoader};
pub(crate) use multimodal_loaders::{MultimodalLoaderTypeExt, MultimodalProcessorFactory};

pub use embedding_loaders::{
    AutoEmbeddingLoader, EmbeddingLoaderType, EmbeddingModel, EmbeddingModelLoader,
    EmbeddingModule, EmbeddingModulePaths, EmbeddingModuleType,
};

pub use diffusion_loaders::{
    DiffusionLoad, DiffusionLoaderType, DiffusionModel, DiffusionModelLoader, DiffusionModelPaths,
    DiffusionModelPathsInner, FluxLoader,
};

use crate::{DeviceMapSetting, PagedAttentionConfig, TryIntoDType};

use super::{Pipeline, paths::AdapterPaths};

const LEGACY_MODEL_OPT_CONFIG: &str = "hf_quant_config.json";
/// Set on the model config JSON when the checkpoint's built-in MTP head should be loaded.
pub const MTP_CONFIG_KEY: &str = "_inference_mtp";

pub(crate) fn load_model_config(
    config_path: &std::path::Path,
    use_checkpoint_quantization: bool,
) -> Result<String> {
    let config = std::fs::read_to_string(config_path)?;
    if !use_checkpoint_quantization {
        return Ok(config);
    }
    let config = normalize_compression_config(&config)?;
    let Some(parent) = config_path.parent() else {
        return Ok(config);
    };
    let model_opt_path = parent.join(LEGACY_MODEL_OPT_CONFIG);
    if !model_opt_path.is_file() {
        return Ok(config);
    }
    let model_opt = std::fs::read_to_string(model_opt_path)?;
    inject_legacy_model_opt_config(&config, &model_opt)
}

fn normalize_compression_config(config: &str) -> Result<String> {
    let mut value: serde_json::Value = serde_json::from_str(config)?;
    if canonicalize_quantization_config(&mut value)? {
        Ok(serde_json::to_string(&value)?)
    } else {
        Ok(config.to_string())
    }
}

fn canonicalize_quantization_config(value: &mut serde_json::Value) -> Result<bool> {
    let object = value
        .as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("model config must be a JSON object"))?;
    let root_quantization = object
        .get("quantization_config")
        .filter(|value| !value.is_null())
        .cloned();
    let text_quantization = object
        .get("text_config")
        .and_then(serde_json::Value::as_object)
        .and_then(|text| text.get("quantization_config"))
        .filter(|value| !value.is_null())
        .cloned();
    let compression = object
        .get("compression_config")
        .filter(|value| !value.is_null())
        .cloned();
    let text_compression = object
        .get("text_config")
        .and_then(serde_json::Value::as_object)
        .and_then(|text| text.get("compression_config"))
        .filter(|value| !value.is_null())
        .cloned();
    let Some(effective) = root_quantization
        .or(text_quantization)
        .or(compression)
        .or(text_compression)
    else {
        return Ok(false);
    };

    let mut changed = false;
    if object
        .get("quantization_config")
        .is_none_or(serde_json::Value::is_null)
    {
        object.insert("quantization_config".to_string(), effective.clone());
        changed = true;
    }
    if let Some(text) = object
        .get_mut("text_config")
        .and_then(serde_json::Value::as_object_mut)
        && text.get("quantization_config") != Some(&effective)
    {
        text.insert("quantization_config".to_string(), effective);
        changed = true;
    }
    Ok(changed)
}

fn inject_legacy_model_opt_config(config: &str, model_opt: &str) -> Result<String> {
    let mut config_value: serde_json::Value = serde_json::from_str(config)?;
    let canonicalized = canonicalize_quantization_config(&mut config_value)?;
    let config_object = config_value
        .as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("model config must be a JSON object"))?;
    let embedded = config_object
        .get("quantization_config")
        .filter(|value| !value.is_null())
        .cloned();
    if embedded.as_ref().is_some_and(|value| {
        !value
            .get("quant_method")
            .and_then(serde_json::Value::as_str)
            .is_some_and(|method| method.to_ascii_lowercase().starts_with("modelopt"))
    }) {
        return if canonicalized {
            Ok(serde_json::to_string(&config_value)?)
        } else {
            Ok(config.to_string())
        };
    }

    let model_opt_value: serde_json::Value = serde_json::from_str(model_opt)?;
    let quantization = match embedded.as_ref() {
        Some(embedded) => QuantizedConfig::from_modelopt_configs(Some(embedded), &model_opt_value),
        None => QuantizedConfig::from_modelopt_config(&model_opt_value),
    }
    .map_err(anyhow::Error::msg)?;
    config_object.insert(
        "quantization_config".to_string(),
        serde_json::to_value(quantization)?,
    );
    canonicalize_quantization_config(&mut config_value)?;
    Ok(serde_json::to_string(&config_value)?)
}

pub(crate) fn inject_mtp_config_flag(config: &str) -> anyhow::Result<String> {
    let mut config: serde_json::Value = serde_json::from_str(config)?;
    let object = config
        .as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("model config must be a JSON object"))?;
    object.insert(MTP_CONFIG_KEY.to_string(), serde_json::Value::Bool(true));
    Ok(config.to_string())
}

pub(crate) fn stamp_qk_rope_layout(
    config: &str,
    layout: crate::gguf::normal_registry::RopePairing,
) -> Result<String> {
    let mut config: serde_json::Value = serde_json::from_str(config)?;
    let object = config
        .as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("model config must be a JSON object"))?;
    object.insert(
        QK_ROPE_LAYOUT_CONFIG_KEY.to_string(),
        serde_json::Value::String(
            match layout {
                crate::gguf::normal_registry::RopePairing::Adjacent => "adjacent",
                crate::gguf::normal_registry::RopePairing::HalfSplit => "half_split",
            }
            .to_string(),
        ),
    );
    Ok(serde_json::to_string(&config)?)
}

pub(crate) fn validate_lora_qk_rope_layout(config: &str, has_adapter: bool) -> Result<()> {
    if has_adapter
        && qk_rope_layout_from_config(config)?
            == Some(crate::gguf::normal_registry::RopePairing::Adjacent)
    {
        anyhow::bail!(
            "LoRA adapters are not supported on a multimodal model whose Q/K tensors use adjacent RoPE layout; load the original safetensors model or omit the adapter"
        );
    }
    Ok(())
}

/// `ModelPaths` abstracts the mechanism to get all necessary files for running a model. For
/// example `LocalModelPaths` implements `ModelPaths` when all files are in the local file system.
pub trait ModelPaths: AsAny + Debug + Send + Sync {
    /// Model weights files (multiple files supported).
    fn get_weight_filenames(&self) -> &[PathBuf];

    /// Retrieve the [`PretrainedConfig`] file.
    ///
    /// [`PretrainedConfig`]: https://huggingface.co/docs/transformers/v4.40.2/en/main_classes/configuration#transformers.PretrainedConfig
    fn get_config_filename(&self) -> &PathBuf;

    /// A serialised [`tokenizers.Tokenizer`] HuggingFace object.
    ///
    /// [`tokenizers.Tokenizer`]: https://huggingface.co/docs/transformers/v4.40.2/en/main_classes/tokenizer
    fn get_tokenizer_filename(&self) -> &PathBuf;

    /// File where the content is expected to deserialize to [`ChatTemplate`].
    ///
    /// [`ChatTemplate`]: crate::ChatTemplate
    fn get_template_filename(&self) -> &Option<PathBuf>;

    /// Filepath for general model configuration.
    fn get_gen_conf_filename(&self) -> Option<&PathBuf>;

    /// Get the preprocessor config (for the multimodal models). This is used to pre process images.
    fn get_preprocessor_config(&self) -> &Option<PathBuf>;

    /// Get the video preprocessor config, for multimodal models that ship separate video settings.
    fn get_video_preprocessor_config(&self) -> Option<&PathBuf> {
        None
    }

    /// Get the processor config (for the multimodal models). This is primarily used for the chat template.
    fn get_processor_config(&self) -> &Option<PathBuf>;

    /// Get the explicit chat template.
    fn get_chat_template_explicit(&self) -> &Option<PathBuf>;

    /// Get adapter paths.
    fn get_adapter_paths(&self) -> &AdapterPaths;

    /// Get embedding model `modules.json` compatible with sentence-transformers
    fn get_modules(&self) -> Option<&[EmbeddingModulePaths]>;
}

#[derive(Clone, Debug)]
/// All local paths and metadata necessary to load a model.
pub struct LocalModelPaths<P: Debug> {
    pub tokenizer_filename: P,
    pub config_filename: P,
    pub template_filename: Option<P>,
    pub filenames: Vec<P>,
    pub adapter_paths: AdapterPaths,
    pub gen_conf: Option<P>,
    pub preprocessor_config: Option<P>,
    pub video_preprocessor_config: Option<P>,
    pub processor_config: Option<P>,
    pub chat_template_json_filename: Option<P>,
}

impl<P: Debug> LocalModelPaths<P> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        tokenizer_filename: P,
        config_filename: P,
        template_filename: P,
        filenames: Vec<P>,
        adapter_paths: AdapterPaths,
        gen_conf: Option<P>,
        preprocessor_config: Option<P>,
        processor_config: Option<P>,
        chat_template_json_filename: Option<P>,
    ) -> Self {
        Self {
            tokenizer_filename,
            config_filename,
            template_filename: Some(template_filename),
            filenames,
            adapter_paths,
            gen_conf,
            preprocessor_config,
            video_preprocessor_config: None,
            processor_config,
            chat_template_json_filename,
        }
    }
}

impl ModelPaths for LocalModelPaths<PathBuf> {
    fn get_config_filename(&self) -> &PathBuf {
        &self.config_filename
    }
    fn get_tokenizer_filename(&self) -> &PathBuf {
        &self.tokenizer_filename
    }
    fn get_weight_filenames(&self) -> &[PathBuf] {
        &self.filenames
    }
    fn get_template_filename(&self) -> &Option<PathBuf> {
        &self.template_filename
    }
    fn get_gen_conf_filename(&self) -> Option<&PathBuf> {
        self.gen_conf.as_ref()
    }
    fn get_preprocessor_config(&self) -> &Option<PathBuf> {
        &self.preprocessor_config
    }
    fn get_video_preprocessor_config(&self) -> Option<&PathBuf> {
        self.video_preprocessor_config.as_ref()
    }
    fn get_processor_config(&self) -> &Option<PathBuf> {
        &self.processor_config
    }
    fn get_chat_template_explicit(&self) -> &Option<PathBuf> {
        &self.chat_template_json_filename
    }
    fn get_adapter_paths(&self) -> &AdapterPaths {
        &self.adapter_paths
    }
    fn get_modules(&self) -> Option<&[EmbeddingModulePaths]> {
        None
    }
}

#[derive(Clone, Debug)]
/// All local paths and metadata necessary to load an embedding model.
pub struct EmbeddingModelPaths<P: Debug> {
    pub tokenizer_filename: P,
    pub config_filename: P,
    pub modules: Vec<EmbeddingModulePaths>,
    pub filenames: Vec<P>,
    pub adapter_paths: AdapterPaths,
}

impl<P: Debug> EmbeddingModelPaths<P> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        tokenizer_filename: P,
        config_filename: P,
        filenames: Vec<P>,
        adapter_paths: AdapterPaths,
        modules: Vec<EmbeddingModulePaths>,
    ) -> Self {
        Self {
            tokenizer_filename,
            config_filename,
            filenames,
            adapter_paths,
            modules,
        }
    }
}

impl ModelPaths for EmbeddingModelPaths<PathBuf> {
    fn get_config_filename(&self) -> &PathBuf {
        &self.config_filename
    }
    fn get_tokenizer_filename(&self) -> &PathBuf {
        &self.tokenizer_filename
    }
    fn get_weight_filenames(&self) -> &[PathBuf] {
        &self.filenames
    }
    fn get_template_filename(&self) -> &Option<PathBuf> {
        &None
    }
    fn get_gen_conf_filename(&self) -> Option<&PathBuf> {
        None
    }
    fn get_preprocessor_config(&self) -> &Option<PathBuf> {
        &None
    }
    fn get_processor_config(&self) -> &Option<PathBuf> {
        &None
    }
    fn get_chat_template_explicit(&self) -> &Option<PathBuf> {
        &None
    }
    fn get_adapter_paths(&self) -> &AdapterPaths {
        &self.adapter_paths
    }
    fn get_modules(&self) -> Option<&[EmbeddingModulePaths]> {
        Some(&self.modules)
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
/// The source of the HF token.
pub enum TokenSource {
    Literal(String),
    EnvVar(String),
    Path(String),
    CacheToken,
    None,
}

impl FromStr for TokenSource {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parts: Vec<&str> = s.splitn(2, ':').collect();
        match parts[0] {
            "literal" => parts
                .get(1)
                .map(|&value| TokenSource::Literal(value.to_string()))
                .ok_or_else(|| "Expected a value for 'literal'".to_string()),
            "env" => Ok(TokenSource::EnvVar(
                parts
                    .get(1)
                    .unwrap_or(&"HUGGING_FACE_HUB_TOKEN")
                    .to_string(),
            )),
            "path" => parts
                .get(1)
                .map(|&value| TokenSource::Path(value.to_string()))
                .ok_or_else(|| "Expected a value for 'path'".to_string()),
            "cache" => Ok(TokenSource::CacheToken),
            "none" => Ok(TokenSource::None),
            _ => Err("Invalid token source format".to_string()),
        }
    }
}

impl fmt::Display for TokenSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenSource::Literal(value) => write!(f, "literal:{value}"),
            TokenSource::EnvVar(value) => write!(f, "env:{value}"),
            TokenSource::Path(value) => write!(f, "path:{value}"),
            TokenSource::CacheToken => write!(f, "cache"),
            TokenSource::None => write!(f, "none"),
        }
    }
}

/// The kind of model to build.
#[derive(Clone, Default, strum::Display)]
pub enum ModelKind {
    #[default]
    #[strum(to_string = "normal (no adapters)")]
    Normal,

    #[strum(to_string = "gguf quantized from {quant} (no adapters)")]
    GgufQuantized { quant: QuantizationKind },

    #[strum(to_string = "lora")]
    Lora,

    #[strum(to_string = "lora, gguf quantized from {quant}")]
    GgufLora { quant: QuantizationKind },

    #[strum(to_string = "anymoe: target: `{target}`")]
    AnyMoe { target: Box<ModelKind> },
}

#[derive(Clone, Copy, strum::Display, strum::EnumIs)]
#[strum(serialize_all = "kebab-case")]
pub enum QuantizationKind {
    /// GGML
    Ggml,
    /// GGUF
    Gguf,
    /// GPTQ
    Gptq,
}

impl ModelKind {
    pub fn quantized_kind(&self) -> Vec<Option<QuantizationKind>> {
        use ModelKind::*;

        match self {
            Normal | Lora => vec![None],
            GgufQuantized { quant } | GgufLora { quant } => vec![Some(*quant)],
            AnyMoe { target } => target.quantized_kind(),
        }
    }
}

#[derive(Deserialize)]
pub struct QuantizationConfigShim {
    quantization_config: Option<QuantizedConfig>,
}

impl QuantizationConfigShim {
    pub fn get_quant_config_pack_factor(config: &str, dtype: DType) -> Result<usize> {
        let QuantizationConfigShim {
            quantization_config,
        } = serde_json::from_str(config)?;

        if let Some(quantization_config) = quantization_config {
            Ok(quantization_config.pack_factor(dtype))
        } else {
            Ok(1)
        }
    }
}

/// The `Loader` trait abstracts the loading process. The primary entrypoint is the
/// `load_model` method.
///
/// # Example
/// ```no_run
/// use inference_core::{Loader, TokenSource, DeviceMapSetting, AutoDeviceMapParams, ModelDType};
/// use inference_tensor::Device;
///
/// let loader: Box<dyn Loader> = todo!();
/// let pipeline = loader.load_model_from_hf(
///     None,
///     TokenSource::CacheToken,
///     &ModelDType::Auto,
///     &Device::cuda_if_available(0).unwrap(),
///     false,
///     DeviceMapSetting::Auto(AutoDeviceMapParams::default_text()),
///     None,
///     None,
/// ).unwrap();
/// ```
pub trait Loader: Send + Sync {
    /// If `revision` is None, then it defaults to `main`.
    /// If `dtype` is None, then it defaults to the model default (usually BF16).
    /// If model is not found on HF, will attempt to resolve locally.
    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    fn load_model_from_hf(
        &self,
        revision: Option<String>,
        token_source: TokenSource,
        dtype: &dyn TryIntoDType,
        device: &Device,
        silent: bool,
        mapper: DeviceMapSetting,
        in_situ_quant: Option<IsqType>,
        paged_attn_config: Option<PagedAttentionConfig>,
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>>;

    /// Load a model from the specified paths.
    /// Also initializes `DEBUG`.
    #[allow(
        clippy::type_complexity,
        clippy::too_many_arguments,
        clippy::borrowed_box
    )]
    fn load_model_from_path(
        &self,
        paths: &dyn ModelPaths,
        dtype: &dyn TryIntoDType,
        device: &Device,
        silent: bool,
        mapper: DeviceMapSetting,
        in_situ_quant: Option<IsqType>,
        paged_attn_config: Option<PagedAttentionConfig>,
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>>;

    fn get_id(&self) -> String;
    fn get_kind(&self) -> ModelKind;
}

#[cfg(test)]
mod model_config_tests {
    use super::*;

    #[test]
    fn legacy_model_opt_config_is_embedded() -> Result<()> {
        let config = r#"{"model_type":"llama"}"#;
        let legacy = r#"{
            "producer":{"name":"modelopt","version":"0.19.0"},
            "quantization":{"quant_algo":"FP8","exclude_modules":["lm_head"]}
        }"#;
        let merged: serde_json::Value =
            serde_json::from_str(&inject_legacy_model_opt_config(config, legacy)?)?;
        let quantization = &merged["quantization_config"];
        assert_eq!(quantization["quant_method"], "modelopt");
        assert_eq!(quantization["quantization"]["quant_algo"], "FP8");
        assert_eq!(merged["model_type"], "llama");
        assert_eq!(
            QuantizationConfigShim::get_quant_config_pack_factor(&merged.to_string(), DType::BF16)?,
            IsqType::F8E4M3.pack_factor(DType::BF16)
        );
        Ok(())
    }

    #[test]
    fn compression_config_alias_is_loaded_with_explicit_precedence() -> Result<()> {
        let compressed = serde_json::json!({
            "format": "float-quantized",
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "weights": {
                        "dynamic": false,
                        "num_bits": 8,
                        "strategy": "tensor",
                        "symmetric": true,
                        "type": "float"
                    }
                }
            }
        });
        let dir = tempfile::tempdir()?;
        let config_path = dir.path().join("config.json");
        std::fs::write(
            &config_path,
            serde_json::to_vec(&serde_json::json!({
                "model_type": "llama",
                "compression_config": compressed
            }))?,
        )?;
        let loaded: serde_json::Value =
            serde_json::from_str(&load_model_config(&config_path, true)?)?;
        let quantized: QuantizedConfig =
            serde_json::from_value(loaded["quantization_config"].clone())?;
        assert!(matches!(
            quantized,
            QuantizedConfig::CompressedTensors { .. }
        ));

        let explicit = serde_json::json!({"quant_method": "fp8"});
        let normalized: serde_json::Value = serde_json::from_str(&normalize_compression_config(
            &serde_json::json!({
                "quantization_config": explicit,
                "compression_config": compressed,
                "text_config": {"compression_config": compressed}
            })
            .to_string(),
        )?)?;
        assert_eq!(normalized["quantization_config"], explicit);
        assert_eq!(normalized["text_config"]["quantization_config"], explicit);

        let conflicting = serde_json::json!({"quant_method": "fp8", "activation_scheme": "static"});
        let normalized: serde_json::Value = serde_json::from_str(&normalize_compression_config(
            &serde_json::json!({
                "quantization_config": explicit,
                "text_config": {"quantization_config": conflicting}
            })
            .to_string(),
        )?)?;
        assert_eq!(normalized["quantization_config"], explicit);
        assert_eq!(normalized["text_config"]["quantization_config"], explicit);

        let nested = serde_json::json!({"quant_method": "fp8", "activation_scheme": "dynamic"});
        let normalized: serde_json::Value = serde_json::from_str(&normalize_compression_config(
            &serde_json::json!({
                "compression_config": compressed,
                "text_config": {"quantization_config": nested}
            })
            .to_string(),
        )?)?;
        assert_eq!(normalized["quantization_config"], nested);
        assert_eq!(normalized["text_config"]["quantization_config"], nested);

        let normalized: serde_json::Value = serde_json::from_str(&normalize_compression_config(
            &serde_json::json!({
                "text_config": {"compression_config": compressed}
            })
            .to_string(),
        )?)?;
        assert_eq!(
            normalized["quantization_config"]["format"],
            "float-quantized"
        );
        assert_eq!(
            normalized["text_config"]["quantization_config"],
            normalized["quantization_config"]
        );
        let merged: serde_json::Value = serde_json::from_str(&inject_legacy_model_opt_config(
            &serde_json::json!({
                "text_config": {"compression_config": compressed}
            })
            .to_string(),
            r#"{"quantization":{"quant_algo":"FP8"}}"#,
        )?)?;
        assert_eq!(merged["quantization_config"]["format"], "float-quantized");
        Ok(())
    }

    #[test]
    fn uqff_config_loading_skips_checkpoint_quantization_metadata() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let config_path = dir.path().join("config.json");
        let config = r#"{"model_type":"llama","compression_config":{"unsupported":true}}"#;
        std::fs::write(&config_path, config)?;
        std::fs::write(dir.path().join(LEGACY_MODEL_OPT_CONFIG), "not json")?;
        assert_eq!(load_model_config(&config_path, false)?, config);
        Ok(())
    }

    #[test]
    fn embedded_quantization_config_wins_over_legacy_model_opt() -> Result<()> {
        let config = r#"{"quantization_config":{"quant_method":"fp8"}}"#;
        let legacy = r#"{"quantization":{"quant_algo":"FP8"}}"#;
        assert_eq!(inject_legacy_model_opt_config(config, legacy)?, config);
        Ok(())
    }

    #[test]
    fn embedded_model_opt_fields_override_legacy_config() -> Result<()> {
        let config = r#"{
            "quantization_config": {
                "quant_method": "modelopt",
                "quant_algo": "FP8_PER_CHANNEL_PER_TOKEN",
                "ignore": ["new_head"]
            }
        }"#;
        let legacy = r#"{
            "producer":{"name":"modelopt","version":"legacy"},
            "quantization":{"quant_algo":"FP8","exclude_modules":["old_head"]}
        }"#;
        let merged: serde_json::Value =
            serde_json::from_str(&inject_legacy_model_opt_config(config, legacy)?)?;
        let quantization = &merged["quantization_config"];
        assert_eq!(quantization["quant_method"], "modelopt");
        assert_eq!(quantization["quant_algo"], "FP8_PER_CHANNEL_PER_TOKEN");
        assert_eq!(quantization["ignore"], serde_json::json!(["new_head"]));
        assert_eq!(quantization["producer"]["version"], "legacy");
        Ok(())
    }

    #[test]
    fn legacy_model_opt_config_is_propagated_to_text_config() -> Result<()> {
        let config = r#"{"model_type":"test","text_config":{"hidden_size":128}}"#;
        let legacy = r#"{"quantization":{"quant_algo":"FP8"}}"#;
        let merged: serde_json::Value =
            serde_json::from_str(&inject_legacy_model_opt_config(config, legacy)?)?;
        assert_eq!(
            merged["text_config"]["quantization_config"],
            merged["quantization_config"]
        );
        Ok(())
    }

    #[test]
    fn nested_model_opt_config_is_merged_and_propagated() -> Result<()> {
        let config = r#"{
            "model_type":"test",
            "text_config":{"quantization_config":{
                "quant_method":"modelopt",
                "quant_algo":"FP8_PER_CHANNEL_PER_TOKEN"
            }}
        }"#;
        let legacy = r#"{
            "producer":{"name":"modelopt","version":"0.19.0"},
            "quantization":{"quant_algo":"FP8"}
        }"#;
        let merged: serde_json::Value =
            serde_json::from_str(&inject_legacy_model_opt_config(config, legacy)?)?;
        assert_eq!(
            merged["quantization_config"]["quant_algo"],
            "FP8_PER_CHANNEL_PER_TOKEN"
        );
        assert_eq!(
            merged["quantization_config"]["producer"]["version"],
            "0.19.0"
        );
        assert_eq!(
            merged["text_config"]["quantization_config"],
            merged["quantization_config"]
        );
        Ok(())
    }

    #[test]
    fn qk_rope_layout_marker_round_trips() -> Result<()> {
        for layout in [
            crate::gguf::normal_registry::RopePairing::Adjacent,
            crate::gguf::normal_registry::RopePairing::HalfSplit,
        ] {
            let stamped = stamp_qk_rope_layout(r#"{"hidden_size":16}"#, layout)?;
            assert_eq!(qk_rope_layout_from_config(&stamped)?, Some(layout));
            assert_eq!(
                serde_json::from_str::<serde_json::Value>(&stamped)?["hidden_size"],
                16
            );
        }
        assert!(qk_rope_layout_from_config(r#"{"_inference_qk_rope_layout":"invalid"}"#).is_err());
        Ok(())
    }

    #[test]
    fn adjacent_qk_rope_layout_rejects_lora_adapters() -> Result<()> {
        assert!(
            validate_lora_qk_rope_layout(r#"{"_inference_qk_rope_layout":"adjacent"}"#, true,)
                .is_err()
        );
        assert!(
            validate_lora_qk_rope_layout(r#"{"_inference_qk_rope_layout":"adjacent"}"#, false,)
                .is_ok()
        );
        assert!(validate_lora_qk_rope_layout(r#"{"hidden_size":16}"#, true).is_ok());
        Ok(())
    }
}
