use super::hf::{
    RemoteAccessIssue, build_api_with_cache, hf_access_error, remote_issue_from_api_error,
};
use super::{
    DiffusionLoaderBuilder, DiffusionLoaderType, EmbeddingLoaderBuilder, EmbeddingLoaderType,
    EmbeddingSpecificConfig, Loader, ModelKind, ModelPaths, MultimodalLoaderBuilder,
    MultimodalLoaderType, MultimodalSpecificConfig, NormalLoaderBuilder, NormalLoaderType,
    NormalSpecificConfig, SpeechLoader, TokenSource,
};
use crate::pipeline::LoadOptions;
use crate::utils::progress::ProgressScopeGuard;
use crate::{AutoDeviceMapParams, DeviceMapSetting, LoraAdapterSpec, LoraRuntimeConfig, Pipeline};
use anyhow::Result;
use hf_hub::{
    Cache, Repo, RepoType,
    api::sync::{ApiError, ApiRepo},
};
use serde::Deserialize;
use std::io;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;
use tracing::{debug, info, warn};

const SORTFORMER_NAME: &str = "sortformer";

/// Automatically selects the appropriate loader based on repository/config metadata.
pub struct AutoLoader {
    model_id: String,
    normal_builder: Mutex<Option<NormalLoaderBuilder>>,
    multimodal_builder: Mutex<Option<MultimodalLoaderBuilder>>,
    embedding_builder: Mutex<Option<EmbeddingLoaderBuilder>>,
    loader: Mutex<Option<Box<dyn Loader>>>,
    hf_cache_path: Option<PathBuf>,
    hf_config_overrides: Option<super::HfConfigOverrides>,
    max_model_len: Option<usize>,
    dynamic_lora_enabled: bool,
}

pub struct AutoLoaderBuilder {
    normal_cfg: NormalSpecificConfig,
    multimodal_cfg: MultimodalSpecificConfig,
    embedding_cfg: EmbeddingSpecificConfig,
    chat_template: Option<String>,
    tokenizer_json: Option<String>,
    model_id: String,
    jinja_explicit: Option<String>,
    no_kv_cache: bool,
    lora_adapters: Option<Vec<LoraAdapterSpec>>,
    lora_runtime_config: Option<LoraRuntimeConfig>,
    hf_cache_path: Option<PathBuf>,
    mtp: bool,
    encoder_cache_memory_bytes: Option<usize>,
}

impl AutoLoaderBuilder {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        normal_cfg: NormalSpecificConfig,
        multimodal_cfg: MultimodalSpecificConfig,
        embedding_cfg: EmbeddingSpecificConfig,
        chat_template: Option<String>,
        tokenizer_json: Option<String>,
        model_id: String,
        no_kv_cache: bool,
        jinja_explicit: Option<String>,
    ) -> Self {
        Self {
            normal_cfg,
            multimodal_cfg,
            embedding_cfg,
            chat_template,
            tokenizer_json,
            model_id,
            jinja_explicit,
            no_kv_cache,
            lora_adapters: None,
            lora_runtime_config: None,
            hf_cache_path: None,
            mtp: false,
            encoder_cache_memory_bytes: None,
        }
    }

    /// Load the MTP head built into the checkpoint so it can drive speculative decoding.
    pub fn with_mtp(mut self, mtp: bool) -> Self {
        self.mtp = mtp;
        self
    }

    pub fn with_encoder_cache_memory_bytes(mut self, max_bytes: Option<usize>) -> Self {
        if let Some(max_bytes) = max_bytes {
            assert!(max_bytes > 0, "encoder cache memory must be nonzero");
        }
        self.encoder_cache_memory_bytes = max_bytes;
        self
    }

    pub fn with_lora(
        mut self,
        adapters: Vec<LoraAdapterSpec>,
        runtime_config: LoraRuntimeConfig,
    ) -> Self {
        self.lora_adapters = Some(adapters);
        self.lora_runtime_config = Some(runtime_config);
        self
    }

    pub fn hf_cache_path(mut self, path: PathBuf) -> Self {
        self.hf_cache_path = Some(path);
        self
    }

    pub fn build(self) -> Box<dyn Loader> {
        let Self {
            normal_cfg,
            multimodal_cfg,
            embedding_cfg,
            chat_template,
            tokenizer_json,
            model_id,
            jinja_explicit,
            no_kv_cache,
            lora_adapters,
            lora_runtime_config,
            hf_cache_path,
            mtp,
            encoder_cache_memory_bytes,
        } = self;

        let hf_config_overrides = normal_cfg
            .hf_config_overrides
            .clone()
            .or(multimodal_cfg.hf_config_overrides.clone());
        let max_model_len = normal_cfg.max_model_len.or(multimodal_cfg.max_model_len);
        let mut normal_builder = NormalLoaderBuilder::new(
            normal_cfg,
            chat_template.clone(),
            tokenizer_json.clone(),
            Some(model_id.clone()),
            no_kv_cache,
            jinja_explicit.clone(),
        );
        if let (Some(adapters), Some(runtime_config)) = (lora_adapters.clone(), lora_runtime_config)
        {
            normal_builder = normal_builder.with_lora(adapters, runtime_config);
        }
        if let Some(ref path) = hf_cache_path {
            normal_builder = normal_builder.hf_cache_path(path.clone());
        }
        normal_builder = normal_builder.with_mtp(mtp);

        let mut multimodal_builder = MultimodalLoaderBuilder::new(
            multimodal_cfg,
            chat_template,
            tokenizer_json.clone(),
            Some(model_id.clone()),
            jinja_explicit,
        );
        if let (Some(adapters), Some(runtime_config)) = (lora_adapters.clone(), lora_runtime_config)
        {
            multimodal_builder = multimodal_builder.with_lora(adapters, runtime_config);
        }
        if let Some(ref path) = hf_cache_path {
            multimodal_builder = multimodal_builder.hf_cache_path(path.clone());
        }
        multimodal_builder = multimodal_builder
            .with_mtp(mtp)
            .with_encoder_cache_memory_bytes(encoder_cache_memory_bytes);

        let mut embedding_builder =
            EmbeddingLoaderBuilder::new(embedding_cfg, tokenizer_json, Some(model_id.clone()));
        if let Some(ref path) = hf_cache_path {
            embedding_builder = embedding_builder.hf_cache_path(path.clone());
        }

        Box::new(AutoLoader {
            model_id,
            normal_builder: Mutex::new(Some(normal_builder)),
            multimodal_builder: Mutex::new(Some(multimodal_builder)),
            embedding_builder: Mutex::new(Some(embedding_builder)),
            loader: Mutex::new(None),
            hf_cache_path,
            hf_config_overrides,
            max_model_len,
            dynamic_lora_enabled: lora_adapters.is_some(),
        })
    }
}

#[derive(Deserialize)]
struct AutoConfig {
    #[serde(default)]
    architectures: Vec<String>,
}

struct ConfigArtifacts {
    contents: Option<String>,
    sentence_transformers_present: bool,
    repo_files: Vec<String>,
    remote_access_issue: Option<RemoteAccessIssue>,
}

enum Detected {
    Normal(NormalLoaderType),
    Multimodal(MultimodalLoaderType),
    Embedding(Option<EmbeddingLoaderType>),
    Diffusion(DiffusionLoaderType),
    Speech(crate::pipeline::SpeechLoaderType),
    Transcription(crate::pipeline::TranscriptionLoaderType),
    VoiceActivity,
    Diarization,
}

fn supports_dynamic_lora(detected: &Detected) -> bool {
    match detected {
        Detected::Normal(_) => true,
        Detected::Multimodal(loader) => super::multimodal::supports_dynamic_lora_loader(loader),
        _ => false,
    }
}

impl AutoLoader {
    fn try_get_file(
        api: &ApiRepo,
        model_id: &Path,
        file: &str,
        revision: &str,
    ) -> std::result::Result<Option<PathBuf>, ApiError> {
        crate::pipeline::hf::try_get_file(api, model_id, file, revision)
    }

    fn list_local_repo_files(model_root: &Path) -> Vec<String> {
        fn collect_files(root: &Path, dir: &Path, out: &mut Vec<String>) -> io::Result<()> {
            for entry in std::fs::read_dir(dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.is_dir() {
                    collect_files(root, &path, out)?;
                } else if let Ok(rel) = path.strip_prefix(root) {
                    out.push(rel.to_string_lossy().replace('\\', "/"));
                }
            }
            Ok(())
        }

        if !model_root.is_dir() {
            return Vec::new();
        }

        let mut files = Vec::new();
        if collect_files(model_root, model_root, &mut files).is_err() {
            return Vec::new();
        }
        files
    }

    fn read_config_from_path(&self, paths: &dyn ModelPaths) -> Result<ConfigArtifacts> {
        let config_path = paths.get_config_filename();
        let contents = match std::fs::read_to_string(config_path) {
            Ok(contents) => Some(contents),
            Err(err) if err.kind() == io::ErrorKind::NotFound => None,
            Err(err) => return Err(err.into()),
        };
        let contents = contents
            .map(|config| self.apply_config_overrides(&config))
            .transpose()?;
        let model_root = Path::new(&self.model_id);
        let repo_files = if model_root.exists() {
            Self::list_local_repo_files(model_root)
        } else {
            Vec::new()
        };
        let sentence_transformers_present = Self::has_sentence_transformers_sibling(config_path)
            || repo_files
                .iter()
                .any(|f| f == "config_sentence_transformers.json");
        Ok(ConfigArtifacts {
            contents,
            sentence_transformers_present,
            repo_files,
            remote_access_issue: None,
        })
    }

    fn read_config_from_hf(
        &self,
        revision: Option<String>,
        token_source: &TokenSource,
        silent: bool,
    ) -> Result<ConfigArtifacts> {
        let cache = self
            .hf_cache_path
            .clone()
            .map(Cache::new)
            .unwrap_or_default();
        let api = build_api_with_cache(token_source, !silent, Some(cache))?;
        let revision = revision.unwrap_or_else(|| "main".to_string());
        let api = api.repo(Repo::with_revision(
            self.model_id.clone(),
            RepoType::Model,
            revision.clone(),
        ));
        let model_id = Path::new(&self.model_id);
        let mut remote_access_issue = None;
        let contents = match Self::try_get_file(&api, model_id, "config.json", &revision) {
            Ok(Some(path)) => Some(std::fs::read_to_string(&path)?),
            Ok(None) => None,
            Err(err) => {
                let issue =
                    remote_issue_from_api_error(model_id, Some("config.json"), &revision, &err);
                warn!(
                    "Auto loader could not fetch `config.json` for `{}`: {}",
                    self.model_id, issue.message
                );
                remote_access_issue = Some(issue);
                None
            }
        };
        let contents = contents
            .map(|config| self.apply_config_overrides(&config))
            .transpose()?;
        let sentence_transformers_present =
            model_id.join("config_sentence_transformers.json").exists()
                || Self::fetch_sentence_transformers_config(&api, model_id, &revision);
        let repo_files = if model_id.exists() {
            Self::list_local_repo_files(model_id)
        } else {
            crate::pipeline::hf::list_repo_files(
                &api,
                std::path::Path::new(model_id),
                false,
                &revision,
            )?
        };
        Ok(ConfigArtifacts {
            contents,
            sentence_transformers_present,
            repo_files,
            remote_access_issue,
        })
    }

    fn has_sentence_transformers_sibling(config_path: &Path) -> bool {
        config_path
            .parent()
            .map(|parent| parent.join("config_sentence_transformers.json").exists())
            .unwrap_or(false)
    }

    fn apply_config_overrides(&self, config: &str) -> Result<String> {
        match self.hf_config_overrides.as_ref() {
            Some(overrides) => overrides.apply(config),
            None => Ok(config.to_string()),
        }
    }

    fn fetch_sentence_transformers_config(api: &ApiRepo, model_id: &Path, revision: &str) -> bool {
        match crate::pipeline::hf::try_get_file(
            api,
            model_id,
            "config_sentence_transformers.json",
            revision,
        ) {
            Ok(Some(_)) => true,
            Ok(None) => false,
            Err(err) => {
                debug!(
                    "No `config_sentence_transformers.json` found for `{}`: {err}",
                    model_id.display()
                );
                false
            }
        }
    }

    fn detect(&self, artifacts: &ConfigArtifacts) -> Result<Detected> {
        if let Some(tp) = DiffusionLoaderType::auto_detect_from_files(&artifacts.repo_files) {
            return Ok(Detected::Diffusion(tp));
        }

        if let Some(ref config) = artifacts.contents
            && let Some(tp) = crate::pipeline::SpeechLoaderType::auto_detect_from_config(config)
        {
            return Ok(Detected::Speech(tp));
        }

        if let Some(ref config) = artifacts.contents
            && let Some(tp) =
                crate::pipeline::TranscriptionLoaderType::auto_detect_from_config(config)
        {
            return Ok(Detected::Transcription(tp));
        }

        if let Some(ref config) = artifacts.contents
            && crate::pipeline::DiarizationLoaderType::auto_detect_from_config(config).is_some()
        {
            return Ok(Detected::Diarization);
        }

        if artifacts.sentence_transformers_present {
            if let Some(ref config) = artifacts.contents {
                let cfg: AutoConfig = serde_json::from_str(config)?;
                if let Some(name) = cfg.architectures.first()
                    && let Ok(tp) = EmbeddingLoaderType::from_causal_lm_name(name)
                {
                    info!(
                        "Detected `config_sentence_transformers.json`; using embedding loader `{tp}`."
                    );
                    return Ok(Detected::Embedding(Some(tp)));
                }
            }
            if artifacts.contents.is_none()
                && let Some(issue) = artifacts.remote_access_issue.as_ref()
            {
                return Err(hf_access_error(Path::new(&self.model_id), issue));
            }
            info!(
                "Detected `config_sentence_transformers.json`; routing via auto embedding loader."
            );
            return Ok(Detected::Embedding(None));
        }

        // Detect Mistral-native models that use params.json instead of config.json
        if artifacts.contents.is_none() && artifacts.repo_files.iter().any(|f| f == "params.json") {
            // Voxtral uses params.json with a "multimodal" key containing "whisper_model_args"
            info!("Detected `params.json` in repo; routing as Voxtral.");
            return Ok(Detected::Multimodal(MultimodalLoaderType::Voxtral));
        }

        // a local path only: a Hub id would be listed here, before the loader runs
        if artifacts.contents.is_none()
            && std::path::Path::new(&self.model_id).exists()
            && super::transcription::silero_gguf(&self.model_id, None, &TokenSource::None, true)
                .is_ok()
        {
            info!("Detected a Silero VAD GGUF; routing as voice activity detection.");
            return Ok(Detected::VoiceActivity);
        }

        // a Hub repo's `.nemo` is taken at its listing only when it names Sortformer, so another NeMo model is not
        // downloaded to be refused; the loader then checks the archive
        let names_sortformer = |s: &str| s.to_ascii_lowercase().contains(SORTFORMER_NAME);
        let is_nemo = |f: &String| {
            Path::new(f)
                .extension()
                .is_some_and(|e| e == inference_models_speech::nemo::EXTENSION)
                && (names_sortformer(f) || names_sortformer(&self.model_id))
        };
        let local = Path::new(&self.model_id).exists();
        if artifacts.contents.is_none()
            && ((local
                && super::transcription::sortformer_nemo(
                    &self.model_id,
                    None,
                    &TokenSource::None,
                    true,
                )
                .is_ok())
                || (!local && artifacts.repo_files.iter().any(is_nemo)))
        {
            info!("Detected a Streaming Sortformer `.nemo`; routing as speaker diarization.");
            return Ok(Detected::Diarization);
        }

        if artifacts.contents.is_none()
            && super::speech::local_kokoro_gguf(&self.model_id).is_some()
        {
            info!("Detected a Kokoro GGUF; routing as Kokoro.");
            return Ok(Detected::Speech(crate::pipeline::SpeechLoaderType::Kokoro));
        }

        let config = artifacts.contents.as_ref().ok_or_else(|| {
            if let Some(issue) = artifacts.remote_access_issue.as_ref() {
                hf_access_error(Path::new(&self.model_id), issue)
            } else {
                anyhow::anyhow!(
                    "Auto loader could not determine model type: missing `config.json` and no diffusion/speech markers found."
                )
            }
        })?;
        let cfg: AutoConfig = serde_json::from_str(config)?;
        if cfg.architectures.len() != 1 {
            anyhow::bail!("Expected exactly one architecture in config");
        }
        let name = &cfg.architectures[0];
        if let Ok(tp) = MultimodalLoaderType::from_causal_lm_name(name) {
            return Ok(Detected::Multimodal(tp));
        }
        let tp = NormalLoaderType::from_causal_lm_name(name)?;
        Ok(Detected::Normal(tp))
    }

    fn ensure_loader(&self, detected: Detected) -> Result<()> {
        let mut guard = self.loader.lock().unwrap();
        if guard.is_some() {
            return Ok(());
        }
        if self.dynamic_lora_enabled && !supports_dynamic_lora(&detected) {
            anyhow::bail!("dynamic LoRA is not supported for this model architecture");
        }
        if matches!(
            &detected,
            Detected::Embedding(_) | Detected::Diffusion(_) | Detected::Speech(_)
        ) && (self.max_model_len.is_some() || self.hf_config_overrides.is_some())
        {
            anyhow::bail!(
                "HF config overrides and max_model_len are supported only for text and multimodal models"
            );
        }
        match detected {
            Detected::Normal(tp) => {
                let builder = self
                    .normal_builder
                    .lock()
                    .unwrap()
                    .take()
                    .expect("builder taken");
                let loader = builder.build(Some(tp))?;
                *guard = Some(loader);
            }
            Detected::Multimodal(tp) => {
                let builder = self
                    .multimodal_builder
                    .lock()
                    .unwrap()
                    .take()
                    .expect("builder taken");
                let loader = builder.build(Some(tp))?;
                *guard = Some(loader);
            }
            Detected::Embedding(tp) => {
                let builder = self
                    .embedding_builder
                    .lock()
                    .unwrap()
                    .take()
                    .expect("builder taken");
                let loader = builder.build(tp)?;
                *guard = Some(loader);
            }
            Detected::Diffusion(tp) => {
                let loader = DiffusionLoaderBuilder::new(Some(self.model_id.clone())).build(tp);
                *guard = Some(loader);
            }
            Detected::Speech(tp) => {
                let loader: Box<dyn Loader> = Box::new(SpeechLoader {
                    model_id: self.model_id.clone(),
                    dac_model_id: None,
                    arch: Some(tp),
                    cfg: None,
                });
                *guard = Some(loader);
            }
            Detected::Transcription(tp) => {
                let loader: Box<dyn Loader> = Box::new(super::TranscriptionLoader {
                    model_id: self.model_id.clone(),
                    arch: Some(tp),
                    vad_model_id: None,
                });
                *guard = Some(loader);
            }
            Detected::VoiceActivity => {
                let loader: Box<dyn Loader> = Box::new(super::VoiceActivityLoader {
                    model_id: self.model_id.clone(),
                });
                *guard = Some(loader);
            }
            Detected::Diarization => {
                let loader: Box<dyn Loader> = Box::new(super::DiarizationLoader {
                    model_id: self.model_id.clone(),
                });
                *guard = Some(loader);
            }
        }
        Ok(())
    }
}

fn device_map_for_detected(mapper: DeviceMapSetting, detected: &Detected) -> DeviceMapSetting {
    match (mapper, detected) {
        (
            DeviceMapSetting::Auto(AutoDeviceMapParams::Text {
                max_seq_len,
                max_batch_size,
            }),
            Detected::Multimodal(_),
        ) => DeviceMapSetting::Auto(AutoDeviceMapParams::Multimodal {
            max_seq_len,
            max_batch_size,
            max_image_shape: (
                AutoDeviceMapParams::DEFAULT_MAX_IMAGE_LENGTH,
                AutoDeviceMapParams::DEFAULT_MAX_IMAGE_LENGTH,
            ),
            max_num_images: AutoDeviceMapParams::DEFAULT_MAX_NUM_IMAGES,
        }),
        (
            DeviceMapSetting::Auto(AutoDeviceMapParams::Multimodal {
                max_seq_len,
                max_batch_size,
                ..
            }),
            Detected::Normal(_),
        ) => DeviceMapSetting::Auto(AutoDeviceMapParams::Text {
            max_seq_len,
            max_batch_size,
        }),
        (mapper, _) => mapper,
    }
}

impl Loader for AutoLoader {
    fn load_model_from_hf(
        &self,
        revision: Option<String>,
        token_source: TokenSource,
        options: LoadOptions<'_>,
    ) -> Result<Arc<tokio::sync::Mutex<dyn Pipeline + Send + Sync>>> {
        let silent = options.silent;
        let _progress_guard = ProgressScopeGuard::new(silent);
        let config = self.read_config_from_hf(revision.clone(), &token_source, silent)?;
        let detected = self.detect(&config)?;
        let options = LoadOptions {
            mapper: device_map_for_detected(options.mapper, &detected),
            ..options
        };
        self.ensure_loader(detected)?;
        self.loader
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .load_model_from_hf(revision, token_source, options)
    }

    fn load_model_from_path(
        &self,
        paths: &dyn ModelPaths,
        options: LoadOptions<'_>,
    ) -> Result<Arc<tokio::sync::Mutex<dyn Pipeline + Send + Sync>>> {
        let _progress_guard = ProgressScopeGuard::new(options.silent);
        let config = self.read_config_from_path(paths)?;
        let detected = self.detect(&config)?;
        let options = LoadOptions {
            mapper: device_map_for_detected(options.mapper, &detected),
            ..options
        };
        self.ensure_loader(detected)?;
        self.loader
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .load_model_from_path(paths, options)
    }

    fn get_id(&self) -> String {
        self.model_id.clone()
    }

    fn get_kind(&self) -> ModelKind {
        self.loader
            .lock()
            .unwrap()
            .as_ref()
            .map(|l| l.get_kind())
            .unwrap_or(ModelKind::Normal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn detector() -> AutoLoader {
        AutoLoader {
            model_id: "Qwen/Qwen3.6-35B-A3B".to_string(),
            normal_builder: Mutex::new(None),
            multimodal_builder: Mutex::new(None),
            embedding_builder: Mutex::new(None),
            loader: Mutex::new(None),
            hf_cache_path: None,
            hf_config_overrides: None,
            max_model_len: None,
            dynamic_lora_enabled: true,
        }
    }

    fn without_config(model_id: &str, repo_files: &[&str]) -> Result<Detected> {
        let detector = AutoLoader {
            model_id: model_id.to_string(),
            ..detector()
        };
        detector.detect(&ConfigArtifacts {
            contents: None,
            sentence_transformers_present: false,
            repo_files: repo_files.iter().map(|f| f.to_string()).collect(),
            remote_access_issue: None,
        })
    }

    #[test]
    fn a_sortformer_nemo_routes_as_diarization_and_a_transformer_one_does_not() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let weight = inference_tensor::Tensor::zeros(
            1,
            inference_tensor::DType::F32,
            &inference_tensor::Device::Cpu,
        )?;
        let target =
            "target: nemo.collections.asr.models.sortformer_diar_models.SortformerEncLabelModel";
        let write = |name: &str, encoder: &str| -> Result<std::path::PathBuf> {
            let path = dir.path().join(name);
            let yaml =
                format!("{target}\nencoder:\n  _target_: nemo.collections.asr.modules.{encoder}\n");
            inference_models_speech::nemo::write_nemo(&path, &yaml, &[("w", &weight)])?;
            Ok(path)
        };
        let sortformer = write("sortformer.nemo", "ConformerEncoder")?;
        let plain = write("plain.nemo", "TransformerEncoder")?;
        assert!(matches!(
            without_config(&sortformer.to_string_lossy(), &[])?,
            Detected::Diarization
        ));
        assert!(without_config(&plain.to_string_lossy(), &[]).is_err());
        assert!(matches!(
            without_config("org/sortformer-repo", &["README.md", "diar.nemo"])?,
            Detected::Diarization
        ));
        assert!(without_config("org/speech-recognizer", &["README.md", "asr.nemo"]).is_err());
        Ok(())
    }

    #[test]
    fn qwen3_5_moe_auto_detection_routes_dynamic_lora_to_multimodal() {
        let detected = detector()
            .detect(&ConfigArtifacts {
                contents: Some(
                    r#"{"architectures":["Qwen3_5MoeForConditionalGeneration"]}"#.to_string(),
                ),
                sentence_transformers_present: false,
                repo_files: Vec::new(),
                remote_access_issue: None,
            })
            .unwrap();

        assert!(matches!(
            detected,
            Detected::Multimodal(MultimodalLoaderType::Qwen3_5Moe)
        ));
    }

    #[test]
    fn dynamic_lora_auto_detection_accepts_supported_multimodal_language_models() {
        assert!(supports_dynamic_lora(&Detected::Normal(
            NormalLoaderType::Qwen3Moe
        )));
        for loader in [
            MultimodalLoaderType::Qwen2VL,
            MultimodalLoaderType::Qwen2_5VL,
            MultimodalLoaderType::Qwen3VL,
            MultimodalLoaderType::Qwen3VLMoE,
            MultimodalLoaderType::Qwen3_5,
            MultimodalLoaderType::Qwen3_5Moe,
            MultimodalLoaderType::Gemma3,
            MultimodalLoaderType::Gemma3n,
            MultimodalLoaderType::Idefics3,
            MultimodalLoaderType::Mistral3,
            MultimodalLoaderType::Llama4,
            MultimodalLoaderType::Lfm2Vl,
            MultimodalLoaderType::Gemma4,
            MultimodalLoaderType::MuseGlimmer,
        ] {
            assert!(supports_dynamic_lora(&Detected::Multimodal(loader)));
        }
        assert!(!supports_dynamic_lora(&Detected::Multimodal(
            MultimodalLoaderType::Phi3V
        )));
        assert!(!supports_dynamic_lora(&Detected::Embedding(None)));
    }

    #[test]
    fn multimodal_detection_promotes_text_auto_device_mapping() {
        let mapper = device_map_for_detected(
            DeviceMapSetting::Auto(AutoDeviceMapParams::Text {
                max_seq_len: 8192,
                max_batch_size: 3,
            }),
            &Detected::Multimodal(MultimodalLoaderType::Qwen3_5),
        );

        let DeviceMapSetting::Auto(AutoDeviceMapParams::Multimodal {
            max_seq_len,
            max_batch_size,
            max_image_shape,
            max_num_images,
        }) = mapper
        else {
            panic!("expected multimodal device mapping")
        };
        assert_eq!(max_seq_len, 8192);
        assert_eq!(max_batch_size, 3);
        assert_eq!(
            max_image_shape,
            (
                AutoDeviceMapParams::DEFAULT_MAX_IMAGE_LENGTH,
                AutoDeviceMapParams::DEFAULT_MAX_IMAGE_LENGTH,
            )
        );
        assert_eq!(max_num_images, AutoDeviceMapParams::DEFAULT_MAX_NUM_IMAGES);
    }
}
