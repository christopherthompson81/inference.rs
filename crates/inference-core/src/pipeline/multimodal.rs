use super::decoder::{DecoderModel, DecoderPipeline, MediaState};
use super::decoder_core::{DecoderCore, DecoderCoreArgs, LoadedModelView};
use super::isq::{UqffFullSer, UqffWriteConfig};
use super::loaders::MultimodalLoaderTypeExt;
use super::{
    AutoMultimodalLoader, Loader, ModelKind, ModelPaths, MultimodalLoaderType,
    MultimodalModelLoader, TokenSource,
};
use crate::attention::ATTENTION_CHUNK_SIZE;

use crate::pipeline::IsqOrganization;
use crate::pipeline::tokenizer::get_tokenizer;
use crate::utils::progress::ProgressScopeGuard;
use crate::vision_models::preprocessor_config::PreProcessorConfig;
use crate::vision_models::processor_config::ProcessorConfig;
use crate::{
    DeviceMapSetting, LoraAdapterSpec, LoraRuntimeConfig, PagedAttentionConfig, Pipeline, Topology,
    TryIntoDType,
};
use anyhow::Result;
use either::Either;
use inference_protocol::chat_template::{BeginEndUnkPadTok, ChatTemplateValue};
use inference_quant::IsqType;
use inference_tensor::Device;
use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use tokenizers::AddedToken;
use tokio::sync::Mutex;
use tracing::{debug, info, trace, warn};

/// A loader for a multimodal (non-quantized) model.
pub struct MultimodalLoader {
    inner: Box<dyn MultimodalModelLoader>,
    model_id: String,
    config: MultimodalSpecificConfig,
    kind: ModelKind,
    chat_template: Option<String>,
    tokenizer_json: Option<String>,
    from_uqff: RwLock<Option<Vec<PathBuf>>>,
    jinja_explicit: Option<String>,
    hf_cache_path: Option<PathBuf>,
    lora_adapters: Option<Vec<LoraAdapterSpec>>,
    lora_runtime_config: Option<LoraRuntimeConfig>,
    loader_type: Option<MultimodalLoaderType>,
    prepared_source: Option<super::loading::PreparedSource>,
    mtp: bool,
    encoder_cache_memory_bytes: Option<usize>,
}

#[derive(Default)]
/// A builder for a loader for a multimodal (non-quantized) model.
pub struct MultimodalLoaderBuilder {
    model_id: Option<String>,
    config: MultimodalSpecificConfig,
    kind: ModelKind,
    chat_template: Option<String>,
    tokenizer_json: Option<String>,
    jinja_explicit: Option<String>,
    hf_cache_path: Option<PathBuf>,
    lora_adapters: Option<Vec<LoraAdapterSpec>>,
    lora_runtime_config: Option<LoraRuntimeConfig>,
    mtp: bool,
    encoder_cache_memory_bytes: Option<usize>,
}

#[derive(Clone, Default)]
/// Config specific to loading a multimodal model.
pub struct MultimodalSpecificConfig {
    pub topology: Option<Topology>,
    pub write_uqff: Option<UqffWriteConfig>,
    pub from_uqff: Option<Vec<PathBuf>>,
    pub max_edge: Option<u32>,
    pub max_model_len: Option<usize>,
    pub hf_config_overrides: Option<super::HfConfigOverrides>,
    pub imatrix: Option<PathBuf>,
    pub calibration_file: Option<PathBuf>,
    pub hf_cache_path: Option<PathBuf>,
    pub matformer_config_path: Option<PathBuf>,
    pub matformer_slice_name: Option<String>,
    pub organization: IsqOrganization,
}

impl MultimodalLoaderBuilder {
    pub fn new(
        config: MultimodalSpecificConfig,
        chat_template: Option<String>,
        tokenizer_json: Option<String>,
        model_id: Option<String>,
        jinja_explicit: Option<String>,
    ) -> Self {
        let hf_cache_path = config.hf_cache_path.clone();
        Self {
            config,
            chat_template,
            tokenizer_json,
            model_id,
            jinja_explicit,
            kind: ModelKind::Normal,
            hf_cache_path,
            lora_adapters: None,
            lora_runtime_config: None,
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
        self.kind = ModelKind::Lora;
        self.lora_adapters = Some(adapters);
        self.lora_runtime_config = Some(runtime_config);
        self
    }

    pub fn hf_cache_path(mut self, hf_cache_path: PathBuf) -> Self {
        self.hf_cache_path = Some(hf_cache_path);
        self
    }

    fn build_inner(
        self,
        loader: Option<MultimodalLoaderType>,
        prepared_source: Option<super::loading::PreparedSource>,
    ) -> anyhow::Result<Box<dyn Loader>> {
        let loader_type = loader.clone();
        let loader: Box<dyn MultimodalModelLoader> = match loader {
            Some(tp) => tp.loader()?,
            None => Box::new(AutoMultimodalLoader),
        };
        Ok(Box::new(MultimodalLoader {
            inner: loader,
            model_id: self.model_id.unwrap(),
            config: self.config,
            kind: self.kind,
            chat_template: self.chat_template,
            tokenizer_json: self.tokenizer_json,
            jinja_explicit: self.jinja_explicit,
            from_uqff: RwLock::new(None),
            hf_cache_path: self.hf_cache_path,
            lora_adapters: self.lora_adapters,
            lora_runtime_config: self.lora_runtime_config,
            loader_type,
            prepared_source,
            mtp: self.mtp,
            encoder_cache_memory_bytes: self.encoder_cache_memory_bytes,
        }))
    }

    pub fn build(self, loader: Option<MultimodalLoaderType>) -> anyhow::Result<Box<dyn Loader>> {
        self.build_inner(loader, None)
    }

    pub(crate) fn build_with_source(
        mut self,
        loader: MultimodalLoaderType,
        source: super::loading::PreparedSource,
        kind: ModelKind,
    ) -> anyhow::Result<Box<dyn Loader>> {
        self.kind = kind;
        self.build_inner(Some(loader), Some(source))
    }
}

impl MultimodalLoader {
    fn validate_dynamic_lora(&self) -> Result<()> {
        super::validate_lora_loader_config(
            self.lora_adapters.as_deref(),
            self.lora_runtime_config,
        )?;
        if self.lora_adapters.is_some()
            && !self
                .loader_type
                .as_ref()
                .is_some_and(supports_dynamic_lora_loader)
        {
            anyhow::bail!("dynamic LoRA is not supported for this multimodal architecture");
        }
        Ok(())
    }
}

pub(super) fn supports_dynamic_lora_loader(loader: &MultimodalLoaderType) -> bool {
    matches!(
        loader,
        MultimodalLoaderType::Qwen2VL
            | MultimodalLoaderType::Qwen2_5VL
            | MultimodalLoaderType::Qwen3VL
            | MultimodalLoaderType::Qwen3VLMoE
            | MultimodalLoaderType::Qwen3_5
            | MultimodalLoaderType::Qwen3_5Moe
            | MultimodalLoaderType::Gemma3
            | MultimodalLoaderType::Gemma3n
            | MultimodalLoaderType::Idefics3
            | MultimodalLoaderType::Mistral3
            | MultimodalLoaderType::Llama4
            | MultimodalLoaderType::Lfm2Vl
            | MultimodalLoaderType::Gemma4
            | MultimodalLoaderType::MuseGlimmer
    )
}

impl MultimodalLoader {
    #[allow(clippy::too_many_arguments)]
    fn load_from_paths(
        &self,
        paths: &dyn ModelPaths,
        dtype: &dyn TryIntoDType,
        device: &Device,
        silent: bool,
        mapper: DeviceMapSetting,
        in_situ_quant: Option<IsqType>,
        mut paged_attn_config: Option<PagedAttentionConfig>,
        uqff: super::loading::UqffLoad<'_>,
    ) -> Result<super::loading::LoadOutcome> {
        let _progress_guard = ProgressScopeGuard::new(silent);
        let serve_written = uqff.serves_written(device);
        let in_situ_quant = in_situ_quant.filter(|_| !uqff.is_reload());
        // a reload reads the written UQFF: the imatrix and calibration were spent writing it
        let (imatrix, calibration_file) = match uqff {
            super::loading::UqffLoad::Reload(_) => (None, None),
            _ => (
                self.config.imatrix.as_ref(),
                self.config.calibration_file.as_ref(),
            ),
        };
        let from_uqff_files = self.from_uqff.read().unwrap();
        let uqff_files = uqff.reload_files().or(from_uqff_files.as_deref());
        self.validate_dynamic_lora()?;
        let config = super::loading::prepare_model_config(
            self.prepared_source.as_ref(),
            paths.get_config_filename(),
            uqff.reads(),
            self.config.hf_config_overrides.as_ref(),
            self.mtp,
        )?;
        super::loaders::validate_lora_qk_rope_layout(&config, self.lora_adapters.is_some())?;
        let modalities = self.inner.modalities(&config)?;
        let runtime_config = self
            .inner
            .runtime_config(&config, self.config.max_model_len)?;

        if !self.inner.supports_paged_attention(&config) {
            paged_attn_config = None;
        }
        let supports_encoder_cache = self.inner.supports_encoder_cache(&config);
        if self.encoder_cache_memory_bytes.is_some() && !supports_encoder_cache {
            inference_quant::log::once_log_warn(
                "Configured encoder cache capacity ignored because this model has no encoder cache",
            );
        }
        if let (Some(bytes), Some(cache_config)) = (
            self.encoder_cache_memory_bytes
                .filter(|_| supports_encoder_cache),
            paged_attn_config.as_mut(),
        ) {
            *cache_config = (*cache_config).with_base_device_memory_reservation(bytes)?;
        }

        debug!("Prompt chunk size is {ATTENTION_CHUNK_SIZE}.");

        // Tokenizer deserialization can briefly use far more memory than its final representation.
        let (processor, preprocessor_config, tokenizer) = {
            let processor_config_json = match self.prepared_source.as_ref() {
                Some(source) => source.processor_config.clone(),
                None => paths
                    .get_processor_config()
                    .as_ref()
                    .map(|f| fs::read_to_string(f).unwrap()),
            };

            // Some models only ship nested preprocessor settings in processor_config.json.
            let mut preprocessor_config: PreProcessorConfig = match self.prepared_source.as_ref() {
                Some(source) => source
                    .preprocessor_config
                    .as_deref()
                    .map(serde_json::from_str)
                    .transpose()?
                    .unwrap_or_else(|| {
                        processor_config_json.as_deref().map_or_else(
                            PreProcessorConfig::default,
                            |json| {
                                PreProcessorConfig::from_processor_config_json(json)
                                    .unwrap_or_default()
                            },
                        )
                    }),
                None => match paths.get_preprocessor_config().as_ref() {
                    Some(preprocessor_config) => {
                        serde_json::from_str(&fs::read_to_string(preprocessor_config).unwrap())
                            .unwrap()
                    }
                    None => processor_config_json.as_deref().map_or_else(
                        PreProcessorConfig::default,
                        |json| match PreProcessorConfig::from_processor_config_json(json) {
                            Ok(config) => config,
                            Err(err) => {
                                warn!(
                                    "Failed to synthesize preprocessor config from processor_config.json: {err}"
                                );
                                PreProcessorConfig::default()
                            }
                        },
                    ),
                },
            };
            if let Some(video_config) = paths.get_video_preprocessor_config() {
                preprocessor_config.video = Some(Box::new(serde_json::from_str(
                    &fs::read_to_string(video_config).unwrap(),
                )?));
            }
            let processor_config: Option<ProcessorConfig> = processor_config_json
                .as_deref()
                .map(|json| serde_json::from_str(json).unwrap());
            let loader_type = match &self.loader_type {
                Some(loader_type) => loader_type.clone(),
                None => AutoMultimodalLoader::loader_type(&config)?,
            };
            let processor = loader_type.get_processor(
                &config,
                processor_config,
                preprocessor_config.clone(),
                self.config.max_edge,
            )?;
            let tokenizer = match self.prepared_source.as_ref() {
                Some(source) => {
                    let mut tokenizer = source.tokenizer.clone();
                    tokenizer
                        .add_special_tokens(
                            processor
                                .get_special_tokens()
                                .iter()
                                .map(|token| AddedToken::from((*token).to_string(), true)),
                        )
                        .map_err(anyhow::Error::msg)?;
                    tokenizer
                }
                None => get_tokenizer(
                    paths.get_tokenizer_filename(),
                    Some(processor.get_special_tokens()),
                )?,
            };
            (processor, preprocessor_config, tokenizer)
        };

        let matformer = super::loading::load_matformer_slice(
            self.config.matformer_config_path.as_deref(),
            self.config.matformer_slice_name.as_deref(),
        )?;
        let auto_device_map_params = |params: &crate::device_map::AutoDeviceMapParams| {
            self.inner.auto_device_map_params(&runtime_config, params)
        };
        let (session, mapper) = super::loading::open_load_session(
            super::loading::LoadSessionInputs {
                mapped: &*self.inner,
                isq: &*self.inner,
                config: &runtime_config,
                settings: super::loading::LoadSettings {
                    topology: self.config.topology.as_ref(),
                    organization: self.config.organization,
                    write_uqff: uqff.write(),
                    from_uqff: uqff.reads(),
                    has_imatrix: imatrix.is_some(),
                    has_calibration: calibration_file.is_some(),
                },
                paths,
                device,
                dtype,
                mapper,
                in_situ_quant,
                uqff_files,
                prepared: self.prepared_source.as_ref(),
                has_lora: self.lora_adapters.is_some(),
                matformer,
                matformer_sizing: true,
                non_mapped_unpacked: true,
                auto_device_map_params: Some(&auto_device_map_params),
                weight_target: "model",
            },
            &mut paged_attn_config,
        )?;
        trace!("Model config: {:?}", self.inner.get_config_repr(&config)?);
        let (model, tracker, dynamic_lora) = super::loading::load_model(
            &*self.inner,
            &session,
            mapper,
            super::loading::ModelLoadInputs {
                config: &runtime_config,
                paths,
                silent,
                organization: self.config.organization,
                from_uqff: uqff.reads(),
                write_uqff: uqff.write().is_some(),
                prepared: self.prepared_source.as_ref(),
                lora: super::loading::lora_runtime(&self.kind, self.lora_runtime_config),
            },
        )?;
        let super::loading::LoadSession {
            device,
            available_devices,
            weight_source,
            max_kv_tokens,
            pipeline_mapper,
            layer_devices,
            dtype,
            plan,
            ..
        } = session;
        let load_device = plan.load_device.clone();

        if let Some(max_bytes) = self
            .encoder_cache_memory_bytes
            .filter(|_| supports_encoder_cache)
        {
            assert!(
                model.configure_encoder_cache_memory_bytes(max_bytes),
                "multimodal loader advertised an encoder cache but the model did not expose one"
            );
        }

        // Release Metal loader scratch buffers before constructing the remaining pipeline state.
        for device in &available_devices {
            if matches!(device, Device::Metal(_)) {
                device.synchronize()?;
            }
        }

        let gen_conf = super::loading::generation_config(
            self.prepared_source
                .as_ref()
                .map(|source| source.generation_config.clone()),
            paths,
            &config,
        );
        if model.is_block_diffusion()
            && let Some(raw) = paths
                .get_gen_conf_filename()
                .and_then(|f| fs::read_to_string(f).ok())
        {
            model.configure_block_diffusion(&raw);
        }
        let mut chat_template = super::loading::load_chat_template(
            paths,
            self.jinja_explicit.as_ref(),
            self.chat_template.as_ref(),
            self.prepared_source.as_ref(),
        );

        // If no chat template was found, use the loader's built-in default (if any).
        if chat_template.chat_template.is_none()
            && let Some(default_tmpl) = self.inner.default_chat_template(&config)
        {
            info!("Using loader's built-in default chat template.");
            chat_template.chat_template = Some(ChatTemplateValue(Either::Left(default_tmpl)));
        }

        // If no bos/eos tokens are set, use the loader's defaults (e.g. for Voxtral
        // which has no tokenizer_config.json).
        if let Some((bos, eos)) = self.inner.default_bos_eos(&config) {
            if chat_template.bos_token.is_none() {
                chat_template.bos_token = Some(BeginEndUnkPadTok(Either::Left(bos)));
            }
            if chat_template.eos_token.is_none() {
                chat_template.eos_token = Some(BeginEndUnkPadTok(Either::Left(eos)));
            }
        }

        // cloned out so the tracker lock is not held through calibration and the UQFF write
        let tracked = tracker.get().clone();
        let written = super::isq_flow::finish_isq_load(super::isq_flow::FinishIsqLoad {
            plan: &plan,
            modules: tracked,
            drive: &super::isq_flow::MultimodalCalibrationDrive(&*model),
            in_situ_quant,
            imatrix,
            calibration_file,
            calibration: super::isq_flow::CalibrationCtx {
                tokenizer: &tokenizer,
                bos_tok_id: chat_template
                    .bos_tok()
                    .as_deref()
                    .and_then(|tok| tokenizer.token_to_id(tok)),
                load_device: &load_device,
                mapper: Some(pipeline_mapper.as_ref()),
            },
            uqff: uqff.write().map(|write| super::isq_flow::UqffArtifact {
                config: write,
                residual: super::loading::uqff_residual_tensors(self.config.organization, &*model),
                full_ser: UqffFullSer {
                    tokenizer: &tokenizer,
                    template_filename: paths.get_template_filename(),
                    effective_chat_template: Some(&chat_template),
                    generation_config: super::loading::uqff_generation_config_file(
                        paths,
                        self.prepared_source.as_ref(),
                    ),
                    config: config.clone(),
                    processor_filename: paths.get_processor_config(),
                    preprocessor_filename: paths.get_preprocessor_config(),
                    modules: None,
                    module_paths: None,
                },
            }),
        })?;
        if serve_written && let Some(files) = written.filter(|files| !files.is_empty()) {
            if self.prepared_source.is_some() {
                anyhow::bail!(
                    "Wrote the UQFF to `{}`; a model read from GGUF serves from it on a GPU only through a load with `from_uqff`.",
                    files[0].display()
                );
            }
            return Ok(super::loading::LoadOutcome::Written(files));
        }

        let tracked_modules = tracker.get().clone();
        // rank-sliced layers re-slice at source read; inexpressible slices fall back per layer
        let source_weight_files = super::loading::source_weight_files(
            self.prepared_source.as_ref(),
            uqff.reads(),
            paths.get_weight_filenames(),
        );
        let core = DecoderCore::new(DecoderCoreArgs {
            model: LoadedModelView {
                target: &*model,
                cache: model.cache(),
                config: model.model_config(),
                max_seq_len: model.max_seq_len(),
                sliding_window: model.config().sliding_window,
                block_diffusion: model.is_block_diffusion(),
            },
            tokenizer,
            chat_template,
            generation_config: gen_conf,
            paged_attn_config,
            dtype,
            layer_devices,
            device,
            mapper: pipeline_mapper,
            silent,
            max_kv_tokens,
            no_kv_cache: false,
            no_prefix_cache: !self.inner.supports_prefix_cacher(&config),
            kind: self.kind.clone(),
            model_id: self.model_id.clone(),
            modalities,
            loaded_for_uqff_write: uqff.write().is_some(),
            tracked_modules,
            source_weight_files,
            source_weight_source: weight_source,
            dynamic_lora,
        })?;
        Ok(super::loading::LoadOutcome::Pipeline(Arc::new(Mutex::new(
            DecoderPipeline::new(
                DecoderModel::Multimodal(model),
                core,
                Some(MediaState {
                    processor,
                    prefixer: self.inner.prefixer(&config),
                    video_sampling: self.inner.video_frame_sampling(&config),
                    preprocessor_config: Arc::new(preprocessor_config),
                }),
            ),
        ))))
    }
}

impl Loader for MultimodalLoader {
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
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>> {
        let _progress_guard = ProgressScopeGuard::new(silent);
        self.validate_dynamic_lora()?;
        let paths = super::loading::hub_model_paths(
            super::loading::HubPathsRequest {
                hf_cache_path: self.hf_cache_path.clone(),
                model_id: &self.model_id,
                tokenizer_json: self.tokenizer_json.as_deref(),
                chat_template: self.chat_template.as_deref(),
                token_source: &token_source,
                revision,
                silent,
                from_uqff: self.config.from_uqff.as_deref(),
            },
            &self.from_uqff,
            |request| super::paths::get_paths(request, self.lora_adapters.as_deref()),
        )?;
        self.load_model_from_path(
            &paths,
            dtype,
            device,
            silent,
            mapper,
            in_situ_quant,
            paged_attn_config,
        )
    }

    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    fn load_model_from_path(
        &self,
        paths: &dyn ModelPaths,
        dtype: &dyn TryIntoDType,
        device: &Device,
        silent: bool,
        mapper: DeviceMapSetting,
        in_situ_quant: Option<IsqType>,
        paged_attn_config: Option<PagedAttentionConfig>,
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>> {
        let uqff = super::loading::UqffLoad::new(
            self.config.from_uqff.is_some(),
            self.config.write_uqff.as_ref(),
        )?;
        super::loading::load_serving_written(uqff, |uqff| {
            self.load_from_paths(
                paths,
                dtype,
                device,
                silent,
                mapper.clone(),
                in_situ_quant,
                paged_attn_config,
                uqff,
            )
        })
    }

    fn get_id(&self) -> String {
        self.model_id.to_string()
    }

    fn get_kind(&self) -> ModelKind {
        self.kind.clone()
    }
}
