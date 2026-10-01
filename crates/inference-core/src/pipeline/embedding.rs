use super::isq::{UqffFullSer, UqffWriteConfig, WeightLoadingMode, WeightLoadingState};
use super::{
    AnyMoePipelineMixin, CacheManagerMixin, EitherCache, ForwardInputsResult, GeneralMetadata,
    IsqPipelineMixin, Loader, MetadataMixin, ModelCategory, ModelKind, ModelPaths,
    PreProcessingMixin, TokenSource,
};
use crate::Modalities;
use crate::SupportedModality;
use crate::attention::ATTENTION_CHUNK_SIZE;
use crate::device_map::DeviceMapper;
use crate::distributed;
use crate::embedding_models::inputs_processor::{EmbeddingProcessor, ModelInputs};
use crate::embedding_models::{Dense, DenseActivation, Normalize, Pooling};
use crate::paged_attention::AttentionImplementation;
use crate::pipeline::EmbeddingLoaderType;
use crate::pipeline::EmbeddingModel;
use crate::pipeline::EmbeddingModelLoader;
use crate::pipeline::sampling::sample_and_add_toks;
use crate::pipeline::tokenizer::get_tokenizer;
use crate::pipeline::{AutoEmbeddingLoader, EmbeddingModulePaths};
use crate::pipeline::{ChatTemplate, IsqOrganization, Processor};
use crate::prefix_cacher::PrefixCacheManagerV2;
use crate::sequence::Sequence;
use crate::utils::progress::{ProgressScopeGuard, new_multi_progress};
use crate::{
    DeviceMapSetting, GLOBAL_HF_CACHE, PagedAttentionConfig, Pipeline, Topology, TryIntoDType,
};
use anyhow::Context;
use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_nn::{Linear, Module};
use futures::future::BoxFuture;
use hf_hub::Cache;
use inference_quant::IsqType;
use inference_quant::log::once_log_info;
use inference_quant::safetensors::MmapedSafetensors;
use rand_isaac::Isaac64Rng;
use std::any::Any;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use tokenizers::Tokenizer;
use tokio::sync::Mutex;
use tracing::{debug, info, trace, warn};

pub struct EmbeddingPipeline {
    model: Box<dyn EmbeddingModel + Send + Sync>,
    tracked_modules: Vec<inference_quant::TrackedModule>,
    source_weight_files: Vec<std::path::PathBuf>,
    tokenizer: Arc<Tokenizer>,
    model_id: String,
    metadata: Arc<GeneralMetadata>,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
    modules: Vec<Box<dyn Module + Send + Sync>>,
    processor: Arc<dyn Processor + Send + Sync>,
    // Embedding runs keep no KV cache, but the engine reads the cache kind of every pipeline it starts.
    dummy_cache: EitherCache,
}

/// A loader for an embedding (non-quantized) model.
pub struct EmbeddingLoader {
    inner: Box<dyn EmbeddingModelLoader>,
    model_id: String,
    config: EmbeddingSpecificConfig,
    kind: ModelKind,
    tokenizer_json: Option<String>,
    from_uqff: RwLock<Option<Vec<PathBuf>>>,
    hf_cache_path: Option<PathBuf>,
    load_context: EmbeddingLoadContext,
}

#[derive(Clone, Copy, Default)]
pub(crate) enum EmbeddingLoadContext {
    #[default]
    Primary,
    Search,
}

impl EmbeddingLoadContext {
    fn weight_target(self) -> &'static str {
        match self {
            Self::Primary => "model",
            Self::Search => "search embedding model",
        }
    }
}

#[derive(Default)]
/// A builder for a loader for an embedding (non-quantized) model.
pub struct EmbeddingLoaderBuilder {
    model_id: Option<String>,
    config: EmbeddingSpecificConfig,
    kind: ModelKind,
    tokenizer_json: Option<String>,
    hf_cache_path: Option<PathBuf>,
    load_context: EmbeddingLoadContext,
}

#[derive(Clone, Default)]
/// Config specific to loading an embedding model.
pub struct EmbeddingSpecificConfig {
    pub topology: Option<Topology>,
    pub write_uqff: Option<UqffWriteConfig>,
    pub from_uqff: Option<Vec<PathBuf>>,
    pub imatrix: Option<PathBuf>,
    pub calibration_file: Option<PathBuf>,
    pub hf_cache_path: Option<PathBuf>,
}

impl EmbeddingLoaderBuilder {
    pub fn new(
        config: EmbeddingSpecificConfig,
        tokenizer_json: Option<String>,
        model_id: Option<String>,
    ) -> Self {
        let hf_cache_path = config.hf_cache_path.clone();
        Self {
            config,
            tokenizer_json,
            model_id,
            kind: ModelKind::Normal,
            hf_cache_path,
            ..Default::default()
        }
    }

    pub fn hf_cache_path(mut self, hf_cache_path: PathBuf) -> Self {
        self.hf_cache_path = Some(hf_cache_path);
        self
    }

    pub(crate) fn with_load_context(mut self, load_context: EmbeddingLoadContext) -> Self {
        self.load_context = load_context;
        self
    }

    pub fn build(self, loader: Option<EmbeddingLoaderType>) -> anyhow::Result<Box<dyn Loader>> {
        let loader: Box<dyn EmbeddingModelLoader> = match loader {
            Some(tp) => tp.loader()?,
            None => Box::new(AutoEmbeddingLoader),
        };
        Ok(Box::new(EmbeddingLoader {
            inner: loader,
            model_id: self.model_id.unwrap(),
            config: self.config,
            kind: self.kind,
            tokenizer_json: self.tokenizer_json,
            from_uqff: RwLock::new(None),
            hf_cache_path: self.hf_cache_path,
            load_context: self.load_context,
        }))
    }
}

impl Loader for EmbeddingLoader {
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
        let cache = self
            .hf_cache_path
            .clone()
            .map(Cache::new)
            .unwrap_or_default();
        GLOBAL_HF_CACHE.get_or_init(|| cache);

        let paths = super::paths::get_embedding_paths(super::paths::PathsRequest {
            model_id: &self.model_id,
            tokenizer_json: self.tokenizer_json.as_deref(),
            chat_template: None,
            token_source: &token_source,
            revision: revision.clone(),
            quantized_model_id: None,
            quantized_filenames: None,
            silent,
            loading_uqff: self.config.from_uqff.is_some(),
        });
        if let Some(from_uqff) = self.config.from_uqff.as_ref() {
            let files = super::paths::get_uqff_paths(
                from_uqff,
                &self.model_id,
                &token_source,
                revision.clone(),
                silent,
            )?;
            *self.from_uqff.write().unwrap() = Some(files);
        }
        self.load_model_from_path(
            &paths?,
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
        mut paged_attn_config: Option<PagedAttentionConfig>,
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>> {
        let _progress_guard = ProgressScopeGuard::new(silent);
        let config = super::loading::prepare_model_config(
            None,
            paths.get_config_filename(),
            self.config.from_uqff.is_some(),
            None,
            false,
        )?;

        if paged_attn_config.is_some() {
            warn!("PagedAttention is not supported for embedding models, disabling it.");
            paged_attn_config = None;
        }

        debug!("Prompt chunk size is {ATTENTION_CHUNK_SIZE}.");

        let write_uqff = self.config.write_uqff.is_some();
        let super::loading::LoadDevices {
            tensor_parallelism,
            device,
            available_devices,
        } = super::loading::resolve_load_devices(
            self.inner.model_config(&config)?.as_ref(),
            device,
            write_uqff,
        )?;
        let use_distributed = tensor_parallelism.is_enabled();
        let super::loading::WeightSources {
            uqff_reader,
            combined: weight_source,
            ..
        } = super::loading::open_weight_sources(self.from_uqff.read().unwrap().as_deref(), None)?;

        let super::loading::ResolvedMapSetting {
            setting: mapper, ..
        } = super::loading::resolve_map_setting(
            super::loading::MapSettingInputs {
                setting: mapper,
                write_uqff,
                distributed: use_distributed,
                available_devices: &available_devices,
                dtype,
                sizing: super::isq_flow::AutoDeviceMapSizingInputs {
                    loader: &*self.inner,
                    config: &config,
                    sizing: super::isq_flow::resolve_auto_device_map_sizing(
                        uqff_reader.is_some(),
                        false,
                        in_situ_quant,
                    ),
                    weight_source: weight_source.as_ref(),
                    prepared_weight_source: None,
                    topology: self.config.topology.as_ref(),
                    organization: IsqOrganization::Default,
                    weight_filenames: paths.get_weight_filenames(),
                    has_lora: false,
                    matformer: None,
                    non_mapped_unpacked: false,
                },
            },
            &mut paged_attn_config,
        )?;

        let super::loading::MaterializedDeviceMapper {
            pipeline_mapper,
            mapper,
            layer_devices,
            dtype,
        } = super::loading::materialize_device_mapper(
            super::loading::DeviceMapperInputs {
                setting: &mapper,
                num_layers: self.inner.num_layers(&config)?,
                device: &device,
                available_devices: &available_devices,
                topology: self.config.topology.as_ref(),
                write_uqff,
                dtype,
            },
            &mut paged_attn_config,
        )?;

        trace!("Model config: {:?}", self.inner.get_config_repr(&config)?);
        if crate::using_flash_attn() {
            once_log_info("FlashAttention is enabled.");
        }

        let topology_overrides = self
            .config
            .topology
            .as_ref()
            .map(|topology| topology.immediate_overrides())
            .unwrap_or_default();

        let plan = super::isq_flow::resolve_and_install_isq_plan(super::isq_flow::IsqPlanInputs {
            in_situ_quant,
            has_imatrix: self.config.imatrix.is_some(),
            has_calibration: self.config.calibration_file.is_some(),
            write_uqff_types: self.config.write_uqff.as_ref().map(|c| c.types.clone()),
            has_write_uqff: self.config.write_uqff.is_some(),
            loading_from_uqff: self.config.from_uqff.is_some(),
            organization: Default::default(),
            topology_overrides,
            loader: &*self.inner,
            config: &config,
            device: &device,
        })?;
        let use_immediate = plan.immediate_isq_installed;
        let loading_isq = plan.loading_isq;
        let load_device = plan.load_device.clone();

        let attention_mechanism = if paged_attn_config.is_some() {
            AttentionImplementation::PagedAttention
        } else {
            AttentionImplementation::Eager
        };

        let multi_progress = Arc::new(new_multi_progress());

        let modules_config: Vec<_> = paths
            .get_modules()
            .context("Embedding models require the `modules.json` file.")?
            .to_vec();
        assert!(matches!(
            modules_config.first(),
            Some(EmbeddingModulePaths::Transformer { .. })
        ));

        let mut modules: Vec<Box<dyn Module + Send + Sync>> = Vec::new();
        for module in &modules_config {
            match module {
                EmbeddingModulePaths::Transformer { .. } => (),
                EmbeddingModulePaths::Pooling { config, .. } => {
                    let layer: Pooling = serde_json::from_str(&std::fs::read_to_string(config)?)?;
                    modules.push(Box::new(layer));
                }
                EmbeddingModulePaths::Dense { config, model, .. } => {
                    let config: Dense = serde_json::from_str(&std::fs::read_to_string(config)?)?;
                    let safetensors = unsafe { MmapedSafetensors::new(model)? };
                    let weight = safetensors.load("linear.weight", &device, Some(dtype))?;
                    let bias = if config.bias {
                        Some(safetensors.load("linear.bias", &device, Some(dtype))?)
                    } else {
                        None
                    };
                    let (out_f, in_f) = weight.dims2()?;
                    assert_eq!((out_f, in_f), (config.out_features, config.in_features));
                    if !matches!(config.activation_function, DenseActivation::Identity) {
                        anyhow::bail!("Expected Identity activation function.");
                    }

                    modules.push(Box::new(Linear::new(weight, bias)));
                }
                EmbeddingModulePaths::Normalize { .. } => {
                    modules.push(Box::new(Normalize));
                }
            }
        }
        info!(
            "{}",
            WeightLoadingMode::from(WeightLoadingState {
                from_uqff: self.config.from_uqff.is_some(),
                loading_isq,
                immediate_isq: use_immediate,
                write_uqff: self.config.write_uqff.is_some(),
            })
            .message(self.load_context.weight_target())
        );

        let load_parts = super::loading::LoadMetadataParts {
            loading_isq,
            attention: attention_mechanism,
            device: device.clone(),
            multi_progress: multi_progress.clone(),
            matformer: None,
        };
        let (model, tracker) = if use_distributed {
            let (mapper, sharded_vb) =
                distributed::prepare_distributed_mapper(distributed::DistributedMapperConfig {
                    dtype,
                    device: &device,
                    available_devices: &available_devices,
                    global_world_size_override: tensor_parallelism.world_size(),
                    silent,
                    config: &config,
                    loading_isq,
                    from_uqff: self.config.from_uqff.is_some(),
                    write_uqff: self.config.write_uqff.is_some(),
                    organization: IsqOrganization::Default,
                    isq_loader: &*self.inner,
                    mapped_loader: &*self.inner,
                    weights: distributed::DistributedWeightSource::Paths(paths),
                })?;
            let sharded_vb = match uqff_reader.clone() {
                Some(reader) => sharded_vb.with_uqff_reader(reader),
                _ => sharded_vb,
            };

            // Special case for where things can be more optimially loaded.
            match self.kind {
                ModelKind::Normal => {
                    let tracker = sharded_vb.tracker().clone();
                    let model = self.inner.load(
                        &config,
                        sharded_vb,
                        load_parts.metadata(mapper, None),
                        attention_mechanism,
                    )?;
                    (model, tracker)
                }
                _ => unreachable!(),
            }
        } else {
            match self.kind {
                ModelKind::Normal => {
                    let weights = super::loading::WeightFiles {
                        paths,
                        dtype,
                        device: &load_device,
                        layer_devices: layer_devices.clone(),
                        silent,
                        uqff_reader: uqff_reader.clone(),
                    };
                    let placeholders = super::loading::uqff_placeholders(
                        &*self.inner,
                        &config,
                        loading_isq,
                        self.config.from_uqff.is_some(),
                        false,
                    )?;
                    let device_for_tensor =
                        self.inner
                            .get_device_for_tensor(&config, &*mapper, loading_isq)?;
                    let vb = weights.load(placeholders, device_for_tensor)?;
                    let tracker = vb.tracker().clone();
                    let model = self.inner.load(
                        &config,
                        vb,
                        load_parts.metadata(mapper, None),
                        attention_mechanism,
                    )?;
                    (model, tracker)
                }
                _ => unreachable!(),
            }
        };

        let tokenizer = get_tokenizer(paths.get_tokenizer_filename(), None)?;

        let modules_json = EmbeddingModulePaths::serialize_modules(&modules_config);
        // cloned out so the tracker lock is not held through calibration and the UQFF write
        let tracked = tracker.get().clone();
        super::isq_flow::finish_isq_load(super::isq_flow::FinishIsqLoad {
            plan: &plan,
            modules: tracked,
            drive: &super::isq_flow::EmbeddingCalibrationDrive(&*model),
            in_situ_quant,
            imatrix: self.config.imatrix.as_ref(),
            calibration_file: self.config.calibration_file.as_ref(),
            calibration: super::isq_flow::CalibrationCtx {
                tokenizer: &tokenizer,
                bos_tok_id: None,
                load_device: &load_device,
                mapper: Some(pipeline_mapper.as_ref()),
            },
            uqff: self
                .config
                .write_uqff
                .as_ref()
                .map(|write| super::isq_flow::UqffArtifact {
                    config: write,
                    residual: model.residual_tensors(),
                    full_ser: UqffFullSer {
                        tokenizer: &tokenizer,
                        template_filename: paths.get_template_filename(),
                        effective_chat_template: None,
                        generation_config: paths.get_gen_conf_filename(),
                        config: config.clone(),
                        processor_filename: &None,
                        preprocessor_filename: &None,
                        modules: Some(&modules_json),
                        module_paths: Some(&modules_config),
                    },
                }),
        })?;

        let has_causal_attention = self.inner.has_causal_attention(&config)?;
        let max_seq_len = self.inner.model_config(&config)?.max_seq_len();
        let tracked_modules = tracker.get().clone();
        // rank-sliced layers re-slice at source read; inexpressible slices fall back per layer
        let source_weight_files = super::loading::source_weight_files(
            None,
            self.config.from_uqff.is_some(),
            paths.get_weight_filenames(),
        );

        Ok(Arc::new(Mutex::new(EmbeddingPipeline {
            dummy_cache: EitherCache::Full(crate::pipeline::Cache::new(0, false)),
            model,
            tracked_modules,
            source_weight_files,
            tokenizer: tokenizer.into(),
            model_id: self.model_id.clone(),
            metadata: Arc::new(GeneralMetadata {
                max_seq_len,
                llg_factory: None,
                is_xlora: false,
                no_prefix_cache: false,
                num_hidden_layers: 1, // read only to size caches
                eos_tok: vec![],
                kind: ModelKind::Normal,
                no_kv_cache: true, // NOTE(EricLBuehler): no cache for these.
                activation_dtype: dtype,
                sliding_window: None,
                cache_config: None,
                cache_engine: None,
                model_metadata: None,
                modalities: Modalities {
                    input: vec![SupportedModality::Text],
                    output: vec![SupportedModality::Embedding],
                },
                loaded_for_uqff_write: self.config.write_uqff.is_some(),
            }),
            mapper: pipeline_mapper,
            modules,
            processor: Arc::new(EmbeddingProcessor {
                has_causal_attention,
            }),
        })))
    }

    fn get_id(&self) -> String {
        self.model_id.to_string()
    }

    fn get_kind(&self) -> ModelKind {
        self.kind.clone()
    }
}

impl PreProcessingMixin for EmbeddingPipeline {
    fn get_processor(&self) -> Arc<dyn Processor> {
        self.processor.clone()
    }
    fn get_chat_template(&self) -> Option<Arc<ChatTemplate>> {
        None
    }
    fn get_input_processor_config(&self) -> Option<Arc<dyn Any>> {
        None
    }
}

impl IsqPipelineMixin for EmbeddingPipeline {
    fn re_isq_model(&mut self, dtype: IsqType) -> Result<()> {
        super::isq_flow::requantize_tracked_modules(&self.tracked_modules, dtype)
    }

    fn begin_calibration(&mut self) -> Result<()> {
        super::isq_flow::begin_calibration(&self.tracked_modules).map(|_| ())
    }

    fn calibration_status(&self) -> Result<super::isq_flow::CalibrationStatus> {
        Ok(super::isq_flow::calibration_status(&self.tracked_modules))
    }

    fn apply_calibration(
        &mut self,
        save_cimatrix: Option<std::path::PathBuf>,
    ) -> Result<super::isq_flow::CalibrationStatus> {
        super::isq_flow::apply_calibration(
            &self.tracked_modules,
            &self.source_weight_files,
            None,
            save_cimatrix.as_deref(),
        )
    }
}

impl CacheManagerMixin for EmbeddingPipeline {
    fn clone_in_cache(&self, _seqs: &mut [&mut Sequence]) -> candle_core::Result<()> {
        Ok(())
    }
    fn clone_out_cache(&self, _seqs: &mut [&mut Sequence]) {}
    fn set_none_cache(
        &self,
        _seqs: &mut [&mut Sequence],
        _reset_non_granular: bool,
        _modify_draft_cache: bool,
        _load_preallocated_cache: bool,
    ) -> candle_core::Result<()> {
        Ok(())
    }
    fn cache(&self) -> &EitherCache {
        &self.dummy_cache
    }
}

impl MetadataMixin for EmbeddingPipeline {
    fn device(&self) -> Device {
        self.model.device().clone()
    }
    fn get_metadata(&self) -> Arc<GeneralMetadata> {
        self.metadata.clone()
    }
    fn name(&self) -> String {
        self.model_id.clone()
    }
    fn reset_non_granular_state(&self) {}
    fn tokenizer(&self) -> Option<Arc<Tokenizer>> {
        Some(self.tokenizer.clone())
    }
    fn device_mapper(&self) -> Option<&dyn DeviceMapper> {
        Some(&*self.mapper)
    }
}

impl Pipeline for EmbeddingPipeline {
    fn forward_inputs(
        &mut self,
        inputs: Box<dyn Any>,
        _return_raw_logits: bool,
    ) -> candle_core::Result<ForwardInputsResult> {
        let ModelInputs {
            input_ids,
            flash_meta,
        } = *inputs.downcast::<ModelInputs>().expect("Downcast failed.");

        let mut xs = self.model.forward(&input_ids, &flash_meta)?;
        for module in &self.modules {
            xs = module.forward(&xs)?;
        }

        Ok(ForwardInputsResult::Embeddings { embeddings: xs })
    }
    fn sample_causal_gen<'a>(
        &'a self,
        seqs: &'a mut [&mut Sequence],
        logits: Vec<Tensor>,
        prefix_cacher: &'a mut PrefixCacheManagerV2,
        disable_eos_stop: bool,
        rng: Arc<std::sync::Mutex<Isaac64Rng>>,
    ) -> BoxFuture<'a, Result<(), candle_core::Error>> {
        sample_and_add_toks(self, seqs, logits, prefix_cacher, disable_eos_stop, rng)
    }
    fn category(&self) -> ModelCategory {
        ModelCategory::Embedding
    }
}

impl AnyMoePipelineMixin for EmbeddingPipeline {}
