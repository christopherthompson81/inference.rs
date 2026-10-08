use super::decoder_core::{DecoderCore, DecoderCoreArgs, LoadedModelView};
use super::loaders::NormalLoaderTypeExt;
use super::{
    AnyMoePipelineMixin, CacheManagerMixin, EitherCache, ForwardInputsResult, ForwardStepResult,
    IsqOrganization, IsqPipelineMixin, MetadataMixin, ModelCategory, PreProcessingMixin,
};
use super::{AutoNormalLoader, NormalLoaderType};
use super::{
    DecodeGraphPrecaptureCtx, GeneralMetadata, Loader, ModelKind, ModelPaths, NormalModel,
    NormalModelLoader, TokenSource, text_models_inputs_processor::ModelInputs,
};
use crate::amoe::AnyMoeExpertType;
use crate::attention::ATTENTION_CHUNK_SIZE;
#[cfg(feature = "cuda")]
use crate::cuda::gdn::GDN_PAD_SLOT;
use crate::device_map::DeviceMapper;
#[cfg(feature = "cuda")]
struct CudaDecodeGraphCaptureInputs<'a> {
    kv_cache: &'a [(Tensor, Tensor)],
    flash_meta: &'a FlashParams,
    recurrent_batch_kind: RecurrentBatchKind,
}
#[cfg(feature = "cuda")]
struct CudaDecodeGraphForwardInput<'a> {
    input_ids: &'a Tensor,
    seqlen_offsets: &'a [usize],
    context_lens: &'a [(usize, usize)],
    position_ids: &'a [usize],
    paged_attn_meta: Option<(Vec<(Tensor, Tensor)>, &'a PagedAttentionInputMetadata)>,
    flash_meta: &'a FlashParams,
    recurrent_batch_kind: RecurrentBatchKind,
}
#[cfg(feature = "cuda")]
use crate::attention::FlashParams;
#[cfg(feature = "cuda")]
use crate::gdn::RecurrentBatchKind;
#[cfg(feature = "cuda")]
use crate::paged_attention::PagedAttentionInputMetadata;
use crate::pipeline::ChatTemplate;
#[cfg(feature = "cuda")]
use crate::pipeline::cuda_graph::{
    CudaDecodeGraphCaptureCtx, CudaDecodeGraphKey, CudaDecodeGraphLaunch, CudaDecodeGraphReplay,
    CudaDecodeGraphReplayInput, CudaDecodeGraphState, CudaGraphComponent, CudaGraphDecodeStep,
    CudaGraphDecodeStepInputs, CudaGraphDispatchMode, CudaGraphDispatchReason, CudaGraphEvent,
    CudaGraphEventGuard, CudaGraphPrecaptureInputs, capture_cuda_decode_graph,
    cuda_decode_graph_batch_kind_supported, cuda_decode_graph_supported_for_model,
    cuda_decode_graphs_enabled, cuda_graph_batch_bucket, cuda_graph_precapture_batches,
    cuda_graph_precapture_max_batch, disable_cuda_decode_graph, finish_cuda_graph_capture_attempt,
    hybrid_graph_slots, install_hybrid_graph_state_indices, record_cuda_graph_dispatch,
    snapshot_hybrid_recurrent_checkpoints, snapshot_hybrid_state_indices,
};
use crate::pipeline::isq::{UqffFullSer, UqffWriteConfig};
use crate::pipeline::sampling::{sample_and_add_toks, sample_and_add_toks_batched};
use crate::pipeline::tokenizer::get_tokenizer;
use crate::pipeline::{Modalities, ModelForwardContext, SupportedModality};
use crate::prefix_cacher::PrefixCacheManagerV2;
use crate::sequence::Sequence;
use crate::utils::progress::ProgressScopeGuard;
use crate::{
    DeviceMapSetting, DynamicLoraRuntime, LoraAdapterSpec, LoraRuntimeConfig, PagedAttentionConfig,
    Pipeline, Topology, TryIntoDType,
};
use anyhow::Result;
use futures::{FutureExt, future::BoxFuture};
use inference_quant::IsqType;
use inference_tensor::{DType, Device, Tensor, Var};
use rand_isaac::Isaac64Rng;
use std::any::Any;
use std::path::PathBuf;
#[cfg(feature = "cuda")]
use std::sync::Mutex as StdMutex;
use std::sync::{Arc, RwLock};
use tokenizers::Tokenizer;
use tokio::sync::Mutex;
use tracing::{debug, trace};
#[cfg(feature = "cuda")]
use tracing::{info, warn};

const ADJACENT_PARTIAL_ROTARY_LORA: &str = "LoRA adapters are not supported when Q/K use adjacent RoPE pairs over part of each head (MLA or partial rotary); load the original safetensors model or omit the adapter";

pub struct NormalPipeline {
    model: Box<dyn NormalModel + Send + Sync>,
    core: DecoderCore,
}

fn normal_model_requires_uniform_prompt_batch(
    is_hybrid: bool,
    packed_prefill_available: bool,
    has_speculative_proposer: bool,
) -> bool {
    (is_hybrid && !packed_prefill_available)
        || (has_speculative_proposer && !packed_prefill_available)
}

/// A loader for a "normal" (non-quantized) model.
pub struct NormalLoader {
    inner: Box<dyn NormalModelLoader>,
    model_id: String,
    config: NormalSpecificConfig,
    lora_adapters: Option<Vec<LoraAdapterSpec>>,
    lora_runtime_config: Option<LoraRuntimeConfig>,
    kind: ModelKind,
    no_kv_cache: bool,
    chat_template: Option<String>,
    tokenizer_json: Option<String>,
    from_uqff: RwLock<Option<Vec<PathBuf>>>,
    jinja_explicit: Option<String>,
    hf_cache_path: Option<PathBuf>,
    prepared_source: Option<super::loading::PreparedSource>,
    mtp: bool,
}

pub(crate) fn new_dynamic_lora_registry(
    config: &str,
    rope_pairing: Option<crate::gguf::normal_registry::RopePairing>,
) -> Result<Arc<inference_quant::LoraLayerRegistry>> {
    let config = serde_json::from_str::<serde_json::Value>(config)?;
    let qwen35_moe_identity = config
        .get("architectures")
        .and_then(serde_json::Value::as_array)
        .and_then(|architectures| architectures.first())
        .and_then(serde_json::Value::as_str)
        == Some("Qwen3NextForCausalLM")
        && config
            .get(crate::gdn::GDN_V_HEAD_LAYOUT_CONFIG_KEY)
            .and_then(serde_json::Value::as_str)
            == Some("tiled");
    let registry = if qwen35_moe_identity {
        inference_quant::LoraLayerRegistry::new_with_site_prefix_alias(
            "model",
            "model.language_model",
        )?
    } else {
        inference_quant::LoraLayerRegistry::new()
    };
    let registry = match rope_pairing {
        Some(crate::gguf::normal_registry::RopePairing::Adjacent) => {
            // MLA and partial rotary pair only part of each head, which the per-head row map does not describe
            let partial_rotary = config.get("qk_rope_head_dim").is_some()
                || config
                    .get("partial_rotary_factor")
                    .and_then(serde_json::Value::as_f64)
                    .is_some_and(|factor| factor < 1.0);
            if partial_rotary {
                anyhow::bail!(ADJACENT_PARTIAL_ROTARY_LORA);
            }
            registry.with_adjacent_qk_rope(attention_head_dim(&config)?)?
        }
        _ => registry,
    };
    Ok(Arc::new(registry))
}

fn attention_head_dim(config: &serde_json::Value) -> Result<usize> {
    let field = |name: &str| config.get(name).and_then(serde_json::Value::as_u64);
    let head_dim = match (
        field("head_dim"),
        field("hidden_size"),
        field("num_attention_heads"),
    ) {
        (Some(head_dim), _, _) => head_dim,
        (None, Some(hidden), Some(heads)) if heads > 0 => hidden / heads,
        _ => anyhow::bail!("the model config has no head_dim, hidden_size or num_attention_heads"),
    };
    Ok(usize::try_from(head_dim)?)
}

#[derive(Default)]
/// A builder for a loader for a "normal" (non-quantized) model.
pub struct NormalLoaderBuilder {
    model_id: Option<String>,
    config: NormalSpecificConfig,
    lora_adapters: Option<Vec<LoraAdapterSpec>>,
    lora_runtime_config: Option<LoraRuntimeConfig>,
    kind: ModelKind,
    no_kv_cache: bool,
    chat_template: Option<String>,
    tokenizer_json: Option<String>,
    jinja_explicit: Option<String>,
    hf_cache_path: Option<PathBuf>,
    mtp: bool,
}

#[derive(Clone, Default)]
/// Config specific to loading a normal model.
pub struct NormalSpecificConfig {
    pub topology: Option<Topology>,
    pub organization: IsqOrganization,
    pub write_uqff: Option<UqffWriteConfig>,
    pub from_uqff: Option<Vec<PathBuf>>,
    pub imatrix: Option<PathBuf>,
    pub calibration_file: Option<PathBuf>,
    pub hf_cache_path: Option<PathBuf>,
    pub hf_config_overrides: Option<super::HfConfigOverrides>,
    pub max_model_len: Option<usize>,
    pub matformer_config_path: Option<PathBuf>,
    pub matformer_slice_name: Option<String>,
}

impl NormalLoaderBuilder {
    pub fn new(
        config: NormalSpecificConfig,
        chat_template: Option<String>,
        tokenizer_json: Option<String>,
        model_id: Option<String>,
        no_kv_cache: bool,
        jinja_explicit: Option<String>,
    ) -> Self {
        let hf_cache_path = config.hf_cache_path.clone();
        Self {
            config,
            chat_template,
            tokenizer_json,
            model_id,
            kind: ModelKind::Normal,
            jinja_explicit,
            no_kv_cache,
            hf_cache_path,
            ..Default::default()
        }
    }

    /// Load the MTP head built into the checkpoint so it can drive speculative decoding.
    pub fn with_mtp(mut self, mtp: bool) -> Self {
        self.mtp = mtp;
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

    /// If the loader type is not specified, loader type is automatically determined from the
    /// `architectures` array in the config.
    fn build_inner(
        self,
        loader_tp: Option<NormalLoaderType>,
        prepared_source: Option<super::loading::PreparedSource>,
    ) -> anyhow::Result<NormalLoader> {
        super::validate_lora_loader_config(
            self.lora_adapters.as_deref(),
            self.lora_runtime_config,
        )?;
        let loader: Box<dyn NormalModelLoader> = match loader_tp {
            Some(tp) => tp.loader()?,
            None => Box::new(AutoNormalLoader),
        };
        Ok(NormalLoader {
            inner: loader,
            model_id: self.model_id.unwrap(),
            config: self.config,
            lora_adapters: self.lora_adapters,
            lora_runtime_config: self.lora_runtime_config,
            kind: self.kind,
            no_kv_cache: self.no_kv_cache,
            chat_template: self.chat_template,
            tokenizer_json: self.tokenizer_json,
            jinja_explicit: self.jinja_explicit,
            from_uqff: RwLock::new(None),
            hf_cache_path: self.hf_cache_path,
            prepared_source,
            mtp: self.mtp,
        })
    }

    pub fn build(self, loader_tp: Option<NormalLoaderType>) -> anyhow::Result<Box<dyn Loader>> {
        Ok(Box::new(self.build_inner(loader_tp, None)?))
    }

    pub(crate) fn build_with_source(
        mut self,
        loader_tp: NormalLoaderType,
        source: super::loading::PreparedSource,
        kind: ModelKind,
    ) -> anyhow::Result<Box<dyn Loader>> {
        self.kind = kind;
        Ok(Box::new(self.build_inner(Some(loader_tp), Some(source))?))
    }
}

impl Loader for NormalLoader {
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
        mut paged_attn_config: Option<PagedAttentionConfig>,
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>> {
        let _progress_guard = ProgressScopeGuard::new(silent);
        let config = super::loading::prepare_model_config(
            self.prepared_source.as_ref(),
            paths.get_config_filename(),
            self.config.from_uqff.is_some(),
            self.config.hf_config_overrides.as_ref(),
            self.mtp,
        )?;
        // The UQFF artifact keeps the checkpoint config; max_model_len and the like apply to this load only.
        let source_config = config;
        let config = self
            .inner
            .runtime_config(&source_config, self.config.max_model_len)?
            .into_owned();

        if !self.inner.supports_paged_attention(&config)? {
            paged_attn_config = None;
        }

        debug!("Prompt chunk size is {ATTENTION_CHUNK_SIZE}.");

        let matformer = super::loading::load_matformer_slice(
            self.config.matformer_config_path.as_deref(),
            self.config.matformer_slice_name.as_deref(),
        )?;
        let (session, mapper) = super::loading::open_load_session(
            super::loading::LoadSessionInputs {
                mapped: &*self.inner,
                isq: &*self.inner,
                config: &config,
                settings: super::loading::LoadSettings {
                    topology: self.config.topology.as_ref(),
                    organization: self.config.organization,
                    write_uqff: self.config.write_uqff.as_ref(),
                    from_uqff: self.config.from_uqff.is_some(),
                    has_imatrix: self.config.imatrix.is_some(),
                    has_calibration: self.config.calibration_file.is_some(),
                },
                paths,
                device,
                dtype,
                mapper,
                in_situ_quant,
                uqff_files: self.from_uqff.read().unwrap().as_deref(),
                prepared: self.prepared_source.as_ref(),
                has_lora: self.lora_adapters.is_some(),
                matformer,
                matformer_sizing: false,
                non_mapped_unpacked: false,
                auto_device_map_params: None,
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
                config: &config,
                paths,
                silent,
                organization: self.config.organization,
                from_uqff: self.config.from_uqff.is_some(),
                write_uqff: self.config.write_uqff.is_some(),
                prepared: self.prepared_source.as_ref(),
                lora: super::loading::lora_runtime(&self.kind, self.lora_runtime_config),
            },
        )?;
        let super::loading::LoadSession {
            device,
            weight_source,
            max_kv_tokens,
            pipeline_mapper,
            layer_devices,
            dtype,
            plan,
            ..
        } = session;
        let load_device = plan.load_device.clone();

        let tokenizer = match self.prepared_source.as_ref() {
            Some(source) => source.tokenizer.clone(),
            None => get_tokenizer(paths.get_tokenizer_filename(), None)?,
        };
        let gen_conf = super::loading::generation_config(
            self.prepared_source
                .as_ref()
                .map(|source| source.generation_config.clone()),
            paths,
            &config,
        );

        let chat_template = super::loading::load_chat_template(
            paths,
            self.jinja_explicit.as_ref(),
            self.chat_template.as_ref(),
            self.prepared_source.as_ref(),
        );

        // cloned out so the tracker lock is not held through calibration and the UQFF write
        let tracked = tracker.get().clone();
        super::isq_flow::finish_isq_load(super::isq_flow::FinishIsqLoad {
            plan: &plan,
            modules: tracked,
            drive: &super::isq_flow::NormalCalibrationDrive(&*model),
            in_situ_quant,
            imatrix: self.config.imatrix.as_ref(),
            calibration_file: self.config.calibration_file.as_ref(),
            calibration: super::isq_flow::CalibrationCtx {
                tokenizer: &tokenizer,
                bos_tok_id: chat_template
                    .bos_tok()
                    .as_deref()
                    .and_then(|tok| tokenizer.token_to_id(tok)),
                load_device: &load_device,
                mapper: Some(pipeline_mapper.as_ref()),
            },
            uqff: self
                .config
                .write_uqff
                .as_ref()
                .map(|write| super::isq_flow::UqffArtifact {
                    config: write,
                    residual: super::loading::uqff_residual_tensors(
                        self.config.organization,
                        &*model,
                    ),
                    full_ser: UqffFullSer {
                        tokenizer: &tokenizer,
                        template_filename: paths.get_template_filename(),
                        effective_chat_template: Some(&chat_template),
                        generation_config: super::loading::uqff_generation_config_file(
                            paths,
                            self.prepared_source.as_ref(),
                        ),
                        config: source_config.clone(),
                        processor_filename: &None,
                        preprocessor_filename: &None,
                        modules: None,
                        module_paths: None,
                    },
                }),
        })?;

        let tracked_modules = tracker.get().clone();
        let source_weight_files = super::loading::source_weight_files(
            self.prepared_source.as_ref(),
            self.config.from_uqff.is_some(),
            paths.get_weight_filenames(),
        );

        let core = DecoderCore::new(DecoderCoreArgs {
            model: LoadedModelView {
                target: &*model,
                cache: model.cache(),
                config: model.model_config(),
                max_seq_len: model.max_seq_len(),
                sliding_window: model.config().sliding_window,
                block_diffusion: false,
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
            no_kv_cache: self.no_kv_cache,
            no_prefix_cache: false,
            kind: self.kind.clone(),
            model_id: self.model_id.clone(),
            modalities: Modalities {
                input: vec![SupportedModality::Text],
                output: vec![SupportedModality::Text],
            },
            loaded_for_uqff_write: self.config.write_uqff.is_some(),
            tracked_modules,
            source_weight_files,
            source_weight_source: weight_source,
            dynamic_lora,
        })?;
        Ok(Arc::new(Mutex::new(NormalPipeline { model, core })))
    }

    fn get_id(&self) -> String {
        self.model_id.clone()
    }

    fn get_kind(&self) -> ModelKind {
        self.kind.clone()
    }
}

impl PreProcessingMixin for NormalPipeline {
    fn get_chat_template(&self) -> Option<Arc<ChatTemplate>> {
        Some(self.core.chat_template.clone())
    }
    fn get_input_processor_config(&self) -> Option<Arc<dyn Any>> {
        None
    }
}

impl IsqPipelineMixin for NormalPipeline {
    fn re_isq_model(&mut self, dtype: IsqType) -> Result<()> {
        if !self.core.tracked_modules.is_empty() {
            self.cleanup_cuda_graphs();
        }
        super::isq_flow::requantize_tracked_modules(&self.core.tracked_modules, dtype)
    }

    fn begin_calibration(&mut self) -> Result<()> {
        super::isq_flow::begin_calibration(&self.core.tracked_modules)?;
        #[cfg(feature = "cuda")]
        self.core
            .cuda_decode_graph
            .lock()
            .expect("CUDA graph mutex poisoned")
            .suspend();
        Ok(())
    }

    fn calibration_status(&self) -> Result<super::isq_flow::CalibrationStatus> {
        Ok(super::isq_flow::calibration_status(
            &self.core.tracked_modules,
        ))
    }

    fn apply_calibration(
        &mut self,
        save_cimatrix: Option<std::path::PathBuf>,
    ) -> Result<super::isq_flow::CalibrationStatus> {
        self.cleanup_cuda_graphs();
        let result = super::isq_flow::apply_calibration(
            &self.core.tracked_modules,
            &self.core.source_weight_files,
            self.core.source_weight_source.as_deref(),
            save_cimatrix.as_deref(),
        );
        #[cfg(feature = "cuda")]
        if result.is_ok()
            || !super::isq_flow::calibration_status(&self.core.tracked_modules).collecting
        {
            self.core
                .cuda_decode_graph
                .lock()
                .expect("CUDA graph mutex poisoned")
                .resume();
        }
        result
    }
}

impl CacheManagerMixin for NormalPipeline {
    fn clone_in_cache(&self, seqs: &mut [&mut Sequence]) -> inference_tensor::Result<()> {
        super::cache_manager::clone_in_cache_by_kind(self, seqs)
    }
    fn clone_out_cache(&self, seqs: &mut [&mut Sequence]) {
        super::cache_manager::clone_out_cache_by_kind(self, seqs)
    }
    fn set_none_cache(
        &self,
        seqs: &mut [&mut Sequence],
        modify_draft_cache: bool,
        load_preallocated_cache: bool,
    ) -> inference_tensor::Result<()> {
        super::cache_manager::set_none_cache_by_kind(
            self,
            seqs,
            modify_draft_cache,
            load_preallocated_cache,
        )?;
        Ok(())
    }
    fn cache(&self) -> &EitherCache {
        self.model.cache()
    }
}

impl MetadataMixin for NormalPipeline {
    fn device(&self) -> Device {
        self.model.device().clone()
    }
    fn tokenizer(&self) -> Option<Arc<Tokenizer>> {
        Some(self.core.tokenizer.clone())
    }
    fn name(&self) -> String {
        self.core.model_id.clone()
    }
    fn cleanup_cuda_graphs(&self) {
        #[cfg(feature = "cuda")]
        super::cuda_graph::clear_decode_graphs(&self.core.cuda_decode_graph, self.model.cache());
    }
    fn reclaim_cuda_graph_memory(&self, max_entries: usize) -> usize {
        #[cfg(feature = "cuda")]
        {
            super::cuda_graph::reclaim_decode_graphs(
                &self.core.cuda_decode_graph,
                &*self.model,
                max_entries,
            )
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = max_entries;
            0
        }
    }
    fn precapture_cuda_decode_graphs(&self, ctx: &DecodeGraphPrecaptureCtx) {
        #[cfg(feature = "cuda")]
        {
            if let Err(err) = self.precapture_cuda_decode_graphs_impl(ctx) {
                self.core
                    .cuda_decode_graph
                    .lock()
                    .expect("CUDA graph mutex poisoned")
                    .clear();
                warn!("CUDA decode graph precapture failed, graphs will be captured lazily: {err}");
            }
            if let Err(err) = self.model.precapture_speculative_cuda_graphs() {
                warn!(
                    "Speculative CUDA graph precapture failed, graphs will be captured lazily: {err}"
                );
            }
        }
        #[cfg(not(feature = "cuda"))]
        let _ = ctx;
    }
    fn get_metadata(&self) -> Arc<GeneralMetadata> {
        self.core.metadata.clone()
    }
    fn generation_defaults(&self) -> Option<crate::ModelGenerationDefaults> {
        self.core.generation_defaults.clone()
    }
    fn device_mapper(&self) -> Option<&dyn DeviceMapper> {
        Some(&*self.core.mapper)
    }
}

impl crate::speculative::driver::SpeculativePipelineExt for NormalPipeline {
    fn speculative_target(&self) -> &dyn inference_nn::speculative::SpeculativeTargetMixin {
        &*self.model
    }

    fn speculative_target_mut(
        &mut self,
    ) -> &mut dyn inference_nn::speculative::SpeculativeTargetMixin {
        &mut *self.model
    }

    #[cfg(feature = "cuda")]
    fn cuda_sparse_rejection_workspace(
        &self,
    ) -> &StdMutex<Option<crate::speculative::CudaSparseRejectionWorkspace>> {
        &self.core.cuda_sparse_rejection
    }
}

#[cfg(feature = "cuda")]
impl NormalPipeline {
    fn try_cuda_decode_graph_forward(
        &self,
        input: CudaDecodeGraphForwardInput<'_>,
    ) -> inference_tensor::Result<Option<CudaDecodeGraphReplay>> {
        let CudaDecodeGraphForwardInput {
            input_ids,
            seqlen_offsets,
            context_lens,
            position_ids,
            paged_attn_meta,
            flash_meta,
            recurrent_batch_kind,
        } = input;
        if !cuda_decode_graphs_enabled() {
            record_cuda_graph_dispatch(
                CudaGraphComponent::Target,
                CudaGraphDispatchMode::Skipped,
                CudaGraphDispatchReason::Disabled,
            );
            return Ok(None);
        }
        if !cuda_decode_graph_batch_kind_supported(recurrent_batch_kind) {
            record_cuda_graph_dispatch(
                CudaGraphComponent::Target,
                CudaGraphDispatchMode::Skipped,
                CudaGraphDispatchReason::Prefill,
            );
            return Ok(None);
        }
        if !self.model.supports_cuda_decode_graphs()
            || !cuda_decode_graph_supported_for_model(self.core.metadata.model_metadata.as_deref())
        {
            record_cuda_graph_dispatch(
                CudaGraphComponent::Target,
                CudaGraphDispatchMode::Skipped,
                CudaGraphDispatchReason::ModelUnsupported,
            );
            return Ok(None);
        }
        if self.model.has_speculative_proposer() {
            record_cuda_graph_dispatch(
                CudaGraphComponent::Target,
                CudaGraphDispatchMode::Skipped,
                CudaGraphDispatchReason::SpeculativeConflict,
            );
            return Ok(None);
        }
        let Some((kv_cache, metadata)) = paged_attn_meta else {
            record_cuda_graph_dispatch(
                CudaGraphComponent::Target,
                CudaGraphDispatchMode::Skipped,
                CudaGraphDispatchReason::PagedAttentionUnavailable,
            );
            return Ok(None);
        };
        if metadata.is_first_prompt_chunk || metadata.num_cached_tokens.is_some() {
            record_cuda_graph_dispatch(
                CudaGraphComponent::Target,
                CudaGraphDispatchMode::Skipped,
                CudaGraphDispatchReason::Prefill,
            );
            return Ok(None);
        }
        if metadata.decode_rows.is_none() {
            record_cuda_graph_dispatch(
                CudaGraphComponent::Target,
                CudaGraphDispatchMode::Skipped,
                CudaGraphDispatchReason::IncompatibleShape,
            );
            return Ok(None);
        }
        let (batch, q_len) = input_ids.dims2()?;
        if q_len != 1
            || seqlen_offsets.len() != batch
            || context_lens.len() != batch
            || position_ids.len() != batch
            || !input_ids.device().is_cuda()
        {
            record_cuda_graph_dispatch(
                CudaGraphComponent::Target,
                CudaGraphDispatchMode::Skipped,
                CudaGraphDispatchReason::IncompatibleShape,
            );
            return Ok(None);
        }
        let Some(bucket) = cuda_graph_batch_bucket(CudaGraphComponent::Target, q_len, batch) else {
            record_cuda_graph_dispatch(
                CudaGraphComponent::Target,
                CudaGraphDispatchMode::Eager,
                CudaGraphDispatchReason::BatchUnsupported,
            );
            return Ok(None);
        };
        let Some(_) = self.core.metadata.cache_config.as_ref() else {
            record_cuda_graph_dispatch(
                CudaGraphComponent::Target,
                CudaGraphDispatchMode::Skipped,
                CudaGraphDispatchReason::CacheConfigUnavailable,
            );
            return Ok(None);
        };
        // Captured kernels require canonical strides, but an already contiguous input needs no copy.
        let input_ids = &input_ids.contiguous()?;

        let mut state = self
            .core
            .cuda_decode_graph
            .lock()
            .expect("CUDA graph mutex poisoned");
        if state.disabled() {
            record_cuda_graph_dispatch(
                CudaGraphComponent::Target,
                CudaGraphDispatchMode::Eager,
                CudaGraphDispatchReason::RuntimeDisabled,
            );
            return Ok(None);
        }
        let hybrid_slots = if self.model.cache().is_hybrid() {
            let slots = hybrid_graph_slots(&mut self.model.cache().hybrid())?;
            if let Some(slots) = &slots {
                state.observe_recurrent_storage_generation(slots.storage_generation);
            }
            slots
        } else {
            None
        };
        let Some(step) = CudaGraphDecodeStep::padded(
            CudaGraphDecodeStepInputs {
                input_ids,
                seqlen_offsets,
                context_lens,
                position_ids,
                metadata,
                state_indices: hybrid_slots.as_ref().map(|slots| slots.real.as_slice()),
                pad_slot: hybrid_slots.as_ref().map(|_| GDN_PAD_SLOT),
            },
            bucket,
        )?
        else {
            record_cuda_graph_dispatch(
                CudaGraphComponent::Target,
                CudaGraphDispatchMode::Eager,
                CudaGraphDispatchReason::PaddingUnavailable,
            );
            return Ok(None);
        };
        let key = CudaDecodeGraphKey::new(&step.input_ids, &step.metadata, recurrent_batch_kind)?;
        if let Some(replay) = state.replay(&key, &step, CudaDecodeGraphReplayInput::Host)? {
            return Ok(Some(replay));
        }

        let replay_key = key.clone();
        let _ = self.capture_cuda_decode_graph_step(
            &mut state,
            key,
            &step,
            CudaDecodeGraphCaptureInputs {
                kv_cache: kv_cache.as_slice(),
                flash_meta,
                recurrent_batch_kind,
            },
            true,
        )?;
        super::synchronize_cuda_contexts(step.input_ids.device(), self.core.mapper.as_ref())
            .map_err(|err| {
                inference_tensor::Error::msg(format!(
                    "CUDA graph rollback synchronization failed: {err}"
                ))
            })?;
        let replay = state
            .replay(&replay_key, &step, CudaDecodeGraphReplayInput::Host)?
            .ok_or_else(|| {
                inference_tensor::Error::msg("newly captured CUDA decode graph was not replayable")
            })?;
        record_cuda_graph_dispatch(
            CudaGraphComponent::Target,
            CudaGraphDispatchMode::Eager,
            CudaGraphDispatchReason::CachePopulation,
        );
        Ok(Some(replay))
    }

    fn precapture_cuda_decode_graphs_impl(
        &self,
        ctx: &DecodeGraphPrecaptureCtx,
    ) -> inference_tensor::Result<()> {
        let device = self.device();
        if !cuda_decode_graphs_enabled()
            || !device.is_cuda()
            || !self.model.supports_cuda_decode_graphs()
            || !cuda_decode_graph_supported_for_model(self.core.metadata.model_metadata.as_deref())
            || self.model.has_speculative_proposer()
        {
            return Ok(());
        }
        let (Some(_), Some(cache_engine)) = (
            &self.core.metadata.cache_config,
            &self.core.metadata.cache_engine,
        ) else {
            return Ok(());
        };
        let kv_cache = cache_engine.get_kv_cache().clone();
        let hybrid_slots = if self.model.cache().is_hybrid() {
            let mut cache = self.model.cache().hybrid();
            let Some(pad_slot) = cache.graph_pad_slot()? else {
                return Ok(());
            };
            let pad_slot = cache.active_physical_slot(pad_slot)?;
            let pad_slot = u32::try_from(pad_slot).map_err(|_| {
                inference_tensor::Error::msg(format!(
                    "recurrent graph pad slot {pad_slot} exceeds u32"
                ))
            })?;
            Some(pad_slot)
        } else {
            None
        };
        let mut state = self
            .core
            .cuda_decode_graph
            .lock()
            .expect("CUDA graph mutex poisoned");
        if state.disabled() {
            return Ok(());
        }
        let start = std::time::Instant::now();
        let mut captured = 0usize;
        let inputs = CudaGraphPrecaptureInputs::new(ctx, 1, &device, self.device_mapper())?;
        let live = hybrid_slots.map(|pad_slot| vec![pad_slot]);
        let max_bucket =
            cuda_graph_precapture_max_batch(CudaGraphComponent::Target, 1, ctx.max_batch_size);
        for bucket in cuda_graph_precapture_batches(CudaGraphComponent::Target, 1)
            .filter(|bucket| *bucket <= max_bucket)
        {
            let Some(step) = CudaGraphDecodeStep::padded(
                inputs.step_inputs(live.as_deref(), hybrid_slots.map(|_| GDN_PAD_SLOT)),
                bucket,
            )?
            else {
                continue;
            };
            let key = CudaDecodeGraphKey::new(
                &step.input_ids,
                &step.metadata,
                RecurrentBatchKind::Decode,
            )?;
            if state.contains(&key) {
                continue;
            }
            self.capture_cuda_decode_graph_step(
                &mut state,
                key,
                &step,
                CudaDecodeGraphCaptureInputs {
                    kv_cache: kv_cache.as_slice(),
                    flash_meta: &inputs.flash_meta,
                    recurrent_batch_kind: RecurrentBatchKind::Decode,
                },
                false,
            )?;
            captured += 1;
        }
        if captured > 0 {
            info!(
                "Captured {captured} CUDA decode graphs through batch bucket {} in {:.2?}",
                max_bucket,
                start.elapsed()
            );
        }
        Ok(())
    }

    /// Captures after one eager warmup; live calls roll it back so the first replay is canonical.
    fn capture_cuda_decode_graph_step(
        &self,
        state: &mut CudaDecodeGraphState,
        key: CudaDecodeGraphKey,
        step: &CudaGraphDecodeStep,
        inputs: CudaDecodeGraphCaptureInputs<'_>,
        rollback_live_state: bool,
    ) -> inference_tensor::Result<Tensor> {
        let CudaDecodeGraphCaptureInputs {
            kv_cache,
            flash_meta,
            recurrent_batch_kind,
        } = inputs;
        let graph_event =
            CudaGraphEventGuard::new(CudaGraphComponent::Target, CudaGraphEvent::Capture);
        let Device::Cuda(cuda_device) = step.input_ids.device() else {
            inference_tensor::bail!("CUDA graph decode expected CUDA input ids");
        };
        let _htod_cache_guard = cuda_device.enable_cuda_graph_htod_cache();
        let metadata = step
            .metadata
            .materialize_decode_tensors()
            .map_err(inference_tensor::Error::msg)?;

        let uses_recurrent_transition_log = self.model.cache().is_hybrid()
            && self.model.cache().hybrid().uses_recurrent_transition_log();
        if rollback_live_state
            && recurrent_batch_kind == RecurrentBatchKind::Decode
            && self.model.supports_recurrent_speculative_transitions()
            && uses_recurrent_transition_log
            && !self
                .model
                .apply_recurrent_speculative_transitions_for_current_batch()?
        {
            inference_tensor::bail!(
                "CUDA graph capture could not materialize pending recurrent transitions"
            );
        }

        let recurrent_snapshots = snapshot_hybrid_recurrent_checkpoints(
            self.model.cache(),
            &*self.model,
            recurrent_batch_kind,
        )?;
        let live_state_indices = snapshot_hybrid_state_indices(self.model.cache());
        let capture_attempt: inference_tensor::Result<_> = (|| {
            let state_index_buffers = match &step.state_indices {
                Some(host) => Some(install_hybrid_graph_state_indices(
                    &mut self.model.cache().hybrid(),
                    host,
                )?),
                None => None,
            };
            let mut ctx = ModelForwardContext::new(
                &step.seqlen_offsets,
                &step.context_lens,
                &step.position_ids,
                Some((kv_cache, &metadata)),
                flash_meta,
            )
            .with_recurrent_cache(self.model.cache(), recurrent_batch_kind);
            let warmup_logits = self.model.forward(&step.input_ids, &mut ctx)?;
            step.input_ids.device().synchronize()?;
            let live_logits = step.narrow_rows(&warmup_logits)?;

            // CUDA stream capture records recurrent writes without executing them.
            let entry = capture_cuda_decode_graph(
                CudaDecodeGraphCaptureCtx {
                    key,
                    input_ids: &step.input_ids,
                    seqlen_offsets: &step.seqlen_offsets,
                    position_ids: &step.position_ids,
                    kv_cache,
                    metadata: &metadata,
                    model_metadata: self.core.metadata.model_metadata.as_deref(),
                    activation_dtype: self.core.metadata.activation_dtype,
                    warmup_logits: &warmup_logits,
                    state_indices: state_index_buffers,
                    real_batch: step.real_batch,
                },
                |graph_input_ids, graph_metadata| {
                    let mut ctx = ModelForwardContext::new(
                        &step.seqlen_offsets,
                        &step.context_lens,
                        &step.position_ids,
                        Some((kv_cache, graph_metadata)),
                        flash_meta,
                    )
                    .with_recurrent_cache(self.model.cache(), recurrent_batch_kind);
                    self.model.forward(graph_input_ids, &mut ctx)
                },
            )?;
            Ok((live_logits, entry))
        })();
        let (logits, entry) = finish_cuda_graph_capture_attempt(
            self.model.cache(),
            state,
            capture_attempt,
            recurrent_snapshots.as_deref(),
            live_state_indices.as_ref(),
            rollback_live_state,
        )?;
        state.insert(entry);
        graph_event.success();
        Ok(logits)
    }
}

impl Pipeline for NormalPipeline {
    fn requires_uniform_prompt_batch(&self) -> bool {
        normal_model_requires_uniform_prompt_batch(
            self.model.cache().is_hybrid(),
            self.supports_packed_prefill(),
            self.model.has_speculative_proposer(),
        )
    }

    fn requires_uniform_completion_batch(&self) -> bool {
        false
    }

    fn supports_batched_cuda_sampling(&self) -> bool {
        !self.model.has_speculative_proposer()
    }

    fn supports_speculative_prompt_bootstrap(&self) -> bool {
        self.model.supports_speculative_prompt_bootstrap()
    }

    fn speculative_prefix_replay(&self) -> crate::speculative::SpeculativePrefixReplay {
        self.model.speculative_prefix_replay()
    }

    fn supports_paged_auxiliary_prefix_state(&self) -> bool {
        self.model.supports_paged_auxiliary_prefix_state()
    }

    fn capture_paged_auxiliary_prefix_state(
        &mut self,
        sequence_id: usize,
        cached_tokens: usize,
    ) -> inference_tensor::Result<Option<Arc<dyn crate::kv_cache::PagedAuxiliaryPrefixState>>> {
        self.model
            .capture_paged_auxiliary_prefix_state(sequence_id, cached_tokens)
    }

    fn restore_paged_auxiliary_prefix_state(
        &mut self,
        sequence_id: usize,
        cached_tokens: usize,
        state: &dyn crate::kv_cache::PagedAuxiliaryPrefixState,
    ) -> inference_tensor::Result<()> {
        self.model
            .restore_paged_auxiliary_prefix_state(sequence_id, cached_tokens, state)
    }

    fn supports_packed_prefill(&self) -> bool {
        self.model.supports_packed_prefill()
            && self.core.metadata.cache_engine.is_some()
            && (!self.model.has_speculative_proposer()
                || self.model.supports_speculative_packed_prefill())
            && self.model.device().is_cuda()
            && self
                .core
                .mapper
                .get_unique_devices()
                .iter()
                .all(Device::is_cuda)
            && crate::using_flash_attn()
            && crate::attention::flash_backend_supports_sdpa(
                self.model.config().k_head_dim,
                false,
                self.core.metadata.sliding_window.is_some(),
            )
            && matches!(
                self.core.metadata.activation_dtype,
                DType::F16 | DType::BF16
            )
    }

    fn adapter_runtime(&self) -> Option<Arc<DynamicLoraRuntime>> {
        self.core.dynamic_lora.clone()
    }

    fn forward_inputs(
        &mut self,
        inputs: Box<dyn Any>,
        return_raw_logits: bool,
    ) -> Result<ForwardInputsResult, inference_tensor::Error> {
        Ok(self.forward_step(inputs, return_raw_logits)?.output)
    }

    fn forward_step(
        &mut self,
        inputs: Box<dyn Any>,
        return_raw_logits: bool,
    ) -> Result<ForwardStepResult, inference_tensor::Error> {
        let ModelInputs {
            input_ids,
            seqlen_offsets,
            context_lens,
            position_ids,
            paged_attn_meta,
            flash_meta,
            recurrent_batch_kind,
            adapter_leases,
        } = *inputs.downcast().expect("Downcast failed.");
        let lora_execution = super::resolve_lora_execution(
            self.core.dynamic_lora.as_deref(),
            &input_ids,
            paged_attn_meta.as_ref(),
            &flash_meta,
            &adapter_leases,
        )?;
        let metadata = self.get_metadata();
        let paged_attn_meta = match (&metadata.cache_engine, &paged_attn_meta) {
            (Some(cache_engine), Some(meta)) => Some((cache_engine, meta)),
            (Some(_), None) => {
                // This can happen if Rust-side user code is wrong
                inference_tensor::bail!(
                    "Forward step expected a PagedAttention input metadata. This was not provided, please ensure that the scheduler config is correctly configured for PagedAttention."
                )
            }
            (None, Some(_)) => {
                // This should never happen but we handle it anyway
                inference_tensor::bail!(
                    "Forward step got a PagedAttention input metadata but there is no cache engine. Please raise an issue."
                )
            }
            (None, None) => None,
        };
        #[cfg(feature = "cuda")]
        let mut cuda_graph_eager_fallback = None;
        let paged_attn_meta = paged_attn_meta
            .as_ref()
            .map(|meta| (meta.0.get_kv_cache().clone(), meta.1.clone()));

        #[cfg(feature = "cuda")]
        if lora_execution.is_none() && !return_raw_logits {
            match self.try_cuda_decode_graph_forward(CudaDecodeGraphForwardInput {
                input_ids: &input_ids,
                seqlen_offsets: &seqlen_offsets,
                context_lens: &context_lens,
                position_ids: &position_ids,
                paged_attn_meta: paged_attn_meta.as_ref().map(|(a, b)| (a.clone(), b)),
                flash_meta: &flash_meta,
                recurrent_batch_kind,
            }) {
                Ok(Some(replay)) => {
                    return Ok(ForwardStepResult::cuda_decode(
                        ForwardInputsResult::CausalGeneration {
                            logits: replay.logits,
                        },
                        replay.launch,
                    ));
                }
                Ok(None) => {}
                Err(err) => {
                    if !disable_cuda_decode_graph(
                        &self.core.cuda_decode_graph,
                        self.model.cache(),
                        &err,
                    ) {
                        return Err(err);
                    }
                    cuda_graph_eager_fallback = Some(CudaGraphEventGuard::new(
                        CudaGraphComponent::Target,
                        CudaGraphEvent::EagerFallback,
                    ));
                }
            }
        }

        let paged_attn_meta = paged_attn_meta
            .map(|(kv_cache, metadata)| {
                metadata
                    .materialize_decode_tensors()
                    .map(|metadata| (kv_cache, metadata))
            })
            .transpose()
            .map_err(inference_tensor::Error::msg)?;

        let mut ctx = ModelForwardContext::new(
            &seqlen_offsets,
            &context_lens,
            &position_ids,
            paged_attn_meta
                .as_ref()
                .map(|(kv_cache, meta)| (kv_cache.as_slice(), meta)),
            &flash_meta,
        )
        .with_recurrent_cache(self.model.cache(), recurrent_batch_kind);
        let eager_result = inference_quant::with_lora_execution(lora_execution, || {
            self.model.forward(&input_ids, &mut ctx)
        });
        #[cfg(feature = "cuda")]
        if eager_result.is_ok()
            && let Some(graph_event) = cuda_graph_eager_fallback.take()
        {
            graph_event.success();
        }
        let logits = eager_result?;
        let output = if return_raw_logits {
            ForwardInputsResult::RawLogits { logits }
        } else {
            ForwardInputsResult::CausalGeneration { logits }
        };
        Ok(ForwardStepResult::eager(output))
    }

    #[cfg(feature = "cuda")]
    fn replay_cuda_decode_one_token(
        &mut self,
        launch: CudaDecodeGraphLaunch,
    ) -> inference_tensor::Result<Option<ForwardStepResult>> {
        let replay = {
            let mut state = self
                .core
                .cuda_decode_graph
                .lock()
                .expect("CUDA graph mutex poisoned");
            if state.disabled() {
                return Ok(None);
            }
            state.replay_one_token(launch)
        };
        match replay {
            Ok(Some(replay)) => Ok(Some(ForwardStepResult::cuda_decode(
                ForwardInputsResult::CausalGeneration {
                    logits: replay.logits,
                },
                replay.launch,
            ))),
            Ok(None) => Ok(None),
            Err(err) => {
                let _ = disable_cuda_decode_graph(
                    &self.core.cuda_decode_graph,
                    self.model.cache(),
                    &err,
                );
                Err(err)
            }
        }
    }

    fn attach_speculative(
        &mut self,
        config: crate::speculative::SpeculativeConfig,
    ) -> inference_tensor::Result<()> {
        self.attach_speculative_with_runtime(
            config,
            crate::speculative::MtpRuntimeConfig::default(),
        )
    }

    fn attach_speculative_with_runtime(
        &mut self,
        config: crate::speculative::SpeculativeConfig,
        runtime: crate::speculative::MtpRuntimeConfig,
    ) -> inference_tensor::Result<()> {
        if self.core.dynamic_lora.is_some() {
            inference_tensor::bail!("dynamic LoRA does not support speculative decoding");
        }
        if matches!(config, crate::speculative::SpeculativeConfig::Mtp(_))
            && self.get_metadata().cache_engine.is_none()
        {
            inference_tensor::bail!(
                "MTP speculative decoding currently requires PagedAttention for this pipeline."
            );
        }
        if matches!(config, crate::speculative::SpeculativeConfig::Mtp(_)) {
            self.cleanup_cuda_graphs();
            self.model.disable_recurrent_decode_deferred_storage()?;
        }
        let config = crate::speculative::resolve_speculative_model(config)?;
        if let Some(info) = self
            .model
            .attach_speculative_with_runtime(config, runtime)?
        {
            self.model.log_speculative_attach(&info);
        }
        Ok(())
    }

    fn release_speculative_sequences(&mut self, seq_ids: &[usize]) -> inference_tensor::Result<()> {
        self.model.release_speculative_sequences(seq_ids)
    }

    fn flush_recurrent_speculative_transitions(
        &self,
        seq_ids: &[usize],
    ) -> inference_tensor::Result<()> {
        self.model.flush_recurrent_speculative_transitions(seq_ids)
    }

    #[allow(clippy::too_many_arguments)]
    fn try_sample_speculative_causal_gen<'a>(
        &'a mut self,
        seqs: &'a mut [&mut Sequence],
        logits: &'a [Tensor],
        batched_logits: Option<&'a Tensor>,
        prefix_cacher: &'a mut PrefixCacheManagerV2,
        disable_eos_stop: bool,
        rng: Arc<std::sync::Mutex<Isaac64Rng>>,
        metadata: Option<crate::paged_attention::PagedAttentionMeta>,
        logger: &'a crate::IntervalLogger,
    ) -> BoxFuture<'a, inference_tensor::Result<bool>> {
        Box::pin(async move {
            if !self.model.has_speculative_proposer() {
                crate::speculative::driver::clear_staged_speculative_tokens(seqs);
                return Ok(false);
            }

            let general_metadata = self.get_metadata();
            if let Some(cache_engine) = general_metadata.cache_engine.as_ref() {
                let Some(metadata) = metadata else {
                    crate::speculative::driver::clear_staged_speculative_tokens(seqs);
                    return Ok(false);
                };
                let cache = crate::speculative::cache::PagedSpeculativeCacheAccess::new(
                    &metadata,
                    cache_engine,
                );
                return crate::speculative::driver::try_sample_speculative_causal_gen(
                    self,
                    seqs,
                    logits,
                    batched_logits,
                    prefix_cacher,
                    disable_eos_stop,
                    rng,
                    &cache,
                    logger,
                )
                .await;
            }

            crate::speculative::driver::clear_staged_speculative_tokens(seqs);
            Ok(false)
        })
    }

    fn try_sample_causal_gen_batched<'a>(
        &'a self,
        seqs: &'a mut [&mut Sequence],
        logits: &'a Tensor,
        prefix_cacher: &'a mut PrefixCacheManagerV2,
        disable_eos_stop: bool,
        rng: Arc<std::sync::Mutex<Isaac64Rng>>,
    ) -> BoxFuture<'a, Result<bool, inference_tensor::Error>> {
        if self.model.has_speculative_proposer() {
            return Box::pin(std::future::ready(Ok(false)));
        }
        crate::speculative::driver::clear_staged_speculative_tokens(seqs);
        Box::pin(
            sample_and_add_toks_batched(
                self,
                seqs,
                logits.clone(),
                prefix_cacher,
                disable_eos_stop,
                rng,
            )
            .map(|result| result.map(|()| true)),
        )
    }

    fn sample_causal_gen<'a>(
        &'a self,
        seqs: &'a mut [&mut Sequence],
        logits: Vec<Tensor>,
        prefix_cacher: &'a mut PrefixCacheManagerV2,
        disable_eos_stop: bool,
        rng: Arc<std::sync::Mutex<Isaac64Rng>>,
    ) -> BoxFuture<'a, Result<(), inference_tensor::Error>> {
        sample_and_add_toks(self, seqs, logits, prefix_cacher, disable_eos_stop, rng)
    }
    fn category(&self) -> ModelCategory {
        ModelCategory::Text
    }
}

impl AnyMoePipelineMixin for NormalPipeline {
    fn amoe_finish_training(
        &mut self,
        gate_model_id: Option<String>,
    ) -> inference_tensor::Result<()> {
        self.model.finish_training(gate_model_id)
    }
    fn amoe_layer_vars(&self) -> Vec<Vec<Var>> {
        self.model.get_vars()
    }
    fn amoe_base_model_trainable_params(&self) -> usize {
        self.model.trainable_params()
    }
    fn amoe_take_cached_gating_outputs(&mut self) -> Vec<Tensor> {
        self.model.take_cached_gating_outputs()
    }
    fn amoe_create_layers(
        &mut self,
        model_ids: Vec<String>,
        token: &TokenSource,
        revision: Option<String>,
        match_regex: &str,
        config: crate::amoe::AnyMoeConfig,
        dtype: inference_tensor::DType,
        dev: &Device,
        (prefix, mlp): (String, String),
        layers: Vec<usize>,
        expert_type: AnyMoeExpertType,
        silent: bool,
        gate_model_id: Option<String>,
    ) -> inference_tensor::Result<()> {
        let (vbs, gate_vb) = super::amoe::load_anymoe_weights(super::amoe::AnyMoeWeightSources {
            model_ids,
            token,
            revision,
            match_regex,
            dtype,
            dev,
            layers: &layers,
            silent,
            gate_model_id,
        })?;
        self.model
            .create_anymoe_layers(vbs, config, (prefix, mlp), layers, expert_type, gate_vb)
    }
    fn amoe_supported(&self) -> bool {
        self.model.amoe_supported()
    }
}

#[cfg(test)]
mod tests {
    use super::{new_dynamic_lora_registry, normal_model_requires_uniform_prompt_batch};
    use crate::LoraRuntimeConfig;
    use crate::pipeline::finish_dynamic_lora_runtime;
    use crate::pipeline::{AdapterPaths, LocalModelPaths};
    use inference_quant::{LoraLayerRegistry, LoraLinearSpec, LoraSiteKey};
    use inference_tensor::{DType, Device};
    use std::{path::PathBuf, sync::Arc};

    fn empty_lora_paths() -> LocalModelPaths<PathBuf> {
        LocalModelPaths {
            tokenizer_filename: PathBuf::new(),
            config_filename: PathBuf::new(),
            template_filename: None,
            filenames: Vec::new(),
            adapter_paths: AdapterPaths::Lora(Vec::new()),
            gen_conf: None,
            preprocessor_config: None,
            video_preprocessor_config: None,
            processor_config: None,
            chat_template_json_filename: None,
        }
    }

    #[test]
    fn hybrid_models_require_uniform_prompts_until_packed_prefill_is_proven() {
        assert!(normal_model_requires_uniform_prompt_batch(
            true, false, false
        ));
        assert!(!normal_model_requires_uniform_prompt_batch(
            true, true, false
        ));
        assert!(!normal_model_requires_uniform_prompt_batch(
            false, false, false
        ));
    }

    #[test]
    fn unsupported_speculative_models_remain_uniform() {
        assert!(normal_model_requires_uniform_prompt_batch(
            true, false, true
        ));
        assert!(!normal_model_requires_uniform_prompt_batch(
            true, true, true
        ));
    }

    #[test]
    fn prepared_lora_runtime_finalizes_sites_and_preserves_update_policy() {
        let paths = empty_lora_paths();
        for live_updates in [false, true] {
            let layers = Arc::new(LoraLayerRegistry::new());
            let runtime = finish_dynamic_lora_runtime(
                &paths,
                layers.clone(),
                LoraRuntimeConfig::default(),
                live_updates,
            )
            .unwrap();

            assert_eq!(runtime.supports_live_updates(), live_updates);
            let error = layers
                .register(
                    LoraSiteKey::new("model.layers.0.self_attn.q_proj"),
                    LoraLinearSpec::replicated(2, 2),
                    DType::F32,
                    Device::Cpu,
                )
                .unwrap_err();
            assert!(error.to_string().contains("after registry finalization"));
        }
    }

    #[test]
    fn persisted_qwen35_moe_config_restores_lora_namespace_alias() {
        let config = r#"{
            "architectures":["Qwen3NextForCausalLM"],
            "_inference_gdn_v_head_layout":"tiled"
        }"#;
        let registry = new_dynamic_lora_registry(config, None).unwrap();
        let site = registry
            .register(
                LoraSiteKey::new("model.layers.0.self_attn.q_proj"),
                LoraLinearSpec::replicated(2, 2),
                DType::F32,
                Device::Cpu,
            )
            .unwrap();

        assert_eq!(
            site.key().path(),
            "model.language_model.layers.0.self_attn.q_proj"
        );
    }

    #[test]
    fn dense_qwen35_config_does_not_alias_lora_namespace() {
        let registry = new_dynamic_lora_registry(
            r#"{
                "architectures":["Qwen3_5ForCausalLM"],
                "_inference_gdn_v_head_layout":"tiled"
            }"#,
            None,
        )
        .unwrap();
        let site = registry
            .register(
                LoraSiteKey::new("model.layers.0.self_attn.q_proj"),
                LoraLinearSpec::replicated(2, 2),
                DType::F32,
                Device::Cpu,
            )
            .unwrap();

        assert_eq!(site.key().path(), "model.layers.0.self_attn.q_proj");
    }

    #[test]
    fn adjacent_rope_lora_refuses_heads_rotated_only_in_part() {
        let adjacent = Some(crate::gguf::normal_registry::RopePairing::Adjacent);
        for config in [
            r#"{"head_dim":16,"qk_rope_head_dim":8}"#,
            r#"{"head_dim":16,"partial_rotary_factor":0.5}"#,
        ] {
            let error = new_dynamic_lora_registry(config, adjacent)
                .err()
                .unwrap()
                .to_string();
            assert!(error.contains("original safetensors model"), "{error}");
        }
        assert!(
            new_dynamic_lora_registry(r#"{"head_dim":16,"partial_rotary_factor":1.0}"#, adjacent)
                .is_ok()
        );
        assert!(new_dynamic_lora_registry(r#"{"qk_rope_head_dim":8}"#, None).is_ok());
    }
}
