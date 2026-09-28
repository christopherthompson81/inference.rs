//! Multi-model builder and pipeline construction utilities.

use candle_core::Device;
use inference_core::{
    plan_paged_kv, AddModelConfig, EngineConfig, IsqType, PagedAttentionConfig,
    PagedKvModelRequest, Pipeline, SchedulerConfig, SearchCallback, SearchEmbeddingModel,
    ToolCallbackWithTool,
};
use std::{collections::HashMap, path::PathBuf, sync::Arc};
use tokio::sync::Mutex;

use crate::Model;

/// Enum representing all possible model builders that can be used with [`MultiModelBuilder`].
pub enum AnyModelBuilder {
    /// A text model builder.
    Text(crate::TextModelBuilder),
    /// A multimodal model builder.
    Multimodal(crate::MultimodalModelBuilder),
    /// An auto-detecting model builder.
    Auto(crate::ModelBuilder),
    /// A GGUF model builder.
    Gguf(crate::GgufModelBuilder),
    /// A diffusion (image generation) model builder.
    Diffusion(crate::DiffusionModelBuilder),
    /// A speech synthesis model builder.
    Speech(crate::SpeechModelBuilder),
    /// An embedding model builder.
    Embedding(crate::EmbeddingModelBuilder),
}

impl AnyModelBuilder {
    /// Get the default model ID for this builder.
    pub fn model_id(&self) -> String {
        match self {
            AnyModelBuilder::Text(b) => b.model_id.clone(),
            AnyModelBuilder::Multimodal(b) => b.model_id.clone(),
            AnyModelBuilder::Auto(b) => b.model_id.clone(),
            AnyModelBuilder::Gguf(b) => b.model_id.clone(),
            AnyModelBuilder::Diffusion(b) => b.model_id.clone(),
            AnyModelBuilder::Speech(b) => b.model_id.clone(),
            AnyModelBuilder::Embedding(b) => b.model_id.clone(),
        }
    }

    /// Build the pipeline and configuration for this model.
    pub async fn build_pipeline(
        self,
    ) -> anyhow::Result<(Arc<Mutex<dyn Pipeline>>, SchedulerConfig, AddModelConfig)> {
        match self {
            AnyModelBuilder::Text(b) => build_text_pipeline(b).await,
            AnyModelBuilder::Multimodal(b) => build_multimodal_pipeline(b).await,
            AnyModelBuilder::Auto(b) => build_auto_pipeline(b).await,
            AnyModelBuilder::Gguf(b) => build_gguf_pipeline(b).await,
            AnyModelBuilder::Diffusion(b) => build_diffusion_pipeline(b).await,
            AnyModelBuilder::Speech(b) => build_speech_pipeline(b).await,
            AnyModelBuilder::Embedding(b) => build_embedding_pipeline(b).await,
        }
    }

    fn paged_attn_cfg(&self) -> Option<PagedAttentionConfig> {
        match self {
            AnyModelBuilder::Text(b) => b.paged_attn_cfg,
            AnyModelBuilder::Multimodal(b) => b.paged_attn_cfg,
            AnyModelBuilder::Auto(b) => b.paged_attn_cfg,
            AnyModelBuilder::Gguf(b) => b.paged_attn_cfg,
            AnyModelBuilder::Diffusion(_)
            | AnyModelBuilder::Speech(_)
            | AnyModelBuilder::Embedding(_) => None,
        }
    }

    fn max_num_seqs(&self) -> usize {
        match self {
            AnyModelBuilder::Text(b) => b.max_num_seqs,
            AnyModelBuilder::Multimodal(b) => b.max_num_seqs,
            AnyModelBuilder::Auto(b) => b.max_num_seqs,
            AnyModelBuilder::Gguf(b) => b.max_num_seqs,
            AnyModelBuilder::Diffusion(b) => b.max_num_seqs,
            AnyModelBuilder::Speech(b) => b.max_num_seqs,
            AnyModelBuilder::Embedding(b) => b.max_num_seqs,
        }
    }

    fn with_paged_attn_cfg(mut self, paged_attn_cfg: Option<PagedAttentionConfig>) -> Self {
        match &mut self {
            AnyModelBuilder::Text(b) => b.paged_attn_cfg = paged_attn_cfg,
            AnyModelBuilder::Multimodal(b) => b.paged_attn_cfg = paged_attn_cfg,
            AnyModelBuilder::Auto(b) => b.paged_attn_cfg = paged_attn_cfg,
            AnyModelBuilder::Gguf(b) => b.paged_attn_cfg = paged_attn_cfg,
            AnyModelBuilder::Diffusion(_)
            | AnyModelBuilder::Speech(_)
            | AnyModelBuilder::Embedding(_) => {}
        }
        self
    }
}

// Conversion implementations
impl From<crate::TextModelBuilder> for AnyModelBuilder {
    fn from(b: crate::TextModelBuilder) -> Self {
        AnyModelBuilder::Text(b)
    }
}

impl From<crate::MultimodalModelBuilder> for AnyModelBuilder {
    fn from(b: crate::MultimodalModelBuilder) -> Self {
        AnyModelBuilder::Multimodal(b)
    }
}

impl From<crate::ModelBuilder> for AnyModelBuilder {
    fn from(b: crate::ModelBuilder) -> Self {
        AnyModelBuilder::Auto(b)
    }
}

impl From<crate::GgufModelBuilder> for AnyModelBuilder {
    fn from(b: crate::GgufModelBuilder) -> Self {
        AnyModelBuilder::Gguf(b)
    }
}

impl From<crate::DiffusionModelBuilder> for AnyModelBuilder {
    fn from(b: crate::DiffusionModelBuilder) -> Self {
        AnyModelBuilder::Diffusion(b)
    }
}

impl From<crate::SpeechModelBuilder> for AnyModelBuilder {
    fn from(b: crate::SpeechModelBuilder) -> Self {
        AnyModelBuilder::Speech(b)
    }
}

impl From<crate::EmbeddingModelBuilder> for AnyModelBuilder {
    fn from(b: crate::EmbeddingModelBuilder) -> Self {
        AnyModelBuilder::Embedding(b)
    }
}

struct MultiModelEntry {
    builder: AnyModelBuilder,
    alias: Option<String>,
}

/// Builder for creating a Model with multiple models.
pub struct MultiModelBuilder {
    builders: Vec<MultiModelEntry>,
    default_model_id: Option<String>,
}

impl Default for MultiModelBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl MultiModelBuilder {
    /// Create a new MultiModelBuilder.
    pub fn new() -> Self {
        Self {
            builders: Vec::new(),
            default_model_id: None,
        }
    }

    /// Add a model. The model ID will be the pipeline name (e.g., "google/gemma-4-E4B-it").
    pub fn add_model<B: Into<AnyModelBuilder>>(mut self, builder: B) -> Self {
        self.builders.push(MultiModelEntry {
            builder: builder.into(),
            alias: None,
        });
        self
    }

    /// Add a model with a custom alias (nickname) used for API requests.
    pub fn add_model_with_alias<B: Into<AnyModelBuilder>>(
        mut self,
        alias: impl Into<String>,
        builder: B,
    ) -> Self {
        self.builders.push(MultiModelEntry {
            builder: builder.into(),
            alias: Some(alias.into()),
        });
        self
    }

    /// Set the default model by its model ID or alias.
    pub fn with_default_model(mut self, model_id: impl ToString) -> Self {
        self.default_model_id = Some(model_id.to_string());
        self
    }

    /// Build the multi-model Model instance.
    pub async fn build(self) -> anyhow::Result<Model> {
        if self.builders.is_empty() {
            anyhow::bail!("MultiModelBuilder requires at least one model to be added");
        }

        inference_core::distributed::begin_tensor_parallel_session(self.builders.len())?;

        let entries = self.builders;
        let paged_kv_plan = plan_paged_kv(
            &entries
                .iter()
                .map(|entry| PagedKvModelRequest {
                    paged_attn: entry.builder.paged_attn_cfg(),
                    max_num_seqs: entry.builder.max_num_seqs(),
                })
                .collect::<Vec<_>>(),
            Default::default(),
        )?;
        let entries = entries
            .into_iter()
            .zip(paged_kv_plan.paged_attn)
            .map(|(entry, paged_attn_cfg)| MultiModelEntry {
                builder: entry.builder.with_paged_attn_cfg(paged_attn_cfg),
                alias: entry.alias,
            })
            .collect::<Vec<_>>();

        // Build the first model to create the initial InferenceRs instance
        let mut builders_iter = entries.into_iter();
        let first_entry = builders_iter.next().unwrap();

        let (pipeline, scheduler_config, add_model_config) =
            first_entry.builder.build_pipeline().await?;
        let pipeline_name = pipeline.lock().await.name();
        let primary_id = first_entry
            .alias
            .clone()
            .unwrap_or_else(|| pipeline_name.clone());

        let mut runner_builder = inference_core::InferenceRsBuilder::from_config(
            pipeline,
            scheduler_config,
            add_model_config,
        )
        .with_deferred_daemon_start(true);
        if primary_id != pipeline_name {
            runner_builder = runner_builder.with_model_id(primary_id.clone());
        }

        let inference = runner_builder.build().await;

        if let Some(alias) = first_entry.alias {
            if alias != pipeline_name {
                inference
                    .register_model_alias(pipeline_name.clone(), &primary_id)
                    .map_err(|e| anyhow::anyhow!(e))?;
            }
        }

        // Add remaining models using their pipeline names as IDs (or aliases when provided)
        for entry in builders_iter {
            let (pipeline, scheduler_config, add_model_config) =
                entry.builder.build_pipeline().await?;
            let pipeline_name = pipeline.lock().await.name();
            let primary_id = entry.alias.clone().unwrap_or_else(|| pipeline_name.clone());
            inference
                .add_model(
                    primary_id.clone(),
                    pipeline,
                    scheduler_config,
                    add_model_config,
                )
                .await
                .map_err(|e| anyhow::anyhow!(e))?;

            if let Some(alias) = entry.alias {
                if alias != pipeline_name {
                    inference
                        .register_model_alias(pipeline_name.clone(), &primary_id)
                        .map_err(|e| anyhow::anyhow!(e))?;
                }
            }
        }

        // Set the default model if specified
        if let Some(default_id) = self.default_model_id {
            inference
                .set_default_model_id(&default_id)
                .map_err(|e| anyhow::anyhow!(e))?;
        }
        // Otherwise, the first model is already the default (set by InferenceRs::new)

        if inference_core::distributed::is_daemon() {
            inference.run_daemon_replicator_forever();
        }

        Ok(Model::new(inference))
    }
}

// Pipeline building functions for each model type.
// These are public so individual builders can reuse them to avoid code duplication.

pub(crate) fn maybe_initialize_logging(with_logging: bool) {
    if with_logging {
        inference_core::initialize_logging();
    }
}

pub(crate) fn resolve_device(force_cpu: bool, device: Option<Device>) -> anyhow::Result<Device> {
    Ok(device
        .map(Ok)
        .unwrap_or_else(|| crate::best_device(force_cpu))?)
}

pub(crate) fn resolve_isq_type(
    isq: Option<&crate::IsqSetting>,
    device: &Device,
) -> anyhow::Result<Option<IsqType>> {
    isq.map(|setting| crate::resolve_isq(setting, device))
        .transpose()
}

fn paged_attn_with_serving_capacity(
    config: Option<PagedAttentionConfig>,
    max_num_seqs: usize,
    recurrent_prefix_capacity: usize,
) -> anyhow::Result<Option<PagedAttentionConfig>> {
    Ok(config
        .map(|config| config.with_serving_capacity(max_num_seqs))
        .transpose()?
        .map(|config| config.with_recurrent_prefix_capacity(recurrent_prefix_capacity)))
}

pub(crate) fn build_engine_config(
    throughput_logging_enabled: bool,
    search_embedding_model: Option<SearchEmbeddingModel>,
    search_callback: Option<Arc<SearchCallback>>,
    tool_callbacks: &HashMap<String, ToolCallbackWithTool>,
    no_kv_cache: bool,
    prefix_cache_n: Option<usize>,
) -> EngineConfig {
    EngineConfig {
        throughput_logging_enabled,
        search_embedding_model,
        search_callback,
        tool_callbacks: tool_callbacks.clone(),
        no_kv_cache,
        no_prefix_cache: prefix_cache_n.is_none(),
        prefix_cache_n: prefix_cache_n.unwrap_or(16),
        disable_eos_stop: false,
    }
}

pub(crate) fn join_path_list(paths: Option<&[PathBuf]>, delimiter: &str) -> Option<String> {
    paths.map(|paths| {
        paths
            .iter()
            .map(|path| path.to_string_lossy())
            .collect::<Vec<_>>()
            .join(delimiter)
    })
}

/// Create a Model from pipeline components.
/// This is the common code path used by all individual builder `build()` methods.
pub async fn build_model_from_pipeline(
    pipeline: Arc<Mutex<dyn inference_core::Pipeline>>,
    scheduler_config: SchedulerConfig,
    add_model_config: AddModelConfig,
) -> Model {
    Model::new(
        inference_core::InferenceRsBuilder::from_config(
            pipeline,
            scheduler_config,
            add_model_config,
        )
        .build()
        .await,
    )
}

// The first load and a later reload both go through the stored config, so they build the same pipeline.
async fn load_from_config(
    loader_config: &inference_core::ModelLoaderConfig,
    no_kv_cache: bool,
    mtp_runtime: inference_core::MtpRuntimeConfig,
) -> anyhow::Result<Arc<Mutex<dyn Pipeline>>> {
    let loader = loader_config.build_loader(no_kv_cache)?;
    Ok(loader_config.load(&*loader, mtp_runtime).await?)
}

/// Build a text model pipeline from a TextModelBuilder.
/// Returns the pipeline, scheduler config, and AddModelConfig needed for Model creation.
pub async fn build_text_pipeline(
    builder: crate::TextModelBuilder,
) -> anyhow::Result<(Arc<Mutex<dyn Pipeline>>, SchedulerConfig, AddModelConfig)> {
    let model_selected = plain_text_selection(&builder);
    build_text_pipeline_as(builder, model_selected, Default::default()).await
}

pub(crate) fn plain_text_selection(
    builder: &crate::TextModelBuilder,
) -> inference_core::ModelSelected {
    use inference_core::*;
    ModelSelected::Plain {
        model_id: builder.model_id.clone(),
        tokenizer_json: builder.tokenizer_json.clone(),
        arch: builder.loader_type.clone(),
        dtype: builder.dtype,
        topology: builder.topology_path.clone(),
        organization: Some(builder.organization),
        write_uqff: builder.write_uqff.clone(),
        from_uqff: join_path_list(builder.from_uqff.as_deref(), UQFF_MULTI_FILE_DELIMITER),
        imatrix: builder.imatrix.clone(),
        calibration_file: builder.calibration_file.clone(),
        max_seq_len: AutoDeviceMapParams::DEFAULT_MAX_SEQ_LEN,
        max_batch_size: AutoDeviceMapParams::DEFAULT_MAX_BATCH_SIZE,
        hf_cache_path: builder.hf_cache_path.clone(),
        matformer_config_path: builder.matformer_config_path.clone(),
        matformer_slice_name: builder.matformer_slice_name.clone(),
    }
}

// Loads `model_selected` with the text builder's runtime options; adapter builders pick their own selection.
pub(crate) async fn build_text_pipeline_as(
    mut builder: crate::TextModelBuilder,
    model_selected: inference_core::ModelSelected,
    overrides: inference_core::LoadOverrides,
) -> anyhow::Result<(Arc<Mutex<dyn Pipeline>>, SchedulerConfig, AddModelConfig)> {
    use inference_core::*;

    let mtp_runtime = MtpRuntimeConfig::new(builder.prefix_cache_n.unwrap_or(0));
    builder.paged_attn_cfg = paged_attn_with_serving_capacity(
        builder.paged_attn_cfg,
        builder.max_num_seqs,
        builder.prefix_cache_n.unwrap_or(0),
    )?;
    maybe_initialize_logging(builder.with_logging);

    let device = resolve_device(builder.force_cpu, builder.device.clone())?;
    builder.paged_attn_cfg = reserve_external_mtp_memory_with_runtime(
        builder.paged_attn_cfg,
        builder.mtp_config.as_ref(),
        mtp_runtime,
        &builder.dtype,
        &device,
    )?;
    let isq_type = resolve_isq_type(builder.isq.as_ref(), &device)?;
    let device_map_setting = builder
        .device_mapping
        .clone()
        .unwrap_or(DeviceMapSetting::Auto(AutoDeviceMapParams::default_text()));

    let loader_config = ModelLoaderConfig {
        model_selected,
        token_source: builder.token_source.clone(),
        hf_revision: builder.hf_revision.clone(),
        dtype: builder.dtype,
        device,
        device_map_setting,
        isq: isq_type,
        paged_attn_config: builder.paged_attn_cfg,
        silent: !builder.with_logging,
        chat_template: builder.chat_template.clone(),
        jinja_explicit: builder.jinja_explicit.clone(),
        max_model_len: builder.max_model_len,
        hf_config_overrides: builder.hf_config_overrides.clone(),
        mtp_config: builder.mtp_config.clone(),
        encoder_cache_memory_bytes: None,
        overrides: LoadOverrides {
            topology: builder.topology.clone(),
            ..overrides
        },
    };
    let pipeline = load_from_config(&loader_config, builder.no_kv_cache, mtp_runtime).await?;

    let scheduler_config = SchedulerConfig::for_pipeline(
        &pipeline,
        builder.paged_attn_cfg.is_some(),
        builder.max_num_seqs,
        SchedulerLimits::default(),
    )
    .await?;
    let engine_config = build_engine_config(
        builder.throughput_logging,
        builder.search_embedding_model,
        builder.search_callback.clone(),
        &builder.tool_callbacks,
        builder.no_kv_cache,
        builder.prefix_cache_n,
    );
    let add_model_config = AddModelConfig {
        engine_config,
        mcp_client_config: builder.mcp_client_config.clone(),
        loader_config: Some(loader_config),
        code_exec_config: builder.code_exec_config.clone(),
        shell_config: builder.shell_config.clone(),
    };

    Ok((pipeline, scheduler_config, add_model_config))
}

/// Build a multimodal model pipeline from a MultimodalModelBuilder.
/// Returns the pipeline, scheduler config, and AddModelConfig needed for Model creation.
pub async fn build_multimodal_pipeline(
    mut builder: crate::MultimodalModelBuilder,
) -> anyhow::Result<(Arc<Mutex<dyn Pipeline>>, SchedulerConfig, AddModelConfig)> {
    use inference_core::*;

    let mtp_runtime = MtpRuntimeConfig::new(builder.prefix_cache_n.unwrap_or(0));
    builder.paged_attn_cfg = paged_attn_with_serving_capacity(
        builder.paged_attn_cfg,
        builder.max_num_seqs,
        builder.prefix_cache_n.unwrap_or(0),
    )?;
    maybe_initialize_logging(builder.with_logging);

    let device = resolve_device(builder.force_cpu, builder.device.clone())?;
    builder.paged_attn_cfg = reserve_external_mtp_memory_with_runtime(
        builder.paged_attn_cfg,
        builder.mtp_config.as_ref(),
        mtp_runtime,
        &builder.dtype,
        &device,
    )?;
    let isq_type = resolve_isq_type(builder.isq.as_ref(), &device)?;
    let device_map_setting = builder
        .device_mapping
        .clone()
        .unwrap_or(DeviceMapSetting::Auto(
            AutoDeviceMapParams::default_multimodal(),
        ));

    let loader_config = ModelLoaderConfig {
        model_selected: ModelSelected::MultimodalPlain {
            model_id: builder.model_id.clone(),
            tokenizer_json: builder.tokenizer_json.clone(),
            arch: builder.loader_type,
            dtype: builder.dtype,
            topology: builder.topology_path.clone(),
            write_uqff: builder.write_uqff.clone(),
            from_uqff: join_path_list(builder.from_uqff.as_deref(), UQFF_MULTI_FILE_DELIMITER),
            max_edge: builder.max_edge,
            calibration_file: builder.calibration_file.clone(),
            imatrix: builder.imatrix.clone(),
            max_seq_len: AutoDeviceMapParams::DEFAULT_MAX_SEQ_LEN,
            max_batch_size: AutoDeviceMapParams::DEFAULT_MAX_BATCH_SIZE,
            max_num_images: AutoDeviceMapParams::DEFAULT_MAX_NUM_IMAGES,
            max_image_length: AutoDeviceMapParams::DEFAULT_MAX_IMAGE_LENGTH,
            hf_cache_path: builder.hf_cache_path.clone(),
            matformer_config_path: builder.matformer_config_path.clone(),
            matformer_slice_name: builder.matformer_slice_name.clone(),
            organization: Some(builder.organization),
        },
        token_source: builder.token_source.clone(),
        hf_revision: builder.hf_revision.clone(),
        dtype: builder.dtype,
        device,
        device_map_setting,
        isq: isq_type,
        paged_attn_config: builder.paged_attn_cfg,
        silent: !builder.with_logging,
        chat_template: builder.chat_template.clone(),
        jinja_explicit: builder.jinja_explicit.clone(),
        max_model_len: builder.max_model_len,
        hf_config_overrides: builder.hf_config_overrides.clone(),
        mtp_config: builder.mtp_config.clone(),
        encoder_cache_memory_bytes: builder.encoder_cache_memory_bytes,
        overrides: LoadOverrides {
            topology: builder.topology.clone(),
            ..Default::default()
        },
    };
    let pipeline = load_from_config(&loader_config, false, mtp_runtime).await?;

    let scheduler_config = SchedulerConfig::for_pipeline(
        &pipeline,
        builder.paged_attn_cfg.is_some(),
        builder.max_num_seqs,
        SchedulerLimits::default(),
    )
    .await?;
    let engine_config = build_engine_config(
        builder.throughput_logging,
        builder.search_embedding_model,
        builder.search_callback.clone(),
        &builder.tool_callbacks,
        false,
        builder.prefix_cache_n,
    );
    let add_model_config = AddModelConfig {
        engine_config,
        mcp_client_config: None,
        loader_config: Some(loader_config),
        code_exec_config: None,
        shell_config: builder.shell_config.clone(),
    };

    Ok((pipeline, scheduler_config, add_model_config))
}

/// Build a GGUF model pipeline from a GgufModelBuilder.
/// Returns the pipeline, scheduler config, and AddModelConfig needed for Model creation.
pub async fn build_gguf_pipeline(
    builder: crate::GgufModelBuilder,
) -> anyhow::Result<(Arc<Mutex<dyn Pipeline>>, SchedulerConfig, AddModelConfig)> {
    build_gguf_pipeline_as(builder, gguf_selection, Default::default()).await
}

/// Device-map sizes a GGUF selection records, derived from the builder's device mapping.
pub(crate) struct GgufAutoMapDims {
    pub(crate) max_seq_len: usize,
    pub(crate) max_batch_size: usize,
    pub(crate) max_num_images: Option<usize>,
    pub(crate) max_image_length: Option<usize>,
}

pub(crate) fn gguf_selection(
    builder: &crate::GgufModelBuilder,
    dims: GgufAutoMapDims,
) -> inference_core::ModelSelected {
    use inference_core::*;
    ModelSelected::GGUF {
        tok_model_id: builder.tok_model_id.clone(),
        quantized_model_id: builder.model_id.clone(),
        quantized_filename: builder.files.join(GGUF_MULTI_FILE_DELIMITER),
        tokenizer_json: builder.tokenizer_json.clone(),
        mmproj_filename: builder
            .mmproj_files
            .as_ref()
            .map(|files| files.join(GGUF_MULTI_FILE_DELIMITER)),
        lora_adapters: builder.lora_adapters.clone().unwrap_or_default(),
        lora_runtime_config: builder
            .lora_adapters
            .as_ref()
            .map(|_| builder.lora_runtime_config),
        dtype: builder.dtype,
        topology: builder.topology_path.clone(),
        organization: Some(builder.organization),
        write_uqff: builder.write_uqff.clone(),
        imatrix: builder.imatrix.clone(),
        calibration_file: builder.calibration_file.clone(),
        max_edge: builder.max_edge,
        max_seq_len: dims.max_seq_len,
        max_batch_size: dims.max_batch_size,
        max_num_images: dims.max_num_images,
        max_image_length: dims.max_image_length,
        hf_cache_path: builder.hf_cache_path.clone(),
        matformer_config_path: builder.matformer_config_path.clone(),
        matformer_slice_name: builder.matformer_slice_name.clone(),
    }
}

// Loads the GGUF selection `select` builds with the GGUF builder's runtime options.
pub(crate) async fn build_gguf_pipeline_as(
    mut builder: crate::GgufModelBuilder,
    select: impl FnOnce(&crate::GgufModelBuilder, GgufAutoMapDims) -> inference_core::ModelSelected,
    overrides: inference_core::LoadOverrides,
) -> anyhow::Result<(Arc<Mutex<dyn Pipeline>>, SchedulerConfig, AddModelConfig)> {
    use inference_core::*;

    let mtp_runtime = MtpRuntimeConfig::new(builder.prefix_cache_n.unwrap_or(0));
    builder.paged_attn_cfg = paged_attn_with_serving_capacity(
        builder.paged_attn_cfg,
        builder.max_num_seqs,
        builder.prefix_cache_n.unwrap_or(0),
    )?;
    maybe_initialize_logging(builder.with_logging);

    let device = resolve_device(builder.force_cpu, builder.device.clone())?;
    builder.paged_attn_cfg = reserve_external_mtp_memory_with_runtime(
        builder.paged_attn_cfg,
        builder.mtp_config.as_ref(),
        mtp_runtime,
        &builder.dtype,
        &device,
    )?;
    let default_device_map = if builder.mmproj_files.is_some() {
        AutoDeviceMapParams::default_multimodal()
    } else {
        AutoDeviceMapParams::default_text()
    };
    let device_map_setting = builder
        .device_mapping
        .clone()
        .unwrap_or(DeviceMapSetting::Auto(default_device_map));
    let isq_type = resolve_isq_type(builder.isq.as_ref(), &device)?;

    let (max_seq_len, max_batch_size, max_num_images, max_image_length) = match &device_map_setting
    {
        DeviceMapSetting::Auto(AutoDeviceMapParams::Text {
            max_seq_len,
            max_batch_size,
        }) => (*max_seq_len, *max_batch_size, None, None),
        DeviceMapSetting::Auto(AutoDeviceMapParams::Multimodal {
            max_seq_len,
            max_batch_size,
            max_image_shape,
            max_num_images,
        }) => (
            *max_seq_len,
            *max_batch_size,
            Some(*max_num_images),
            Some(max_image_shape.0.max(max_image_shape.1)),
        ),
        _ => (
            AutoDeviceMapParams::DEFAULT_MAX_SEQ_LEN,
            AutoDeviceMapParams::DEFAULT_MAX_BATCH_SIZE,
            builder
                .mmproj_files
                .as_ref()
                .map(|_| AutoDeviceMapParams::DEFAULT_MAX_NUM_IMAGES),
            builder
                .mmproj_files
                .as_ref()
                .map(|_| AutoDeviceMapParams::DEFAULT_MAX_IMAGE_LENGTH),
        ),
    };

    let loader_config = ModelLoaderConfig {
        model_selected: select(
            &builder,
            GgufAutoMapDims {
                max_seq_len,
                max_batch_size,
                max_num_images,
                max_image_length,
            },
        ),
        token_source: builder.token_source.clone(),
        hf_revision: builder.hf_revision.clone(),
        dtype: builder.dtype,
        device,
        device_map_setting,
        isq: isq_type,
        paged_attn_config: builder.paged_attn_cfg,
        silent: !builder.with_logging,
        chat_template: builder.chat_template.clone(),
        jinja_explicit: builder.jinja_explicit.clone(),
        max_model_len: builder.max_model_len,
        hf_config_overrides: None,
        mtp_config: builder.mtp_config.clone(),
        encoder_cache_memory_bytes: builder.encoder_cache_memory_bytes,
        overrides: LoadOverrides {
            topology: builder.topology.clone(),
            ..overrides
        },
    };
    let pipeline = load_from_config(&loader_config, builder.no_kv_cache, mtp_runtime).await?;

    let scheduler_config = SchedulerConfig::for_pipeline(
        &pipeline,
        builder.paged_attn_cfg.is_some(),
        builder.max_num_seqs,
        SchedulerLimits::default(),
    )
    .await?;
    let engine_config = build_engine_config(
        builder.throughput_logging,
        builder.search_embedding_model,
        builder.search_callback.clone(),
        &builder.tool_callbacks,
        builder.no_kv_cache,
        builder.prefix_cache_n,
    );
    let add_model_config = AddModelConfig {
        engine_config,
        mcp_client_config: builder.mcp_client_config.clone(),
        loader_config: Some(loader_config),
        code_exec_config: builder.code_exec_config.clone(),
        shell_config: builder.shell_config.clone(),
    };

    Ok((pipeline, scheduler_config, add_model_config))
}

/// Build a diffusion model pipeline from a DiffusionModelBuilder.
/// Returns the pipeline, scheduler config, and AddModelConfig needed for Model creation.
pub async fn build_diffusion_pipeline(
    builder: crate::DiffusionModelBuilder,
) -> anyhow::Result<(Arc<Mutex<dyn Pipeline>>, SchedulerConfig, AddModelConfig)> {
    use inference_core::*;

    maybe_initialize_logging(builder.with_logging);
    let loader_config = ModelLoaderConfig {
        model_selected: ModelSelected::DiffusionPlain {
            model_id: builder.model_id.clone(),
            arch: builder.loader_type,
            dtype: builder.dtype,
        },
        token_source: builder.token_source.clone(),
        hf_revision: builder.hf_revision.clone(),
        dtype: builder.dtype,
        device: resolve_device(builder.force_cpu, None)?,
        device_map_setting: DeviceMapSetting::Auto(AutoDeviceMapParams::default_text()),
        isq: None,
        paged_attn_config: None,
        silent: !builder.with_logging,
        chat_template: None,
        jinja_explicit: None,
        max_model_len: None,
        hf_config_overrides: None,
        mtp_config: None,
        encoder_cache_memory_bytes: None,
        overrides: LoadOverrides::default(),
    };
    let pipeline = load_from_config(&loader_config, false, MtpRuntimeConfig::default()).await?;

    let add_model_config = AddModelConfig {
        engine_config: EngineConfig::default(),
        mcp_client_config: None,
        loader_config: Some(loader_config),
        code_exec_config: None,
        shell_config: None,
    };

    Ok((
        pipeline,
        SchedulerConfig::fixed(builder.max_num_seqs)?,
        add_model_config,
    ))
}

/// Build a speech model pipeline from a SpeechModelBuilder.
/// Returns the pipeline, scheduler config, and AddModelConfig needed for Model creation.
pub async fn build_speech_pipeline(
    builder: crate::SpeechModelBuilder,
) -> anyhow::Result<(Arc<Mutex<dyn Pipeline>>, SchedulerConfig, AddModelConfig)> {
    use inference_core::*;

    maybe_initialize_logging(builder.with_logging);
    let loader_config = ModelLoaderConfig {
        model_selected: ModelSelected::Speech {
            model_id: builder.model_id.clone(),
            dac_model_id: builder.dac_model_id.clone(),
            arch: builder.loader_type,
            dtype: builder.dtype,
        },
        token_source: builder.token_source.clone(),
        hf_revision: builder.hf_revision.clone(),
        dtype: builder.dtype,
        device: resolve_device(builder.force_cpu, None)?,
        device_map_setting: DeviceMapSetting::Auto(AutoDeviceMapParams::default_text()),
        isq: None,
        paged_attn_config: None,
        silent: !builder.with_logging,
        chat_template: None,
        jinja_explicit: None,
        max_model_len: None,
        hf_config_overrides: None,
        mtp_config: None,
        encoder_cache_memory_bytes: None,
        overrides: LoadOverrides {
            speech_cfg: builder.cfg,
            ..Default::default()
        },
    };
    let pipeline = load_from_config(&loader_config, false, MtpRuntimeConfig::default()).await?;

    let add_model_config = AddModelConfig {
        engine_config: EngineConfig::default(),
        mcp_client_config: None,
        loader_config: Some(loader_config),
        code_exec_config: None,
        shell_config: None,
    };

    Ok((
        pipeline,
        SchedulerConfig::fixed(builder.max_num_seqs)?,
        add_model_config,
    ))
}

/// Build an embedding model pipeline from an EmbeddingModelBuilder.
/// Returns the pipeline, scheduler config, and AddModelConfig needed for Model creation.
pub async fn build_embedding_pipeline(
    builder: crate::EmbeddingModelBuilder,
) -> anyhow::Result<(Arc<Mutex<dyn Pipeline>>, SchedulerConfig, AddModelConfig)> {
    use inference_core::*;

    maybe_initialize_logging(builder.with_logging);
    let device = resolve_device(builder.force_cpu, builder.device.clone())?;
    let isq_type = resolve_isq_type(builder.isq.as_ref(), &device)?;
    let loader_config = ModelLoaderConfig {
        model_selected: ModelSelected::Embedding {
            model_id: builder.model_id.clone(),
            tokenizer_json: builder.tokenizer_json.clone(),
            arch: builder.loader_type,
            dtype: builder.dtype,
            topology: builder.topology_path.clone(),
            write_uqff: builder.write_uqff.clone(),
            from_uqff: join_path_list(builder.from_uqff.as_deref(), UQFF_MULTI_FILE_DELIMITER),
            imatrix: builder.imatrix.clone(),
            calibration_file: builder.calibration_file.clone(),
            hf_cache_path: builder.hf_cache_path.clone(),
        },
        token_source: builder.token_source.clone(),
        hf_revision: builder.hf_revision.clone(),
        dtype: builder.dtype,
        device,
        device_map_setting: builder
            .device_mapping
            .clone()
            .unwrap_or(DeviceMapSetting::Auto(AutoDeviceMapParams::default_text())),
        isq: isq_type,
        paged_attn_config: None,
        silent: !builder.with_logging,
        chat_template: None,
        jinja_explicit: None,
        max_model_len: None,
        hf_config_overrides: None,
        mtp_config: None,
        encoder_cache_memory_bytes: None,
        overrides: LoadOverrides {
            topology: builder.topology.clone(),
            ..Default::default()
        },
    };
    let pipeline = load_from_config(&loader_config, false, MtpRuntimeConfig::default()).await?;

    let engine_config = EngineConfig {
        throughput_logging_enabled: builder.throughput_logging,
        ..Default::default()
    };
    let add_model_config = AddModelConfig {
        engine_config,
        mcp_client_config: None,
        loader_config: Some(loader_config),
        code_exec_config: None,
        shell_config: None,
    };

    Ok((
        pipeline,
        SchedulerConfig::fixed(builder.max_num_seqs)?,
        add_model_config,
    ))
}

/// Build a model pipeline using auto-detection from a ModelBuilder.
/// This uses `AutoLoaderBuilder` to detect the model type (text, multimodal, embedding, etc.)
/// from the model's config.json, similar to the CLI `run` command.
pub async fn build_auto_pipeline(
    mut builder: crate::ModelBuilder,
) -> anyhow::Result<(Arc<Mutex<dyn Pipeline>>, SchedulerConfig, AddModelConfig)> {
    use inference_core::*;

    let mtp_runtime = MtpRuntimeConfig::new(builder.prefix_cache_n.unwrap_or(0));
    builder.paged_attn_cfg = paged_attn_with_serving_capacity(
        builder.paged_attn_cfg,
        builder.max_num_seqs,
        builder.prefix_cache_n.unwrap_or(0),
    )?;
    maybe_initialize_logging(builder.with_logging);

    let device = resolve_device(builder.force_cpu, builder.device.clone())?;
    builder.paged_attn_cfg = reserve_external_mtp_memory_with_runtime(
        builder.paged_attn_cfg,
        builder.mtp_config.as_ref(),
        mtp_runtime,
        &builder.dtype,
        &device,
    )?;
    let isq_type = resolve_isq_type(builder.isq.as_ref(), &device)?;
    let device_map_setting = builder
        .device_mapping
        .clone()
        .unwrap_or(DeviceMapSetting::Auto(AutoDeviceMapParams::default_text()));

    let loader_config = ModelLoaderConfig {
        model_selected: ModelSelected::Run {
            model_id: builder.model_id.clone(),
            quant: None,
            tokenizer_json: builder.tokenizer_json.clone(),
            dtype: builder.dtype,
            topology: builder.topology_path.clone(),
            organization: Some(builder.organization),
            write_uqff: builder.write_uqff.clone(),
            from_uqff: join_path_list(builder.from_uqff.as_deref(), UQFF_MULTI_FILE_DELIMITER),
            imatrix: builder.imatrix.clone(),
            calibration_file: builder.calibration_file.clone(),
            max_edge: builder.max_edge,
            max_seq_len: AutoDeviceMapParams::DEFAULT_MAX_SEQ_LEN,
            max_batch_size: AutoDeviceMapParams::DEFAULT_MAX_BATCH_SIZE,
            max_num_images: None,
            max_image_length: None,
            hf_cache_path: builder.hf_cache_path.clone(),
            matformer_config_path: builder.matformer_config_path.clone(),
            matformer_slice_name: builder.matformer_slice_name.clone(),
        },
        token_source: builder.token_source.clone(),
        hf_revision: builder.hf_revision.clone(),
        dtype: builder.dtype,
        device,
        device_map_setting,
        isq: isq_type,
        paged_attn_config: builder.paged_attn_cfg,
        silent: !builder.with_logging,
        chat_template: builder.chat_template.clone(),
        jinja_explicit: builder.jinja_explicit.clone(),
        max_model_len: builder.max_model_len,
        hf_config_overrides: builder.hf_config_overrides.clone(),
        mtp_config: builder.mtp_config.clone(),
        encoder_cache_memory_bytes: builder.encoder_cache_memory_bytes,
        overrides: LoadOverrides {
            topology: builder.topology.clone(),
            ..Default::default()
        },
    };
    let pipeline = load_from_config(&loader_config, builder.no_kv_cache, mtp_runtime).await?;

    let scheduler_config = SchedulerConfig::for_pipeline(
        &pipeline,
        builder.paged_attn_cfg.is_some(),
        builder.max_num_seqs,
        SchedulerLimits::default(),
    )
    .await?;
    let engine_config = build_engine_config(
        builder.throughput_logging,
        builder.search_embedding_model,
        builder.search_callback.clone(),
        &builder.tool_callbacks,
        builder.no_kv_cache,
        builder.prefix_cache_n,
    );
    let add_model_config = AddModelConfig {
        engine_config,
        mcp_client_config: builder.mcp_client_config.clone(),
        loader_config: Some(loader_config),
        code_exec_config: builder.code_exec_config.clone(),
        shell_config: builder.shell_config.clone(),
    };

    Ok((pipeline, scheduler_config, add_model_config))
}
