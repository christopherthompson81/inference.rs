//! Load steps shared by the normal, multimodal and embedding pipeline loaders.

use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::Result;
use inference_quant::{QuantizedWeightSource, UqffReader};
use inference_tensor::{DType, Device};
use tracing::{info, warn};

use crate::{
    PagedAttentionConfig, Topology, TryIntoDType,
    device_map::{self, AutoDeviceMapParams, DeviceMapSetting, DeviceMapper},
    distributed::{self, TensorParallelism, WorkerTransferData},
    matformer::{MatformerConfig, MatformerSliceConfig},
    paged_attention::ModelConfigLike,
};

pub(crate) struct DeviceMapperInputs<'a> {
    pub setting: &'a DeviceMapSetting,
    pub num_layers: usize,
    pub device: &'a Device,
    pub available_devices: &'a [Device],
    pub topology: Option<&'a Topology>,
    pub write_uqff: bool,
    pub dtype: &'a dyn TryIntoDType,
}

pub(crate) struct MaterializedDeviceMapper {
    pub pipeline_mapper: Box<dyn DeviceMapper + Send + Sync>,
    pub mapper: Box<dyn DeviceMapper + Send + Sync>,
    pub layer_devices: Vec<Option<Device>>,
    pub dtype: DType,
}

/// Builds the pipeline's and the loader's mappers, and turns PagedAttention off if any layer lands on the CPU.
pub(crate) fn materialize_device_mapper(
    inputs: DeviceMapperInputs<'_>,
    paged_attn_config: &mut Option<PagedAttentionConfig>,
) -> Result<MaterializedDeviceMapper> {
    // UQFF writing loads onto the CPU with no topology
    let (mapper_device, mapper_topology) = if inputs.write_uqff {
        (&Device::Cpu, None)
    } else {
        (inputs.device, inputs.topology)
    };
    let build = || {
        inputs.setting.into_mapper(
            inputs.num_layers,
            mapper_device,
            mapper_topology,
            inputs.available_devices,
        )
    };
    let mut pipeline_mapper = build()?;
    let mut mapper = build()?;
    let layer_devices = (0..inputs.num_layers)
        .map(|layer| mapper.device_for(layer, false).cloned())
        .collect();
    let dtype = super::isq_flow::resolve_weight_load_dtype(
        inputs.dtype,
        mapper.as_ref(),
        inputs.available_devices,
        inputs.write_uqff,
    )?;
    mapper.set_model_dtype(dtype);
    pipeline_mapper.set_model_dtype(dtype);

    // get_device_layers already keeps PagedAttention off the CPU; a manual map can still mix devices
    if paged_attn_config.is_some() && mapper.get_unique_devices().iter().any(Device::is_cpu) {
        warn!(
            "Device mapping contains a mix of GPU and CPU. There is no CPU support for PagedAttention, disabling PagedAttention."
        );
        *paged_attn_config = None;
    }
    Ok(MaterializedDeviceMapper {
        pipeline_mapper,
        mapper,
        layer_devices,
        dtype,
    })
}

/// The prepared or on-disk config.json, sanitized for UQFF sources, with HF overrides and the MTP flag applied.
pub(crate) fn prepare_model_config(
    prepared: Option<&PreparedSource>,
    config_filename: &Path,
    from_uqff: bool,
    overrides: Option<&super::HfConfigOverrides>,
    mtp: bool,
) -> Result<String> {
    let config = match prepared.map(|source| source.config.as_str()) {
        Some(config) => config.to_string(),
        None => super::loaders::load_model_config(config_filename, !from_uqff)?,
    };
    let config = if from_uqff {
        super::isq::sanitize_quantized_weight_source_config(&config)?
    } else {
        config
    };
    let config = match overrides {
        Some(overrides) => overrides.apply(&config)?,
        None => config,
    };
    // after the overrides, so one of `rope_parameters` is what the older fields take
    let config = super::loaders::lift_rope_parameters(&config)?;
    if mtp {
        super::loaders::inject_mtp_config_flag(&config)
    } else {
        Ok(config)
    }
}

pub(crate) struct LoadDevices {
    pub tensor_parallelism: TensorParallelism,
    pub device: Device,
    pub available_devices: Vec<Device>,
}

/// Opens the CUDA kernel libraries when any of `devices` is CUDA, so a missing one fails before weights load.
pub(crate) fn ensure_cuda_kernels(devices: &[Device]) -> Result<()> {
    #[cfg(feature = "cuda")]
    if devices.iter().any(Device::is_cuda) {
        crate::kernel_fetch::install();
        inference_nn::cuda::ensure_kernels_loaded().map_err(anyhow::Error::msg)?;
    }
    let _ = devices;
    Ok(())
}

/// A distributed worker's own GPU, GPU 0 under tensor parallelism, or every device like `device`.
pub(crate) fn resolve_load_devices(
    model_config: &dyn ModelConfigLike,
    device: &Device,
    write_uqff: bool,
) -> Result<LoadDevices> {
    let tensor_parallelism = distributed::resolve_tensor_parallelism(
        model_config,
        inference_quant::distributed::use_nccl(),
        write_uqff,
    )?;
    let available_devices = if let Ok(payload) = std::env::var(distributed::IS_DAEMON_FLAG) {
        let payload: WorkerTransferData = serde_json::from_str(&payload)?;
        let WorkerTransferData::Init { worker_rank, .. } = payload;
        vec![Device::new_cuda(worker_rank + 1)?]
    } else if tensor_parallelism.is_enabled() {
        vec![Device::new_cuda(0)?]
    } else {
        device_map::get_all_similar_devices(device)?
    };
    ensure_cuda_kernels(&available_devices)?;
    #[cfg(feature = "cuda")]
    for device in &available_devices {
        if let Device::Cuda(dev) = device {
            unsafe { dev.disable_event_tracking() };
        }
    }
    let device = if tensor_parallelism.is_enabled() {
        available_devices[0].clone()
    } else {
        device.clone()
    };
    Ok(LoadDevices {
        tensor_parallelism,
        device,
        available_devices,
    })
}

pub(crate) struct WeightSources {
    pub uqff_reader: Option<Arc<UqffReader>>,
    pub prepared: Option<Arc<dyn QuantizedWeightSource>>,
    /// The UQFF reader when loading from UQFF, else the prepared source; this is what sizing and loading read.
    pub combined: Option<Arc<dyn QuantizedWeightSource>>,
}

pub(crate) fn open_weight_sources(
    from_uqff: Option<&[PathBuf]>,
    prepared: Option<Arc<dyn QuantizedWeightSource>>,
) -> Result<WeightSources> {
    let uqff_reader = from_uqff.map(UqffReader::open).transpose()?.map(Arc::new);
    let combined = uqff_reader
        .clone()
        .map(|reader| reader as Arc<dyn QuantizedWeightSource>)
        .or(prepared.clone());
    Ok(WeightSources {
        uqff_reader,
        prepared,
        combined,
    })
}

pub(crate) fn load_matformer_slice(
    config_path: Option<&Path>,
    slice_name: Option<&str>,
) -> Result<Option<MatformerSliceConfig>> {
    let Some(config_path) = config_path else {
        if let Some(slice_name) = slice_name {
            anyhow::bail!(
                "MatFormer slice `{slice_name}` needs its config file (`matformer_config_path`, \
                 `--matformer-config-path`)"
            );
        }
        return Ok(None);
    };
    info!("Loading Matformer config from {:?}", config_path);
    let config = Arc::new(MatformerConfig::from_file(config_path)?);
    match slice_name {
        Some(slice_name) => {
            info!("Using Matformer slice: {}", slice_name);
            Ok(Some(MatformerSliceConfig::new(
                slice_name.to_string(),
                config,
            )))
        }
        None => {
            warn!(
                "Matformer config loaded but no slice name specified. Models will use their default slice."
            );
            Ok(None)
        }
    }
}

/// A checkpoint already in memory (a GGUF translated to Hugging Face names), loaded without fetching its files.
pub(crate) struct PreparedSource {
    pub config: String,
    pub weights: inference_quant::ShardedVarBuilder,
    pub tokenizer: tokenizers::Tokenizer,
    pub generation_config: Option<super::chat_template::GenerationConfig>,
    pub chat_template: Option<String>,
    pub bos_token: Option<String>,
    pub eos_token: Option<String>,
    pub unk_token: Option<String>,
    pub processor_config: Option<String>,
    pub preprocessor_config: Option<String>,
    pub source_weight_files: Vec<PathBuf>,
    pub rope_pairing: crate::gguf::normal_registry::RopePairing,
    pub label_decision: Option<Arc<crate::decision::LabelDecision>>,
}

pub(crate) struct HubPathsRequest<'a> {
    pub hf_cache_path: Option<PathBuf>,
    pub model_id: &'a str,
    pub tokenizer_json: Option<&'a str>,
    pub chat_template: Option<&'a str>,
    pub token_source: &'a super::TokenSource,
    pub revision: Option<String>,
    pub silent: bool,
    pub from_uqff: Option<&'a [PathBuf]>,
}

/// Points Hub downloads at `hf_cache_path` (or the default cache); the first load in a process decides.
pub(crate) fn install_hf_cache(hf_cache_path: Option<PathBuf>) {
    let cache = hf_cache_path.map(hf_hub::Cache::new).unwrap_or_default();
    crate::GLOBAL_HF_CACHE.get_or_init(|| cache);
}

/// A Hub model's files from `fetch`, its UQFF shards (if any) stored in `uqff_files`; installs the HF cache first.
pub(crate) fn hub_model_paths<P>(
    request: HubPathsRequest<'_>,
    uqff_files: &std::sync::RwLock<Option<Vec<PathBuf>>>,
    fetch: impl FnOnce(super::paths::PathsRequest<'_>) -> Result<P>,
) -> Result<P> {
    install_hf_cache(request.hf_cache_path);
    let paths_request = super::paths::PathsRequest {
        model_id: request.model_id,
        tokenizer_json: request.tokenizer_json,
        chat_template: request.chat_template,
        token_source: request.token_source,
        revision: request.revision.clone(),
        quantized_model_id: None,
        quantized_filenames: None,
        silent: request.silent,
        loading_uqff: request.from_uqff.is_some(),
    };
    let paths = fetch(paths_request);
    if let Some(from_uqff) = request.from_uqff {
        let files = super::paths::get_uqff_paths(
            from_uqff,
            request.model_id,
            request.token_source,
            request.revision.clone(),
            request.silent,
        )?;
        *uqff_files.write().unwrap() = Some(files);
    }
    paths
}

/// The model's chat template, with special tokens the files leave unset taken from a prepared source.
pub(crate) fn load_chat_template(
    paths: &dyn super::ModelPaths,
    jinja_explicit: Option<&String>,
    chat_template: Option<&String>,
    prepared: Option<&PreparedSource>,
) -> super::ChatTemplate {
    use inference_protocol::chat_template::BeginEndUnkPadTok;
    let chat_template_explicit = paths
        .get_chat_template_explicit()
        .as_ref()
        .map(|x| x.to_string_lossy().to_string());
    let mut template = super::get_chat_template(
        paths,
        jinja_explicit,
        chat_template_explicit.as_ref(),
        chat_template,
        prepared.and_then(|source| source.chat_template.clone()),
    );
    if let Some(source) = prepared {
        let token = |token: &Option<String>| {
            token
                .clone()
                .map(|token| BeginEndUnkPadTok(either::Either::Left(token)))
        };
        if template.bos_token.is_none() {
            template.bos_token = token(&source.bos_token);
        }
        if template.eos_token.is_none() {
            template.eos_token = token(&source.eos_token);
        }
        if template.unk_token.is_none() {
            template.unk_token = token(&source.unk_token);
        }
    }
    template
}

/// The unquantized tensors a UQFF stores beside the quantized layers; MoQE keeps everything but the experts.
pub(crate) fn uqff_residual_tensors(
    organization: super::IsqOrganization,
    model: &dyn inference_nn::model::IsqModel,
) -> Vec<(String, inference_tensor::Tensor)> {
    match organization {
        super::IsqOrganization::Default => model.residual_tensors(),
        super::IsqOrganization::MoeExpertsOnly => model
            .residual_tensors_moe_experts_only()
            .unwrap_or_else(|| model.residual_tensors()),
    }
}

/// The `generation_config.json` a UQFF copies; none when a prepared source had none.
pub(crate) fn uqff_generation_config_file<'a>(
    paths: &'a dyn super::ModelPaths,
    prepared: Option<&PreparedSource>,
) -> Option<&'a PathBuf> {
    match prepared {
        Some(source) if source.generation_config.is_none() => None,
        _ => paths.get_gen_conf_filename(),
    }
}

/// Where a load's weights come from beyond the checkpoint: a UQFF it reads, writes, or reloads after writing.
#[derive(Clone, Copy)]
pub(crate) enum UqffLoad<'a> {
    None,
    From,
    Write(&'a super::isq::UqffWriteConfig),
    Reload(&'a [PathBuf]),
}

impl<'a> UqffLoad<'a> {
    pub(crate) fn new(
        from_uqff: bool,
        write: Option<&'a super::isq::UqffWriteConfig>,
    ) -> Result<Self> {
        Ok(match (from_uqff, write) {
            (true, Some(_)) => anyhow::bail!(
                "Writing UQFF (`write_uqff`) while loading from UQFF (`from_uqff`) is not supported."
            ),
            (true, None) => Self::From,
            (false, Some(write)) => Self::Write(write),
            (false, None) => Self::None,
        })
    }

    pub(crate) fn reads(self) -> bool {
        matches!(self, Self::From | Self::Reload(_))
    }

    pub(crate) fn is_reload(self) -> bool {
        matches!(self, Self::Reload(_))
    }

    pub(crate) fn write(self) -> Option<&'a super::isq::UqffWriteConfig> {
        match self {
            Self::Write(write) => Some(write),
            Self::None | Self::From | Self::Reload(_) => None,
        }
    }

    pub(crate) fn reload_files(self) -> Option<&'a [PathBuf]> {
        match self {
            Self::Reload(files) => Some(files),
            Self::None | Self::From | Self::Write(_) => None,
        }
    }

    /// The write loads the model unmapped onto the host; a GPU load serves it by reloading the written shards.
    pub(crate) fn serves_written(self, device: &Device) -> bool {
        matches!(self, Self::Write(write) if !write.artifact_only && !device.is_cpu())
    }
}

pub(crate) enum LoadOutcome {
    Pipeline(Arc<tokio::sync::Mutex<dyn crate::Pipeline + Send + Sync>>),
    /// The UQFF was written and is to be served from these runtime-type shards.
    Written(Vec<PathBuf>),
}

/// Runs a load and, when it wrote a UQFF to serve on its device, the reload of the written shards.
pub(crate) fn load_serving_written(
    uqff: UqffLoad<'_>,
    load: impl Fn(UqffLoad<'_>) -> Result<LoadOutcome>,
) -> Result<Arc<tokio::sync::Mutex<dyn crate::Pipeline + Send + Sync>>> {
    let files = match load(uqff)? {
        LoadOutcome::Pipeline(pipeline) => return Ok(pipeline),
        LoadOutcome::Written(files) => files,
    };
    match load(UqffLoad::Reload(&files))? {
        LoadOutcome::Pipeline(pipeline) => Ok(pipeline),
        LoadOutcome::Written(_) => unreachable!("a reload writes nothing"),
    }
}

/// The adapter runtime a LoRA loader was built with; None for other kinds.
pub(crate) fn lora_runtime(
    kind: &super::ModelKind,
    runtime: Option<crate::LoraRuntimeConfig>,
) -> Option<crate::LoraRuntimeConfig> {
    match kind {
        super::ModelKind::Lora | super::ModelKind::GgufLora { .. } => {
            Some(runtime.expect("LoRA loaders have a runtime config"))
        }
        _ => None,
    }
}

/// The checkpoint files a pipeline keeps for re-quantizing later; a UQFF load has none.
pub(crate) fn source_weight_files(
    prepared: Option<&PreparedSource>,
    from_uqff: bool,
    weight_files: &[PathBuf],
) -> Vec<PathBuf> {
    match prepared {
        Some(source) => source.source_weight_files.clone(),
        None if from_uqff => Vec::new(),
        None => weight_files.to_vec(),
    }
}

pub(crate) struct MapSettingInputs<'a> {
    pub setting: DeviceMapSetting,
    pub write_uqff: bool,
    pub distributed: bool,
    pub available_devices: &'a [Device],
    pub dtype: &'a dyn TryIntoDType,
    pub sizing: super::isq_flow::AutoDeviceMapSizingInputs<'a>,
}

pub(crate) struct ResolvedMapSetting {
    pub setting: DeviceMapSetting,
    /// The KV tokens an automatic map planned for; `None` for a manual, dummy or tensor-parallel map.
    pub max_kv_tokens: Option<usize>,
    /// Experts per layer an automatic map chose to cache on the GPU, keeping the rest in host memory (fewer than the
    /// requested cache when that does not fit).
    pub planned_moe_expert_cache: Option<usize>,
    /// The prompt chunk an offloaded model's map reserved activations for.
    pub prompt_chunk_tokens: Option<usize>,
}

// Context an automatic map keeps room for before it moves MoE experts to host memory (or less, for a model with a
// shorter window); a longer `max_seq_len` asks for more, and a context-sized paged cache asks for its own size instead
const PLANNED_OFFLOAD_CONTEXT: usize = 32 * 1024;
// The planner tries caching each multiple of 1/this of a layer's experts, most first, and then none (every expert
// on the CPU); fine enough that what a quantized cache frees becomes cached experts
const PLANNED_SLOT_STEPS: usize = 64;
// Longest prompt chunk an offloaded model's planner sizes: a chunk stages about every expert a layer routes to however
// long it is, so prefill speeds up with the chunk well past the default
const MAX_PLANNED_PROMPT_CHUNK: usize = 8192;
// Prompt chunk the planner keeps room for before it caches more experts: one slot step is a few percent of decode,
// while a 512-token chunk reads prompts about 3x slower (qwen4exp investigation Runs 27 and 42)
const MIN_PLANNED_PROMPT_CHUNK: usize = 2048;
// Prefix snapshots a planned offload keeps in the recurrent pool: each holds a sequence's recurrent state (~111 MB on
// Qwen3.8-Flash-Next), VRAM the expert cache would otherwise take
const OFFLOAD_PREFIX_SNAPSHOTS: usize = 2;
// GPU memory a context-sized cache leaves free for what prompts allocate beyond the counted staging and chunk
// activations: the staging's per-projection maxima, a step's transients and allocations outside the pool (about
// 1.15 GB at a 22K-token prompt on Flash-Next, qwen4exp investigation Run 37)
const OFFLOAD_RUNTIME_RESERVE_BYTES: usize = 1 << 30;
// The paged cache an `auto` cache becomes when MoE experts leave the GPU: its room goes to cached experts, and 3-bit
// K and V cost nothing measurable on the hybrid models tried (qwen4exp investigation Run 35)
const OFFLOAD_CACHE_TYPE: crate::PagedCacheType = crate::PagedCacheType::Quant { k: 3, v: 3 };

struct ExpertCachePlan<'a> {
    sizing: super::isq_flow::AutoDeviceMapSizingInputs<'a>,
    devices: &'a [Device],
    dtype: DType,
    paged_attn_config: &'a Option<PagedAttentionConfig>,
}

// A hybrid model's recurrent pool, allocated after the map beside the paged cache, at the slots its reservation takes
fn recurrent_pool_bytes(
    model: &dyn ModelConfigLike,
    paged_attn_config: &PagedAttentionConfig,
) -> usize {
    let slots = crate::kv_cache::recurrent_pool_capacity(
        paged_attn_config.serving_capacity.unwrap_or(1),
        paged_attn_config.recurrent_snapshot_slots,
    );
    slots.saturating_mul(model.recurrent_state_bytes_per_sequence())
}

impl ExpertCachePlan<'_> {
    // Whether the weights with `slots` experts per layer cached (`None`: all resident) fit the primary GPU beside the
    // context `params` asks for, `activations`, the post-load reservations, a hybrid model's recurrent pool and the
    // share a utilization limit keeps free. Computed directly: an automatic map shrinks a paged cache to fit.
    fn fits(
        &self,
        slots: Option<usize>,
        params: &AutoDeviceMapParams,
        activations: usize,
    ) -> Result<bool> {
        Ok(self.spare(slots, params, activations)?.is_some())
    }

    // The bytes left on the primary GPU after what `fits` counts; `None` when that does not fit
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    fn spare(
        &self,
        slots: Option<usize>,
        params: &AutoDeviceMapParams,
        activations: usize,
    ) -> Result<Option<usize>> {
        let (loader, config) = (self.sizing.loader, self.sizing.config);
        let sizes = super::isq_flow::auto_device_map_sizes(
            super::isq_flow::AutoDeviceMapSizingInputs {
                moe_expert_cache: slots,
                ..self.sizing
            },
            self.dtype,
        )?;
        let model = loader.model_config(config)?;
        let primary = &self.devices[0];
        let cap = crate::paged_attention::device_memory_cap(
            crate::MemoryUsage.query(primary)?.available(),
            primary,
        );
        let tokens = params.max_seq_len() * params.max_batch_size();
        let cache = match self.paged_attn_config {
            Some(cfg) => {
                let plan =
                    crate::paged_attention::cache_plan(cfg.cache_type, &cfg.cache_layers, &*model);
                let (num, den) =
                    crate::paged_attention::kv_bytes_per_token(&plan, self.dtype, &*model);
                let held_back = match cfg.mem_gpu {
                    crate::MemoryGpuConfig::Utilization(fraction) => {
                        (cap as f64 * (1.0 - f64::from(fraction)).max(0.0)) as usize
                    }
                    _ => 0,
                };
                (tokens * num).div_ceil(den)
                    + held_back
                    + cfg.base_device_memory_reservation_bytes
                    + recurrent_pool_bytes(&*model, cfg)
                    + cfg.auxiliary_layers.map_or(0, |layers| {
                        tokens * layers.bytes_per_token(self.dtype.size_in_bytes())
                    })
            }
            None => tokens * model.total_kv_cache_elements_per_token() * self.dtype.size_in_bytes(),
        };
        Ok(cap.checked_sub(sizes.total_model_size_in_bytes + activations + cache))
    }

    // Activations as the load reserves them, for the requested map rather than the longer planned context (prompts
    // run in chunks)
    fn activations(&self, params: &AutoDeviceMapParams) -> Result<usize> {
        let (loader, config) = (self.sizing.loader, self.sizing.config);
        Ok(loader
            .mapped_max_act_size_elems(config, params)?
            .max(loader.non_mapped_max_act_size_elems(config, params)?)
            * self.dtype.size_in_bytes())
    }

    // The tokens a paged cache sized by its context asks for, else 0
    fn requested_context(&self) -> usize {
        match self.paged_attn_config.as_ref().map(|cfg| cfg.mem_gpu) {
            Some(crate::MemoryGpuConfig::ContextSize(tokens)) => tokens,
            _ => 0,
        }
    }

    fn planned_context(&self, params: &AutoDeviceMapParams) -> Result<AutoDeviceMapParams> {
        let window = self
            .sizing
            .loader
            .model_config(self.sizing.config)?
            .max_seq_len();
        Ok(params.with_max_seq_len(planned_offload_context(
            params.max_seq_len(),
            self.requested_context(),
            window,
        )))
    }

    // A layer's expert count when its experts can stay in host memory: one GPU and GGUF experts the CPU runs
    fn offloadable_experts(&self) -> Result<Option<usize>> {
        // one GPU: the fit is the primary device's, as a split across GPUs would map layers instead
        let gpus = self.devices.iter().filter(|d| d.is_cuda()).count();
        if gpus != 1
            || !self.devices.first().is_some_and(Device::is_cuda)
            || !matches!(
                self.sizing.sizing,
                super::isq_flow::AutoDeviceMapSizing::PreparedWeightSource
                    | super::isq_flow::AutoDeviceMapSizing::Uqff
            )
        {
            return Ok(None);
        }
        let Some(source) = self.sizing.weight_source else {
            return Ok(None);
        };
        Ok(source.layer_expert_bytes()?.map(|(_, experts)| experts))
    }

    // The activations of a MIN_PLANNED_PROMPT_CHUNK-token prompt chunk, for a loader that reports them
    fn prompt_chunk_floor(&self) -> Result<usize> {
        if self.paged_attn_config.is_none() {
            return Ok(0);
        }
        let (loader, config) = (self.sizing.loader, self.sizing.config);
        Ok(loader
            .prompt_activation_bytes_per_token(config)?
            .map_or(0, |bytes| bytes * MIN_PLANNED_PROMPT_CHUNK))
    }

    // The longest prompt chunk, in multiples of the default chunk up to MAX_PLANNED_PROMPT_CHUNK, whose activations fit
    // beside `slots` cached experts a layer and the planned context, with the bytes it reserves; `None` for the default
    fn prompt_chunk(
        &self,
        slots: usize,
        params: &AutoDeviceMapParams,
    ) -> Result<Option<(usize, usize)>> {
        let (loader, config) = (self.sizing.loader, self.sizing.config);
        let Some(per_token) = loader
            .prompt_activation_bytes_per_token(config)?
            .filter(|&bytes| bytes > 0)
        else {
            return Ok(None);
        };
        if self.paged_attn_config.is_none() || self.offloadable_experts()?.is_none() {
            return Ok(None);
        }
        let context = self.planned_context(params)?;
        let Some(spare) = self.spare(Some(slots), &context, self.activations(params)?)? else {
            return Ok(None);
        };
        let quantum = crate::scheduler::DEFAULT_MAX_PREFILL_CHUNK_TOKENS;
        let chunk = (spare / per_token).min(MAX_PLANNED_PROMPT_CHUNK) / quantum * quantum;
        Ok((chunk > quantum).then_some((chunk, chunk * per_token)))
    }

    // The most experts per layer to cache when a GGUF MoE model's layers do not all fit on the GPU beside a usable
    // context, its routed experts staying in host memory; `None` when it fits resident or cannot offload. A requested
    // cache that does not fit shrinks to one that does: the map would otherwise move whole layers to the CPU.
    fn slots(&self, params: &AutoDeviceMapParams) -> Result<Option<usize>> {
        let Some(experts) = self.offloadable_experts()? else {
            return Ok(None);
        };
        let activations = self.activations(params)?;
        let requested = self.requested_context();
        if let Some(asked) = self.sizing.moe_expert_cache {
            // the context the map and a context-sized paged cache ask for, not the planner's longer default
            let context = params.with_max_seq_len(params.max_seq_len().max(requested));
            if self.fits(Some(asked), &context, activations)? {
                return Ok(None);
            }
            let fitting = shrink_expert_slots(asked, experts, |slots| {
                self.fits(Some(slots), &context, activations)
            })?;
            match fitting {
                Some(slots) => warn!(
                    requested = asked,
                    slots,
                    context = context.max_seq_len(),
                    "the requested MoE expert cache does not fit on the GPU beside the context; caching fewer experts \
                     per layer"
                ),
                None => warn!(
                    requested = asked,
                    "MoE layers do not fit on the GPU even with no experts cached; the map moves layers to the CPU"
                ),
            }
            return Ok(fitting);
        }
        let target = self.planned_context(params)?;
        // an offload keeps room for a usable prompt chunk; fitting resident needs none
        let chunk_floor = self.prompt_chunk_floor()?;
        let plan = plan_expert_slots(experts, [&target, params], |slots, params| {
            let floor = if slots.is_some() { chunk_floor } else { 0 };
            self.fits(slots, params, activations + floor)
        })?;
        if let Some((slots, params)) = plan {
            info!(
                slots,
                experts,
                context = params.max_seq_len(),
                "MoE experts do not fit on the GPU beside the context; keeping them in host memory with this many per \
                 layer cached (`--moe-expert-cache` chooses)"
            );
        }
        Ok(plan.map(|(slots, _)| slots))
    }
}

// The cache an `auto` paged cache becomes for a model whose experts can leave the GPU: the offload type when an expert
// cache was asked for or the weights do not fit resident beside an unquantized cache, else unchanged
fn offload_cache_type(
    cache: crate::PagedCacheType,
    offloadable: bool,
    expert_cache_asked: bool,
    fits_resident: impl FnOnce() -> Result<bool>,
) -> Result<crate::PagedCacheType> {
    if cache != crate::PagedCacheType::Auto || !offloadable {
        return Ok(cache);
    }
    Ok(if expert_cache_asked || !fits_resident()? {
        OFFLOAD_CACHE_TYPE
    } else {
        cache
    })
}

// Whether the paged cache holds just the planned context, so the planner's slots take the rest: when it moves experts
// off the GPU itself and no cache size or share was chosen (a share left at the default would hold back a tenth of
// the GPU, about 20 cached experts on a 24 GB card, beyond what prompts use)
fn sizes_cache_to_context(
    expert_cache_asked: bool,
    fits_resident: bool,
    default_memory: bool,
) -> bool {
    !expert_cache_asked && !fits_resident && default_memory
}

// The context an offload plan keeps room for: a context-sized paged cache's own size (0 when none), else a usable
// default within the model's window, and never less than the map's `max_seq_len`
fn planned_offload_context(max_seq_len: usize, requested: usize, window: usize) -> usize {
    let planned = if requested > 0 {
        requested
    } else {
        PLANNED_OFFLOAD_CONTEXT.min(window)
    };
    max_seq_len.max(planned)
}

// The most experts per layer to cache (with the context it fits), trying each multiple of 1/PLANNED_SLOT_STEPS of a
// layer's experts and then none, for the longer context first; `None` when the weights fit resident beside the first
// context, or nothing fits at all
fn plan_expert_slots<C: Copy>(
    experts: usize,
    contexts: [C; 2],
    mut fits: impl FnMut(Option<usize>, C) -> Result<bool>,
) -> Result<Option<(usize, C)>> {
    if fits(None, contexts[0])? {
        return Ok(None);
    }
    let step = (experts / PLANNED_SLOT_STEPS).max(1);
    let candidates = (1..experts.div_ceil(step))
        .rev()
        .map(|k| k * step)
        .chain([0])
        .collect::<Vec<_>>();
    for context in contexts {
        for &slots in &candidates {
            if fits(Some(slots), context)? {
                return Ok(Some((slots, context)));
            }
        }
    }
    Ok(None)
}

// The most experts per layer, below `requested`, that `fits`: each multiple of 1/PLANNED_SLOT_STEPS of a layer's
// experts and then none; `None` when not even that fits
fn shrink_expert_slots(
    requested: usize,
    experts: usize,
    mut fits: impl FnMut(usize) -> Result<bool>,
) -> Result<Option<usize>> {
    let step = (experts / PLANNED_SLOT_STEPS).max(1);
    for slots in (0..requested.min(experts).div_ceil(step))
        .rev()
        .map(|k| k * step)
    {
        if fits(slots)? {
            return Ok(Some(slots));
        }
    }
    Ok(None)
}

// The default paged cache (no size, share or context chosen) holds one sequence at the model's context (its window, or
// max_model_len) instead of most of the free memory, so several models can share a device; it still takes only what
// is free when that is less.
fn need_sized_default_cache(
    paged_attn_config: &mut Option<PagedAttentionConfig>,
    sizing: &super::isq_flow::AutoDeviceMapSizingInputs<'_>,
    dtype: DType,
) -> Result<()> {
    let Some(cfg) = paged_attn_config.as_mut() else {
        return Ok(());
    };
    if !cfg.default_memory || !matches!(cfg.mem_gpu, crate::MemoryGpuConfig::Utilization(_)) {
        return Ok(());
    }
    let model = sizing.loader.model_config(sizing.config)?;
    let plan = crate::paged_attention::cache_plan(cfg.cache_type, &cfg.cache_layers, &*model);
    let (num, den) = crate::paged_attention::kv_bytes_per_token(&plan, dtype, &*model);
    let bytes = (model.max_seq_len() * num).div_ceil(den);
    let target_mb = bytes.div_ceil(crate::pipeline::BYTES_PER_MIB).max(1);
    cfg.mem_gpu = crate::MemoryGpuConfig::BestEffortMbAmount {
        target_mb,
        min_mb: cfg.need_floor.then_some(target_mb),
    };
    Ok(())
}

/// UQFF writing loads unmapped, tensor parallelism shards instead of mapping, and an automatic map is sized here.
pub(crate) fn resolve_map_setting(
    inputs: MapSettingInputs<'_>,
    paged_attn_config: &mut Option<PagedAttentionConfig>,
) -> Result<ResolvedMapSetting> {
    let MapSettingInputs {
        setting,
        write_uqff,
        distributed,
        available_devices,
        dtype,
        sizing,
    } = inputs;
    if write_uqff {
        return Ok(ResolvedMapSetting {
            setting: DeviceMapSetting::dummy(),
            max_kv_tokens: None,
            planned_moe_expert_cache: None,
            prompt_chunk_tokens: None,
        });
    }
    if distributed {
        return Ok(ResolvedMapSetting {
            setting: DeviceMapSetting::DummyNccl {
                nm_device: available_devices[0].clone(),
            },
            max_kv_tokens: None,
            planned_moe_expert_cache: None,
            prompt_chunk_tokens: None,
        });
    }
    let dtype = dtype.try_into_dtype(&available_devices.iter().collect::<Vec<_>>())?;
    need_sized_default_cache(paged_attn_config, &sizing, dtype)?;
    let DeviceMapSetting::Auto(params) = &setting else {
        return Ok(ResolvedMapSetting {
            setting,
            max_kv_tokens: None,
            planned_moe_expert_cache: None,
            prompt_chunk_tokens: None,
        });
    };
    let max_kv_tokens = Some(params.max_seq_len() * params.max_batch_size());
    if let Some(cache) = paged_attn_config.as_ref().map(|cfg| cfg.cache_type) {
        let probe = ExpertCachePlan {
            sizing,
            devices: available_devices,
            dtype,
            paged_attn_config,
        };
        let offloadable = probe.offloadable_experts()?.is_some();
        let asked = sizing.moe_expert_cache.is_some();
        // the planner moves experts off the GPU itself when the weights do not fit resident beside the planned context
        let planned = probe.planned_context(params)?;
        let fits_resident =
            !offloadable || probe.fits(None, &planned, probe.activations(params)?)?;
        let chosen = offload_cache_type(cache, offloadable, asked, || Ok(fits_resident))?;
        if let Some(cfg) = paged_attn_config.as_mut() {
            if chosen != cache {
                info!(
                    cache = %chosen,
                    "MoE experts leave the GPU; quantizing the KV cache so more of them stay cached \
                     (`--pa-cache-type none` keeps the model's dtype)"
                );
                cfg.cache_type = chosen;
            }
            if sizes_cache_to_context(asked, fits_resident, cfg.default_memory) {
                let tokens = planned.max_seq_len() * planned.max_batch_size();
                info!(
                    tokens,
                    "MoE experts leave the GPU; the KV cache holds the planned context and the rest caches experts \
                     (`--pa-context-len` or `--pa-memory-fraction` chooses)"
                );
                cfg.mem_gpu = crate::MemoryGpuConfig::ContextSize(tokens);
                *cfg = cfg.with_base_device_memory_reservation(OFFLOAD_RUNTIME_RESERVE_BYTES)?;
                let snapshots = cfg
                    .recurrent_snapshot_slots
                    .map_or(OFFLOAD_PREFIX_SNAPSHOTS, |slots| {
                        slots.min(OFFLOAD_PREFIX_SNAPSHOTS)
                    });
                cfg.recurrent_snapshot_slots = Some(snapshots);
            }
        }
    }
    let plan = ExpertCachePlan {
        sizing,
        devices: available_devices,
        dtype,
        paged_attn_config,
    };
    let planned_moe_expert_cache = plan.slots(params)?;
    let prompt_chunk = match planned_moe_expert_cache.or(sizing.moe_expert_cache) {
        Some(slots) => plan.prompt_chunk(slots, params)?,
        None => None,
    };
    if let Some((chunk, bytes)) = prompt_chunk
        && let Some(cfg) = paged_attn_config.take()
    {
        info!(
            chunk,
            "prompt chunks sized to the memory the expert cache leaves (`--max-prefill-chunk-tokens` chooses)"
        );
        *paged_attn_config = Some(cfg.with_base_device_memory_reservation(bytes)?);
    }
    let sizing = super::isq_flow::AutoDeviceMapSizingInputs {
        moe_expert_cache: planned_moe_expert_cache.or(sizing.moe_expert_cache),
        ..sizing
    };
    let (loader, config) = (sizing.loader, sizing.config);
    let super::isq_flow::AutoDeviceMapSizes {
        layer_sizes_in_bytes,
        non_mapped_size_in_bytes,
        total_model_size_in_bytes,
        ..
    } = super::isq_flow::auto_device_map_sizes(sizing, dtype)?;
    let map = super::loaders::auto_device_map::get_device_layers(
        loader,
        config,
        loader.num_layers(config)?,
        layer_sizes_in_bytes,
        non_mapped_size_in_bytes,
        total_model_size_in_bytes,
        available_devices,
        dtype,
        params,
        paged_attn_config.as_mut(),
    )?;
    Ok(ResolvedMapSetting {
        setting: DeviceMapSetting::Map(map),
        max_kv_tokens,
        planned_moe_expert_cache,
        prompt_chunk_tokens: prompt_chunk.map(|(chunk, _)| chunk),
    })
}

/// The load-time metadata every model constructor receives, less its mapper and rope pairing.
pub(crate) struct LoadMetadataParts {
    pub loading_isq: bool,
    pub device: Device,
    pub multi_progress: Arc<indicatif::MultiProgress>,
    pub matformer: Option<MatformerSliceConfig>,
}

impl LoadMetadataParts {
    pub fn metadata(
        &self,
        mapper: Box<dyn DeviceMapper + Send + Sync>,
        rope_pairing: Option<crate::model::RopePairing>,
    ) -> crate::pipeline::NormalLoadingMetadata {
        crate::pipeline::NormalLoadingMetadata {
            mapper,
            loading_isq: self.loading_isq,
            real_device: self.device.clone(),
            multi_progress: self.multi_progress.clone(),
            matformer_slicing_config: self.matformer.clone(),
            rope_pairing,
        }
    }
}

type DeviceForTensor = Arc<
    dyn Fn(String) -> crate::utils::varbuilder_utils::DeviceForLoadTensor + Send + Sync + 'static,
>;

/// The files a model's weights load from and where they land; every branch of a pipeline's load shares them.
pub(crate) struct WeightFiles<'a> {
    pub paths: &'a dyn crate::ModelPaths,
    pub dtype: DType,
    pub device: &'a Device,
    pub layer_devices: Vec<Option<Device>>,
    pub silent: bool,
    pub uqff_reader: Option<Arc<UqffReader>>,
}

impl WeightFiles<'_> {
    /// The model's weights; the `placeholders` layers get dummy weights, for a UQFF load to fill.
    pub fn load(
        &self,
        placeholders: Option<Vec<regex::Regex>>,
        device_for_tensor: DeviceForTensor,
    ) -> Result<inference_quant::ShardedVarBuilder> {
        let vb = crate::utils::varbuilder_utils::from_mmaped_safetensors(
            self.paths.get_weight_filenames().to_vec(),
            Some(self.dtype),
            self.device,
            self.layer_devices.clone(),
            self.silent,
            placeholders.map(Arc::new),
            |_| true,
            device_for_tensor,
        )?;
        Ok(with_uqff(vb, self.uqff_reader.clone()))
    }
}

/// `vb` reading quantized layers from `uqff_reader`, when there is one.
pub(crate) fn with_uqff(
    vb: inference_quant::ShardedVarBuilder,
    uqff_reader: Option<Arc<UqffReader>>,
) -> inference_quant::ShardedVarBuilder {
    match uqff_reader {
        Some(reader) => vb.with_uqff_reader(reader),
        None => vb,
    }
}

/// The ISQ layers a UQFF load fills (MoQE's expert layers with `moqe`); `None` unless loading both.
pub(crate) fn uqff_placeholders(
    loader: &dyn super::isq::IsqModelLoader,
    config: &str,
    loading_isq: bool,
    from_uqff: bool,
    moqe: bool,
) -> Result<Option<Vec<regex::Regex>>> {
    if !(loading_isq && from_uqff) {
        return Ok(None);
    }
    let layers = if moqe {
        loader.isq_layer_regexes_moqe(config)?
    } else {
        loader.isq_layer_regexes(config)?
    };
    Ok(Some(layers))
}

/// A model's generation config: a prepared source's, else its `generation_config.json`, else what its config implies.
pub(crate) fn generation_config(
    prepared: Option<Option<crate::pipeline::chat_template::GenerationConfig>>,
    paths: &dyn crate::ModelPaths,
    config: &str,
) -> Option<crate::pipeline::chat_template::GenerationConfig> {
    use crate::pipeline::chat_template::GenerationConfig;
    let from_files = || {
        let path = paths.get_gen_conf_filename()?;
        // a malformed file loses only its sampling defaults, not the model
        let parsed = std::fs::read_to_string(path)
            .map_err(anyhow::Error::from)
            .and_then(|text| Ok(serde_json::from_str::<GenerationConfig>(&text)?));
        parsed
            .inspect_err(|error| warn!("Failed to read generation_config.json: {error}"))
            .ok()
    };
    match prepared {
        Some(prepared) => prepared,
        None => from_files(),
    }
    .or_else(|| GenerationConfig::from_model_config(config))
}

/// The load settings each pipeline's own config carries.
pub(crate) struct LoadSettings<'a> {
    pub topology: Option<&'a Topology>,
    pub organization: super::IsqOrganization,
    pub write_uqff: Option<&'a super::isq::UqffWriteConfig>,
    pub from_uqff: bool,
    pub has_imatrix: bool,
    pub has_calibration: bool,
}

pub(crate) type AdjustAutoParams<'a> =
    dyn Fn(&device_map::AutoDeviceMapParams) -> Result<device_map::AutoDeviceMapParams> + 'a;

/// What [`open_load_session`] needs from a pipeline loader; the fields name where the pipelines differ.
pub(crate) struct LoadSessionInputs<'a> {
    pub mapped: &'a dyn super::loaders::DeviceMappedModelLoader,
    pub isq: &'a dyn super::IsqModelLoader,
    pub config: &'a str,
    pub settings: LoadSettings<'a>,
    pub paths: &'a dyn super::ModelPaths,
    pub device: &'a Device,
    pub dtype: &'a dyn TryIntoDType,
    pub mapper: DeviceMapSetting,
    pub in_situ_quant: Option<inference_quant::IsqType>,
    pub uqff_files: Option<&'a [PathBuf]>,
    pub prepared: Option<&'a PreparedSource>,
    pub has_lora: bool,
    pub matformer: Option<MatformerSliceConfig>,
    // whether the auto device map sizes layers with the matformer slice applied
    pub matformer_sizing: bool,
    pub non_mapped_unpacked: bool,
    pub auto_device_map_params: Option<&'a AdjustAutoParams<'a>>,
    pub weight_target: &'static str,
    pub embeddings_on_host: bool,
    pub moe_expert_cache: Option<usize>,
}

/// Devices, weight sources, device mappers and the installed ISQ plan for one load.
pub(crate) struct LoadSession {
    pub tensor_parallelism: TensorParallelism,
    pub device: Device,
    pub available_devices: Vec<Device>,
    pub uqff_reader: Option<Arc<UqffReader>>,
    pub weight_source: Option<Arc<dyn QuantizedWeightSource>>,
    pub max_kv_tokens: Option<usize>,
    pub prompt_chunk_tokens: Option<usize>,
    pub pipeline_mapper: Box<dyn DeviceMapper + Send + Sync>,
    pub layer_devices: Vec<Option<Device>>,
    pub dtype: DType,
    pub plan: super::isq_flow::IsqLoadPlan,
    pub attention: inference_nn::paged_attention::AttentionImplementation,
    pub load_parts: LoadMetadataParts,
    pub embeddings_on_host: bool,
    pub moe_expert_cache: Option<usize>,
}

// Requantizes a built-in MTP head's experts for the GPU, where they stay while the model's are offloaded
fn mtp_head_expert_overrides() -> Result<Vec<inference_quant::ImmediateIsqOverride>> {
    super::isq_flow::MTP_HEAD_EXPERTS
        .iter()
        .map(|(name, _, ty)| {
            Ok(inference_quant::ImmediateIsqOverride {
                predicate: Some(regex::Regex::new(&format!("^{}$", regex::escape(name)))?),
                layer_range: None,
                ty: Some(*ty),
                device: None,
            })
        })
        .collect()
}

/// Whether the config declares its head untied from the token embedding (on the model or its text model), so the
/// embedding can leave the device without the head reading it over the bus.
fn config_unties_embeddings(config: &str) -> bool {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(config) else {
        return false;
    };
    let untied = |v: &serde_json::Value| {
        v.get("tie_word_embeddings") == Some(&serde_json::Value::Bool(false))
    };
    untied(&value) || value.get("text_config").is_some_and(untied)
}

/// Opens the load; the mapper it returns builds the model unless `load_model` shards it for tensor parallelism.
pub(crate) fn open_load_session(
    inputs: LoadSessionInputs<'_>,
    paged_attn_config: &mut Option<PagedAttentionConfig>,
) -> Result<(LoadSession, Box<dyn DeviceMapper + Send + Sync>)> {
    let LoadSessionInputs {
        mapped,
        isq,
        config,
        settings,
        paths,
        device,
        dtype,
        mut mapper,
        in_situ_quant,
        uqff_files,
        prepared,
        has_lora,
        matformer,
        matformer_sizing,
        non_mapped_unpacked,
        auto_device_map_params,
        weight_target,
        embeddings_on_host,
        moe_expert_cache,
    } = inputs;
    let prepared_weight_source =
        prepared.and_then(|source| source.weights.weight_source().cloned());
    let write_uqff = settings.write_uqff.is_some();
    let LoadDevices {
        tensor_parallelism,
        device,
        available_devices,
    } = resolve_load_devices(mapped.model_config(config)?.as_ref(), device, write_uqff)?;
    let distributed = tensor_parallelism.is_enabled();
    let embeddings_on_host = embeddings_on_host
        && device.is_cuda()
        && !write_uqff
        && !distributed
        && in_situ_quant.is_none()
        && config_unties_embeddings(config);
    let moe_expert_cache =
        moe_expert_cache.filter(|_| device.is_cuda() && !write_uqff && !distributed);
    let WeightSources {
        uqff_reader,
        prepared: prepared_weight_source,
        combined: weight_source,
    } = open_weight_sources(uqff_files, prepared_weight_source)?;

    if let (Some(adjust), false, false, DeviceMapSetting::Auto(params)) =
        (auto_device_map_params, write_uqff, distributed, &mapper)
    {
        mapper = DeviceMapSetting::Auto(adjust(params)?);
    }
    let ResolvedMapSetting {
        setting: mapper,
        max_kv_tokens,
        planned_moe_expert_cache,
        prompt_chunk_tokens,
    } = resolve_map_setting(
        MapSettingInputs {
            setting: mapper,
            write_uqff,
            distributed,
            available_devices: &available_devices,
            dtype,
            sizing: super::isq_flow::AutoDeviceMapSizingInputs {
                loader: mapped,
                config,
                sizing: super::isq_flow::resolve_auto_device_map_sizing(
                    uqff_reader.is_some(),
                    prepared_weight_source.is_some(),
                    in_situ_quant,
                ),
                weight_source: weight_source.as_ref(),
                prepared_weight_source: prepared_weight_source.as_ref(),
                topology: settings.topology,
                organization: settings.organization,
                weight_filenames: paths.get_weight_filenames(),
                has_lora,
                matformer: matformer.as_ref().filter(|_| matformer_sizing),
                non_mapped_unpacked,
                embeddings_on_host,
                moe_expert_cache,
            },
        },
        paged_attn_config,
    )?;

    let MaterializedDeviceMapper {
        pipeline_mapper,
        mapper,
        layer_devices,
        dtype,
    } = materialize_device_mapper(
        DeviceMapperInputs {
            setting: &mapper,
            num_layers: mapped.num_layers(config)?,
            device: &device,
            available_devices: &available_devices,
            topology: settings.topology,
            write_uqff,
            dtype,
        },
        paged_attn_config,
    )?;

    if crate::using_flash_attn() {
        inference_quant::log::once_log_info("FlashAttention is enabled.");
    }
    // a UQFF write builds the model on the host; its dtype still follows the requested device
    let device = if write_uqff { Device::Cpu } else { device };

    let plan = super::isq_flow::resolve_and_install_isq_plan(super::isq_flow::IsqPlanInputs {
        in_situ_quant,
        has_imatrix: settings.has_imatrix,
        has_calibration: settings.has_calibration,
        write_uqff_types: settings.write_uqff.map(|c| c.types.clone()),
        has_write_uqff: write_uqff,
        loading_from_uqff: settings.from_uqff,
        organization: settings.organization,
        topology_overrides: {
            let mut overrides = settings
                .topology
                .map(|topology| topology.immediate_overrides())
                .unwrap_or_default();
            let offloaded = planned_moe_expert_cache.or(moe_expert_cache).is_some();
            if offloaded
                && weight_source
                    .as_ref()
                    .is_some_and(|source| source.contains(super::isq_flow::MTP_HEAD_EXPERTS[0].0))
            {
                overrides.extend(mtp_head_expert_overrides()?);
            }
            overrides
        },
        loader: isq,
        config,
        device: &device,
    })?;

    let attention = if paged_attn_config.is_some() {
        inference_nn::paged_attention::AttentionImplementation::PagedAttention
    } else {
        inference_nn::paged_attention::AttentionImplementation::Eager
    };
    let load_parts = LoadMetadataParts {
        loading_isq: plan.loading_isq,
        device: device.clone(),
        multi_progress: Arc::new(crate::utils::progress::new_multi_progress()),
        matformer,
    };

    info!(
        "{}",
        super::isq::WeightLoadingMode::from(super::isq::WeightLoadingState {
            from_uqff: settings.from_uqff,
            loading_isq: plan.loading_isq,
            immediate_isq: plan.immediate_isq_installed,
            write_uqff,
        })
        .message(weight_target)
    );

    Ok((
        LoadSession {
            tensor_parallelism,
            device,
            available_devices,
            uqff_reader,
            weight_source,
            max_kv_tokens,
            prompt_chunk_tokens,
            pipeline_mapper,
            layer_devices,
            dtype,
            plan,
            attention,
            load_parts,
            embeddings_on_host,
            moe_expert_cache: planned_moe_expert_cache.or(moe_expert_cache),
        },
        mapper,
    ))
}

/// The model-building half of a pipeline loader, so [`load_model`] drives all three pipelines.
pub(crate) trait BuildModel {
    type Model: ?Sized;
    fn as_isq(&self) -> &dyn super::IsqModelLoader;
    fn as_mapped(&self) -> &dyn super::loaders::DeviceMappedModelLoader;
    fn build(
        &self,
        config: &str,
        vb: inference_quant::ShardedVarBuilder,
        metadata: crate::pipeline::NormalLoadingMetadata,
        attention: inference_nn::paged_attention::AttentionImplementation,
    ) -> Result<Box<Self::Model>>;
    fn device_for_tensor(
        &self,
        config: &str,
        mapper: &dyn DeviceMapper,
        loading_isq: bool,
    ) -> Result<DeviceForTensor>;
}

macro_rules! build_model_for {
    ($loader:path => $model:path) => {
        impl BuildModel for dyn $loader {
            type Model = dyn $model + Send + Sync;
            fn as_isq(&self) -> &dyn super::IsqModelLoader {
                self
            }
            fn as_mapped(&self) -> &dyn super::loaders::DeviceMappedModelLoader {
                self
            }
            fn build(
                &self,
                config: &str,
                vb: inference_quant::ShardedVarBuilder,
                metadata: crate::pipeline::NormalLoadingMetadata,
                attention: inference_nn::paged_attention::AttentionImplementation,
            ) -> Result<Box<Self::Model>> {
                self.load(config, vb, metadata, attention)
            }
            fn device_for_tensor(
                &self,
                config: &str,
                mapper: &dyn DeviceMapper,
                loading_isq: bool,
            ) -> Result<DeviceForTensor> {
                self.get_device_for_tensor(config, mapper, loading_isq)
            }
        }
    };
}

build_model_for!(super::loaders::NormalModelLoader => crate::pipeline::NormalModel);
build_model_for!(super::loaders::MultimodalModelLoader => crate::pipeline::MultimodalModel);
build_model_for!(super::loaders::EmbeddingModelLoader => crate::pipeline::EmbeddingModel);

pub(crate) type LoadedModel<M> = (
    Box<M>,
    inference_quant::Tracker,
    Option<Arc<crate::DynamicLoraRuntime>>,
);

/// What [`load_model`] reads beyond the session.
pub(crate) struct ModelLoadInputs<'a> {
    pub config: &'a str,
    pub paths: &'a dyn super::ModelPaths,
    pub silent: bool,
    pub organization: super::IsqOrganization,
    pub from_uqff: bool,
    pub write_uqff: bool,
    pub prepared: Option<&'a PreparedSource>,
    pub lora: Option<crate::LoraRuntimeConfig>,
}

/// Builds the model over a tensor-parallel shard, a prepared source or the weight files, with LoRA if asked.
pub(crate) fn load_model<L: BuildModel + ?Sized>(
    loader: &L,
    session: &LoadSession,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
    inputs: ModelLoadInputs<'_>,
) -> Result<LoadedModel<L::Model>> {
    let ModelLoadInputs {
        config,
        paths,
        silent,
        organization,
        from_uqff,
        write_uqff,
        prepared,
        lora,
    } = inputs;
    let prepared = prepared.map(|source| (&source.weights, source.rope_pairing));
    let loading_isq = session.plan.loading_isq;
    let distributed = session.tensor_parallelism.is_enabled();
    let weights = WeightFiles {
        paths,
        dtype: session.dtype,
        device: &session.plan.load_device,
        layer_devices: session.layer_devices.clone(),
        silent,
        uqff_reader: session.uqff_reader.clone(),
    };
    let from_files = |mapper: &dyn DeviceMapper| -> Result<inference_quant::ShardedVarBuilder> {
        let placeholders = uqff_placeholders(
            loader.as_isq(),
            config,
            loading_isq,
            from_uqff,
            matches!(organization, super::IsqOrganization::MoeExpertsOnly),
        )?;
        weights.load(
            placeholders,
            loader.device_for_tensor(config, mapper, loading_isq)?,
        )
    };

    let (mapper, sharded) = if distributed {
        let (mapper, sharded) =
            distributed::prepare_distributed_mapper(distributed::DistributedMapperConfig {
                dtype: session.dtype,
                device: &session.device,
                available_devices: &session.available_devices,
                global_world_size_override: session.tensor_parallelism.world_size(),
                silent,
                config,
                loading_isq,
                from_uqff,
                write_uqff,
                organization,
                isq_loader: loader.as_isq(),
                mapped_loader: loader.as_mapped(),
                weights: match prepared {
                    Some((weights, _)) => {
                        distributed::DistributedWeightSource::Prepared(weights.clone())
                    }
                    None => distributed::DistributedWeightSource::Paths(paths),
                },
            })?;
        let sharded = match session.uqff_reader.clone() {
            Some(reader) => sharded.with_uqff_reader(reader),
            None => sharded,
        };
        (mapper, Some(sharded))
    } else {
        (mapper, None)
    };
    let vb = match (sharded, prepared) {
        // a LoRA load with no prepared source reads the files even when tensor parallel
        (Some(_), None) if lora.is_some() => from_files(&*mapper)?,
        (Some(sharded), _) => sharded,
        (None, Some((weights, _))) => weights
            .clone()
            .set_dtype(session.dtype)
            .set_device(session.plan.load_device.clone()),
        (None, None) => from_files(&*mapper)?,
    };
    let rope_pairing = prepared.map(|(_, rope_pairing)| rope_pairing);
    // A UQFF written from a GGUF keeps the stamped layout without a prepared source
    let lora_rope_pairing = match rope_pairing {
        Some(pairing) => Some(pairing),
        None => super::loaders::qk_rope_layout_from_config(config)?,
    };
    let layers = lora
        .map(|_| super::normal::new_dynamic_lora_registry(config, lora_rope_pairing))
        .transpose()?;
    let vb = match &layers {
        Some(layers) => vb.with_lora_registry(layers.clone()),
        None => vb,
    };
    let vb = vb
        .with_embeddings_on_host(session.embeddings_on_host)
        .with_moe_expert_cache(session.moe_expert_cache);
    let tracker = vb.tracker().clone();
    let model = loader.build(
        config,
        vb,
        session.load_parts.metadata(mapper, rope_pairing),
        session.attention,
    )?;
    let dynamic_lora = match (layers, lora) {
        (Some(layers), Some(runtime)) => Some(super::finish_dynamic_lora_runtime(
            paths,
            layers,
            runtime,
            !distributed,
        )?),
        _ => None,
    };
    Ok((model, tracker, dynamic_lora))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_auto_cache_is_quantized_only_when_experts_leave_the_gpu() -> Result<()> {
        use crate::PagedCacheType;
        let resident = || Ok(true);
        let crowded = || Ok(false);
        let q8 = PagedCacheType::Quant { k: 8, v: 8 };
        assert_eq!(
            offload_cache_type(PagedCacheType::Auto, true, false, crowded)?,
            OFFLOAD_CACHE_TYPE
        );
        assert_eq!(
            offload_cache_type(PagedCacheType::Auto, true, true, resident)?,
            OFFLOAD_CACHE_TYPE
        );
        assert_eq!(
            offload_cache_type(PagedCacheType::Auto, true, false, resident)?,
            PagedCacheType::Auto
        );
        assert_eq!(
            offload_cache_type(PagedCacheType::Auto, false, true, crowded)?,
            PagedCacheType::Auto
        );
        for pinned in [PagedCacheType::Unquantized, q8] {
            assert_eq!(offload_cache_type(pinned, true, true, crowded)?, pinned);
        }
        Ok(())
    }

    #[test]
    fn only_a_planned_offload_with_the_default_memory_sizes_the_cache_to_its_context() {
        assert!(sizes_cache_to_context(false, false, true));
        assert!(!sizes_cache_to_context(true, false, true));
        assert!(!sizes_cache_to_context(false, true, true));
        assert!(!sizes_cache_to_context(false, false, false));
    }

    #[test]
    fn only_an_explicitly_untied_head_lets_the_embedding_leave_the_device() {
        assert!(config_unties_embeddings(
            r#"{"tie_word_embeddings": false}"#
        ));
        assert!(config_unties_embeddings(
            r#"{"text_config": {"tie_word_embeddings": false}}"#
        ));
        assert!(!config_unties_embeddings(
            r#"{"tie_word_embeddings": true}"#
        ));
        assert!(!config_unties_embeddings(r#"{"hidden_size": 8}"#));
        assert!(!config_unties_embeddings("not json"));
    }

    fn paths_with_gen_conf(gen_conf: PathBuf) -> crate::pipeline::LocalModelPaths<PathBuf> {
        crate::pipeline::LocalModelPaths {
            tokenizer_filename: PathBuf::new(),
            config_filename: PathBuf::new(),
            template_filename: None,
            filenames: Vec::new(),
            adapter_paths: crate::pipeline::AdapterPaths::None,
            gen_conf: Some(gen_conf),
            preprocessor_config: None,
            video_preprocessor_config: None,
            processor_config: None,
            chat_template_json_filename: None,
        }
    }

    fn temperature(conf: Option<crate::pipeline::chat_template::GenerationConfig>) -> Option<f64> {
        conf.and_then(|conf| conf.generation_defaults())
            .and_then(|defaults| defaults.temperature)
    }

    #[test]
    fn a_malformed_generation_config_falls_back_to_the_model_config() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("generation_config.json");
        let config = r#"{"temperature": 0.5}"#;

        std::fs::write(&file, r#"{"temperature": 0.2}"#).unwrap();
        let paths = paths_with_gen_conf(file.clone());
        assert_eq!(
            temperature(generation_config(None, &paths, config)),
            Some(0.2)
        );

        std::fs::write(&file, "{not json").unwrap();
        assert_eq!(
            temperature(generation_config(None, &paths, config)),
            Some(0.5)
        );
        assert_eq!(
            temperature(generation_config(Some(None), &paths, config)),
            Some(0.5)
        );
        let prepared = serde_json::from_str(r#"{"temperature": 0.7}"#).unwrap();
        assert_eq!(
            temperature(generation_config(Some(Some(prepared)), &paths, config)),
            Some(0.7)
        );
    }

    #[test]
    fn a_matformer_slice_without_its_config_is_refused() {
        let error = load_matformer_slice(None, Some("small")).unwrap_err();
        assert!(
            error.to_string().contains("matformer_config_path"),
            "{error}"
        );
        assert!(load_matformer_slice(None, None).unwrap().is_none());
    }
}

#[cfg(test)]
mod expert_slot_plan_tests {
    use super::{
        PLANNED_OFFLOAD_CONTEXT, plan_expert_slots, planned_offload_context, shrink_expert_slots,
    };

    #[test]
    fn an_asked_context_replaces_the_planned_default() {
        let window = 256 * 1024;
        assert_eq!(
            planned_offload_context(4096, 0, window),
            PLANNED_OFFLOAD_CONTEXT
        );
        assert_eq!(planned_offload_context(4096, 8192, window), 8192);
        assert_eq!(planned_offload_context(4096, 0, 2048), 4096);
        assert_eq!(planned_offload_context(65536, 8192, window), 65536);
    }

    const EXPERTS: usize = 256;

    // a model whose weights fit with `cached` experts per layer at the long context, or `short` at the short one
    fn plan(cached: Option<usize>, short: Option<usize>) -> Option<(usize, u8)> {
        plan_expert_slots(EXPERTS, [0u8, 1], |slots, context| {
            let limit = if context == 0 { cached } else { short };
            Ok(match (slots, limit) {
                (None, _) => limit == Some(EXPERTS),
                (Some(slots), Some(limit)) => slots <= limit,
                (Some(_), None) => false,
            })
        })
        .unwrap()
    }

    #[test]
    fn a_model_that_fits_resident_caches_nothing() {
        assert_eq!(plan(Some(EXPERTS), None), None);
    }

    #[test]
    fn the_most_slots_that_fit_win_in_sixty_fourths_of_the_experts() {
        assert_eq!(plan(Some(202), None), Some((200, 0)));
        assert_eq!(plan(Some(3), None), Some((0, 0)));
    }

    #[test]
    fn the_shorter_context_is_tried_only_when_the_longer_fits_nothing() {
        assert_eq!(plan(None, Some(102)), Some((100, 1)));
        assert_eq!(plan(None, None), None);
    }

    #[test]
    fn a_requested_cache_shrinks_to_the_most_slots_below_it_that_fit() {
        let shrink = |requested, limit: Option<usize>| {
            shrink_expert_slots(requested, EXPERTS, |slots| {
                Ok(limit.is_some_and(|l| slots <= l))
            })
            .unwrap()
        };
        assert_eq!(shrink(160, Some(150)), Some(148));
        // never more than requested, even when more would fit
        assert_eq!(shrink(40, Some(200)), Some(36));
        assert_eq!(shrink(160, Some(0)), Some(0));
        assert_eq!(shrink(160, None), None);
    }
}
