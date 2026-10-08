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
    device_map::{self, DeviceMapSetting, DeviceMapper},
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
    let pipeline_mapper = build()?;
    let mapper = build()?;
    let layer_devices = (0..inputs.num_layers)
        .map(|layer| mapper.device_for(layer, false).cloned())
        .collect();
    let dtype = super::isq_flow::resolve_weight_load_dtype(
        inputs.dtype,
        mapper.as_ref(),
        inputs.available_devices,
        inputs.write_uqff,
    )?;

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
        });
    }
    if distributed {
        return Ok(ResolvedMapSetting {
            setting: DeviceMapSetting::DummyNccl {
                nm_device: available_devices[0].clone(),
            },
            max_kv_tokens: None,
        });
    }
    let DeviceMapSetting::Auto(params) = &setting else {
        return Ok(ResolvedMapSetting {
            setting,
            max_kv_tokens: None,
        });
    };
    let max_kv_tokens = Some(params.max_seq_len() * params.max_batch_size());
    let dtype = dtype.try_into_dtype(&available_devices.iter().collect::<Vec<_>>())?;
    let (loader, config) = (sizing.loader, sizing.config);
    let super::isq_flow::AutoDeviceMapSizes {
        layer_sizes_in_bytes,
        non_mapped_size_in_bytes,
        total_model_size_in_bytes,
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
}

/// Devices, weight sources, device mappers and the installed ISQ plan for one load.
pub(crate) struct LoadSession {
    pub tensor_parallelism: TensorParallelism,
    pub device: Device,
    pub available_devices: Vec<Device>,
    pub uqff_reader: Option<Arc<UqffReader>>,
    pub weight_source: Option<Arc<dyn QuantizedWeightSource>>,
    pub max_kv_tokens: Option<usize>,
    pub pipeline_mapper: Box<dyn DeviceMapper + Send + Sync>,
    pub layer_devices: Vec<Option<Device>>,
    pub dtype: DType,
    pub plan: super::isq_flow::IsqLoadPlan,
    pub attention: inference_nn::paged_attention::AttentionImplementation,
    pub load_parts: LoadMetadataParts,
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

    let plan = super::isq_flow::resolve_and_install_isq_plan(super::isq_flow::IsqPlanInputs {
        in_situ_quant,
        has_imatrix: settings.has_imatrix,
        has_calibration: settings.has_calibration,
        write_uqff_types: settings.write_uqff.map(|c| c.types.clone()),
        has_write_uqff: write_uqff,
        loading_from_uqff: settings.from_uqff,
        organization: settings.organization,
        topology_overrides: settings
            .topology
            .map(|topology| topology.immediate_overrides())
            .unwrap_or_default(),
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
            pipeline_mapper,
            layer_devices,
            dtype,
            plan,
            attention,
            load_parts,
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
