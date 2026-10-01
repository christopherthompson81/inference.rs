//! Load steps shared by the normal, multimodal and embedding pipeline loaders.

use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::Result;
use candle_core::{DType, Device};
use inference_quant::{QuantizedWeightSource, UqffReader};
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
    prepared_config: Option<&str>,
    config_filename: &Path,
    from_uqff: bool,
    overrides: Option<&super::HfConfigOverrides>,
    mtp: bool,
) -> Result<String> {
    let config = match prepared_config {
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

/// The checkpoint files a pipeline keeps for re-quantizing later; a UQFF load has none.
pub(crate) fn source_weight_files(
    prepared: Option<&[PathBuf]>,
    from_uqff: bool,
    weight_files: &[PathBuf],
) -> Vec<PathBuf> {
    match prepared {
        Some(files) => files.to_vec(),
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
    pub attention: inference_nn::paged_attention::AttentionImplementation,
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
        let files = self.paths.get_weight_filenames().to_vec();
        self.load_files(files, Vec::new(), placeholders, device_for_tensor)
    }

    /// An X-LoRA model's weights: the model's, its classifier's and its adapters'.
    pub fn load_xlora(
        &self,
        device_for_tensor: DeviceForTensor,
    ) -> Result<inference_quant::ShardedVarBuilder> {
        let super::AdapterPaths::XLora {
            adapter_safetensors,
            classifier_path,
            ..
        } = self.paths.get_adapter_paths()
        else {
            unreachable!("X-LoRA loaders require resolved X-LoRA adapter paths")
        };
        let classifier = classifier_path
            .clone()
            .expect("X-LoRA adapters name a classifier");
        let mut files = self.paths.get_weight_filenames().to_vec();
        files.push(classifier);
        let adapters = adapter_safetensors
            .iter()
            .flatten()
            .map(|(_, path)| path.clone())
            .collect();
        self.load_files(files, adapters, None, device_for_tensor)
    }

    fn load_files(
        &self,
        files: Vec<PathBuf>,
        adapter_files: Vec<PathBuf>,
        placeholders: Option<Vec<regex::Regex>>,
        device_for_tensor: DeviceForTensor,
    ) -> Result<inference_quant::ShardedVarBuilder> {
        let vb = crate::utils::varbuilder_utils::from_mmaped_safetensors(
            files,
            adapter_files,
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
