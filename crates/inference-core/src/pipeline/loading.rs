//! Load steps shared by the normal and multimodal pipeline loaders.

use std::path::Path;

use anyhow::Result;
use candle_core::{DType, Device};
use tracing::warn;

use crate::{
    device_map::{self, DeviceMapSetting, DeviceMapper},
    distributed::{self, TensorParallelism, WorkerTransferData},
    paged_attention::ModelConfigLike,
    PagedAttentionConfig, Topology, TryIntoDType,
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
        warn!("Device mapping contains a mix of GPU and CPU. There is no CPU support for PagedAttention, disabling PagedAttention.");
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
