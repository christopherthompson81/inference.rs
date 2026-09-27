//! Load steps shared by the normal and multimodal pipeline loaders.

use std::path::Path;

use anyhow::Result;
use candle_core::{DType, Device};
use tracing::warn;

use crate::{
    device_map::{DeviceMapSetting, DeviceMapper},
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
