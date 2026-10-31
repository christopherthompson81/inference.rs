use std::{fmt::Debug, sync::Arc};

use crate::utils::normal::TryIntoDType;
use inference_quant::ShardedVarBuilder;
use inference_tensor::{DType, Device, Result, Tensor};

use super::peer::CudaPeerAccess;

pub trait DeviceMapper: Debug {
    fn map(&self, input: Tensor, layer: usize) -> Result<Tensor>;

    fn set_device(
        &self,
        layer: usize,
        varbuilder: ShardedVarBuilder,
        loading_isq: bool,
    ) -> ShardedVarBuilder;
    fn device_for(&self, layer: usize, loading_isq: bool) -> Option<&Device>;
    fn get_unique_devices(&self) -> Vec<Device>;
    fn cast_nm_device(&self, x: &Tensor, loading_isq: bool) -> Result<Tensor>;
    fn set_nm_device(&self, varbuilder: ShardedVarBuilder, loading_isq: bool) -> ShardedVarBuilder;
    fn num_device_mapping_layers(&self) -> usize;
    fn get_comm_for(&self, layer_idx: usize) -> Result<Arc<inference_quant::Comm>>;

    fn get_min_dtype(&self, dtype: &dyn TryIntoDType) -> Result<DType>;

    /// Tells the mapper the model's activation dtype, so CPU layers beside an accelerator can depart from it.
    fn set_model_dtype(&mut self, _dtype: DType) {}

    /// The activation dtype on `device` when it differs from the model's: F32 on the CPU layers of a map that also
    /// uses an accelerator, since the CPU kernels have no fast BF16 or F16 arithmetic.
    fn activation_dtype_on(&self, _device: &Device) -> Option<DType> {
        None
    }
}

// Casts a float activation between the model dtype and the CPU layers' F32; other tensors (ids, positions) pass
fn cast_activation(x: Tensor, model: DType, to: DType) -> Result<Tensor> {
    if x.dtype() != to && (x.dtype() == model || x.dtype() == DType::F32) {
        x.to_dtype(to)
    } else {
        Ok(x)
    }
}

#[derive(Debug)]
pub struct LayerDeviceMapper {
    mappings: Vec<Device>,
    nm_device: Device,
    cuda_peer_access: CudaPeerAccess,
    // the model dtype, when CPU layers run in F32 beside an accelerator
    cpu_f32_from: Option<DType>,
}

impl LayerDeviceMapper {
    pub(super) fn new(
        mappings: Vec<Device>,
        nm_device: Device,
        cuda_peer_access: CudaPeerAccess,
    ) -> Self {
        Self {
            mappings,
            nm_device,
            cuda_peer_access,
            cpu_f32_from: None,
        }
    }

    // a layer past the mapped stack (an MTP head's) sits with the head it feeds
    fn layer_device(&self, layer: usize) -> &Device {
        self.mappings.get(layer).unwrap_or(&self.nm_device)
    }

    fn dtype_on(&self, device: &Device) -> Option<(DType, DType)> {
        let model = self.cpu_f32_from?;
        Some((model, if device.is_cpu() { DType::F32 } else { model }))
    }
}

impl DeviceMapper for LayerDeviceMapper {
    fn map(&self, input: Tensor, layer: usize) -> Result<Tensor> {
        let device = self.layer_device(layer);
        let x = self.cuda_peer_access.to_device(&input, device)?;
        match self.dtype_on(device) {
            Some((model, to)) => cast_activation(x, model, to),
            None => Ok(x),
        }
    }
    fn set_device(
        &self,
        layer: usize,
        varbuilder: ShardedVarBuilder,
        loading_isq: bool,
    ) -> ShardedVarBuilder {
        let device = self.layer_device(layer);
        // an ISQ layer loads where it is quantized and moves later, but in its layer's dtype
        let varbuilder = match self.dtype_on(device) {
            Some((_, to)) => varbuilder.set_dtype(to),
            None => varbuilder,
        };
        if loading_isq {
            return varbuilder;
        }
        varbuilder.set_device(device.clone())
    }
    fn device_for(&self, layer: usize, loading_isq: bool) -> Option<&Device> {
        if loading_isq {
            return Some(&self.nm_device);
        }
        self.mappings.get(layer)
    }
    fn get_unique_devices(&self) -> Vec<Device> {
        self.mappings.iter().fold(Vec::new(), |mut acc, device| {
            if !acc.iter().any(|d| d.same_device(device)) {
                acc.push(device.clone());
            }
            acc
        })
    }
    fn cast_nm_device(&self, x: &Tensor, loading_isq: bool) -> Result<Tensor> {
        if loading_isq {
            return x.to_device(&Device::Cpu);
        }
        let x = self.cuda_peer_access.to_device(x, &self.nm_device)?;
        match self.dtype_on(&self.nm_device) {
            Some((model, to)) => cast_activation(x, model, to),
            None => Ok(x),
        }
    }
    fn set_nm_device(&self, varbuilder: ShardedVarBuilder, loading_isq: bool) -> ShardedVarBuilder {
        if loading_isq {
            return varbuilder;
        }
        let varbuilder = varbuilder.set_device(self.nm_device.clone());
        match self.dtype_on(&self.nm_device) {
            Some((_, to)) => varbuilder.set_dtype(to),
            None => varbuilder,
        }
    }
    fn get_min_dtype(&self, dtype: &dyn TryIntoDType) -> Result<DType> {
        dtype
            .try_into_dtype(&self.mappings.iter().collect::<Vec<_>>())
            .map_err(inference_tensor::Error::msg)
    }
    fn set_model_dtype(&mut self, dtype: DType) {
        let devices = self.mappings.iter().chain([&self.nm_device]);
        let (cpu, accelerator) = devices.fold((false, false), |(cpu, acc), d| {
            (cpu || d.is_cpu(), acc || !d.is_cpu())
        });
        self.cpu_f32_from = (cpu && accelerator && dtype != DType::F32).then_some(dtype);
    }
    fn activation_dtype_on(&self, device: &Device) -> Option<DType> {
        self.dtype_on(device)
            .map(|(_, to)| to)
            .filter(|&to| Some(to) != self.cpu_f32_from)
    }
    fn num_device_mapping_layers(&self) -> usize {
        self.mappings.len()
    }
    fn get_comm_for(&self, layer_idx: usize) -> Result<Arc<inference_quant::Comm>> {
        let id = inference_quant::Id::new();
        Ok(Arc::new(inference_quant::Comm::from_device(
            id,
            self.device_for(layer_idx, false).unwrap_or(&self.nm_device),
            0,
            1,
        )?))
    }
}

#[derive(Debug)]
pub struct DummyDeviceMapper {
    pub nm_device: Device,
}

impl DummyDeviceMapper {
    pub(super) fn new(nm_device: Device) -> Self {
        Self { nm_device }
    }
}

impl DeviceMapper for DummyDeviceMapper {
    fn map(&self, input: Tensor, _: usize) -> Result<Tensor> {
        Ok(input)
    }
    fn set_device(
        &self,
        _: usize,
        varbuilder: ShardedVarBuilder,
        loading_isq: bool,
    ) -> ShardedVarBuilder {
        if loading_isq {
            varbuilder.set_device(Device::Cpu)
        } else {
            varbuilder.set_device(self.nm_device.clone())
        }
    }
    fn device_for(&self, _: usize, _loading_isq: bool) -> Option<&Device> {
        Some(&self.nm_device)
    }
    fn get_unique_devices(&self) -> Vec<Device> {
        vec![self.nm_device.clone()]
    }
    fn cast_nm_device(&self, x: &Tensor, loading_isq: bool) -> Result<Tensor> {
        if loading_isq {
            x.to_device(&Device::Cpu)
        } else {
            x.to_device(&self.nm_device)
        }
    }
    fn set_nm_device(&self, varbuilder: ShardedVarBuilder, loading_isq: bool) -> ShardedVarBuilder {
        if loading_isq {
            varbuilder.set_device(Device::Cpu)
        } else {
            varbuilder.set_device(self.nm_device.clone())
        }
    }
    fn get_min_dtype(&self, dtype: &dyn TryIntoDType) -> Result<DType> {
        dtype
            .try_into_dtype(&[&self.nm_device])
            .map_err(inference_tensor::Error::msg)
    }
    fn num_device_mapping_layers(&self) -> usize {
        1
    }
    fn get_comm_for(&self, layer_idx: usize) -> Result<Arc<inference_quant::Comm>> {
        let id = inference_quant::Id::new();
        Ok(Arc::new(inference_quant::Comm::from_device(
            id,
            self.device_for(layer_idx, false).unwrap_or(&self.nm_device),
            0,
            1,
        )?))
    }
}

#[derive(Debug)]
pub struct NcclDeviceMapper {
    nm_device: Device,
    model_layers: usize,
    comm: Option<Arc<inference_quant::Comm>>,
}

impl NcclDeviceMapper {
    pub(super) fn new(
        nm_device: Device,
        model_layers: usize,
        comm: Option<Arc<inference_quant::Comm>>,
    ) -> Self {
        Self {
            nm_device,
            model_layers,
            comm,
        }
    }
}

impl DeviceMapper for NcclDeviceMapper {
    fn map(&self, input: Tensor, _: usize) -> Result<Tensor> {
        Ok(input)
    }
    fn set_device(
        &self,
        _: usize,
        varbuilder: ShardedVarBuilder,
        loading_isq: bool,
    ) -> ShardedVarBuilder {
        if loading_isq {
            varbuilder.set_device(Device::Cpu)
        } else {
            varbuilder.set_device(self.nm_device.clone())
        }
    }
    fn device_for(&self, _: usize, _loading_isq: bool) -> Option<&Device> {
        Some(&self.nm_device)
    }
    fn get_unique_devices(&self) -> Vec<Device> {
        vec![self.nm_device.clone()]
    }
    fn cast_nm_device(&self, x: &Tensor, loading_isq: bool) -> Result<Tensor> {
        if loading_isq {
            x.to_device(&Device::Cpu)
        } else {
            x.to_device(&self.nm_device)
        }
    }
    fn set_nm_device(&self, varbuilder: ShardedVarBuilder, loading_isq: bool) -> ShardedVarBuilder {
        if loading_isq {
            varbuilder.set_device(Device::Cpu)
        } else {
            varbuilder.set_device(self.nm_device.clone())
        }
    }
    fn get_min_dtype(&self, dtype: &dyn TryIntoDType) -> Result<DType> {
        dtype
            .try_into_dtype(&[&self.nm_device])
            .map_err(inference_tensor::Error::msg)
    }
    fn num_device_mapping_layers(&self) -> usize {
        self.model_layers
    }
    fn get_comm_for(&self, layer_idx: usize) -> Result<Arc<inference_quant::Comm>> {
        if let Some(comm) = &self.comm {
            Ok(comm.clone())
        } else {
            let id = inference_quant::Id::new();
            Ok(Arc::new(inference_quant::Comm::from_device(
                id,
                self.device_for(layer_idx, false).unwrap_or(&self.nm_device),
                0,
                1,
            )?))
        }
    }
}

#[derive(Debug)]
pub struct NcclPipelineParallelMapper {
    mappings: Vec<(Arc<inference_quant::Comm>, Device)>,
    nm_device: Device,
    cuda_peer_access: CudaPeerAccess,
}

impl DeviceMapper for NcclPipelineParallelMapper {
    fn map(&self, input: Tensor, layer: usize) -> Result<Tensor> {
        self.cuda_peer_access
            .to_device(&input, &self.mappings[layer].1)
    }
    fn set_device(
        &self,
        layer: usize,
        varbuilder: ShardedVarBuilder,
        loading_isq: bool,
    ) -> ShardedVarBuilder {
        if loading_isq {
            return varbuilder;
        }
        varbuilder.set_device(self.mappings[layer].1.clone())
    }
    fn device_for(&self, layer: usize, loading_isq: bool) -> Option<&Device> {
        if loading_isq {
            return Some(&self.nm_device);
        }
        self.mappings.get(layer).map(|(_, x)| x)
    }
    fn get_unique_devices(&self) -> Vec<Device> {
        self.mappings
            .iter()
            .fold(Vec::new(), |mut acc, (_, device)| {
                if !acc.iter().any(|d| d.same_device(device)) {
                    acc.push(device.clone());
                }
                acc
            })
    }
    fn cast_nm_device(&self, x: &Tensor, loading_isq: bool) -> Result<Tensor> {
        if loading_isq {
            x.to_device(&Device::Cpu)
        } else {
            self.cuda_peer_access.to_device(x, &self.nm_device)
        }
    }
    fn set_nm_device(&self, varbuilder: ShardedVarBuilder, loading_isq: bool) -> ShardedVarBuilder {
        if loading_isq {
            varbuilder
        } else {
            varbuilder.set_device(self.nm_device.clone())
        }
    }
    fn get_min_dtype(&self, dtype: &dyn TryIntoDType) -> Result<DType> {
        dtype
            .try_into_dtype(&self.mappings.iter().map(|(_, x)| x).collect::<Vec<_>>())
            .map_err(inference_tensor::Error::msg)
    }
    fn num_device_mapping_layers(&self) -> usize {
        self.mappings.len()
    }
    fn get_comm_for(&self, layer_idx: usize) -> Result<Arc<inference_quant::Comm>> {
        Ok(self.mappings[layer_idx].0.clone())
    }
}
