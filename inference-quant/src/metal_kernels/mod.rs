// Portions of this file are adapted from Apple's MLX framework
// (https://github.com/ml-explore/mlx)
// Licensed under the Apache License 2.0
// Copyright © 2023 Apple Inc.

use candle_core::{DType, MetalDevice};

use candle_metal_kernels::metal::{
    Buffer, ComputeCommandEncoder, ComputePipeline, ConstantValues, Device, Function, Library,
    MetalDeviceType, Value as ConstantValue,
};

use objc2_metal::{MTLDevice, MTLSize};

use std::os::raw::c_void;

use std::sync::{Arc, RwLock};

use std::{collections::HashMap, sync::OnceLock};

pub mod utils;

use utils::{
    get_2d_grid_dims, get_2d_grid_dims_divisor, get_block_dims, linear_split, EncoderParam,
    EncoderProvider, Output, RawBytesEncoder,
};

use crate::set_params;

// Backward-compatible aliases to ease the transition from the `metal` crate API.
type ComputeCommandEncoderRef = ComputeCommandEncoder;

type ComputePipelineState = ComputePipeline;

#[cfg(target_os = "macos")]
const KERNELS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/inference_quant.metallib"));

#[cfg(target_os = "ios")]
const KERNELS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/inference_quant_ios.metallib"));

#[cfg(target_os = "tvos")]
const KERNELS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/inference_quant_tvos.metallib"));

include!("source_set.rs");

#[derive(thiserror::Error, Debug)]
pub enum MetalKernelError {
    #[error("Could not lock kernel map: {0}")]
    LockError(String),
    #[error("Error while loading function: {0:?}")]
    LoadFunctionError(String),
    #[error("Failed to create pipeline: {0}")]
    FailedToCreatePipeline(String),
    #[error("dtype mismatch, got {got:?}, expected {expected:?}")]
    DTypeMismatch { expected: Vec<DType>, got: DType },
    #[error("Failed to compile Metal shader: {0}")]
    CompilationError(String),
}

impl<T> From<std::sync::PoisonError<T>> for MetalKernelError {
    fn from(e: std::sync::PoisonError<T>) -> Self {
        Self::LockError(e.to_string())
    }
}

type Pipelines = HashMap<(String, Option<ConstantValues>), ComputePipeline>;

static LIBRARY: OnceLock<Library> = OnceLock::new();

static GLOBAL_KERNELS: OnceLock<Kernels> = OnceLock::new();

#[derive(Debug)]
pub struct Kernels {
    pipelines: RwLock<Pipelines>,
}

impl Default for Kernels {
    fn default() -> Self {
        Self::new()
    }
}

impl Kernels {
    pub fn global() -> &'static Self {
        GLOBAL_KERNELS.get_or_init(Self::new)
    }

    pub fn new() -> Self {
        let pipelines = RwLock::new(Pipelines::new());
        Self { pipelines }
    }

    /// Load the library from precompiled metallib, falling back to runtime compilation if needed.
    /// If this has been previously loaded it will just fetch it from cache.
    #[allow(clippy::const_is_empty)] // KERNELS can be empty when INFERENCE_RS_METAL_PRECOMPILE=0
    pub fn load_library(&self, device: &Device) -> Result<Library, MetalKernelError> {
        if let Some(lib) = LIBRARY.get() {
            Ok(lib.clone())
        } else {
            // Try to load precompiled metallib first (faster startup)
            let lib = if !KERNELS.is_empty() {
                // Load precompiled metallib directly from embedded bytes via DispatchData.
                // This avoids writing to a temp file, which can fail in sandboxed
                // environments (e.g. macOS apps distributed via TestFlight).
                // https://github.com/EricLBuehler/mistral.rs/issues/1897
                let data = dispatch2::DispatchData::from_static_bytes(KERNELS);

                let raw_lib = device
                    .as_ref()
                    .newLibraryWithData_error(&data)
                    .map_err(|e| {
                        MetalKernelError::CompilationError(format!(
                            "Failed to load precompiled metallib: {e}"
                        ))
                    })?;
                Library::new(raw_lib)
            } else {
                // Fall back to runtime compilation if precompiled lib is not available
                // (e.g., when INFERENCE_RS_METAL_PRECOMPILE=0)
                self.compile_kernels_at_runtime(device)?
            };
            Ok(LIBRARY.get_or_init(|| lib).clone())
        }
    }

    fn compile_kernels_at_runtime(&self, device: &Device) -> Result<Library, MetalKernelError> {
        inference_metal_compile::compile_runtime_library(device, &QUANT_METAL_SOURCE_SET)
            .map_err(MetalKernelError::CompilationError)
    }

    fn load_function(
        &self,
        device: &Device,
        name: impl ToString,
        constants: Option<&ConstantValues>,
    ) -> Result<Function, MetalKernelError> {
        let func = self
            .load_library(device)?
            .get_function(&name.to_string(), constants)
            .map_err(|e| MetalKernelError::LoadFunctionError(e.to_string()))?;
        Ok(func)
    }

    /// Load a kernel pipeline by name without function constants.
    pub fn load_pipeline(
        &self,
        device: &Device,
        name: impl ToString,
    ) -> Result<ComputePipelineState, MetalKernelError> {
        self.load_pipeline_with_constants(device, name, None)
    }

    /// Load a kernel pipeline, specializing it with the given Metal function
    /// constants if any. Cached per (name, constants) so distinct
    /// specializations don't share a slot. Hot path (no constants): a
    /// read-lock lookup, no contention.
    pub fn load_pipeline_with_constants(
        &self,
        device: &Device,
        name: impl ToString,
        constants: Option<ConstantValues>,
    ) -> Result<ComputePipelineState, MetalKernelError> {
        let name_str = name.to_string();
        if constants.is_none() {
            let pipelines = self.pipelines.read()?;
            if let Some(pipeline) = pipelines.get(&(name_str.clone(), None)) {
                return Ok(pipeline.clone());
            }
        }
        let mut pipelines = self.pipelines.write()?;
        let key = (name_str, constants);
        if let Some(pipeline) = pipelines.get(&key) {
            return Ok(pipeline.clone());
        }
        let (name, constants) = key;
        let func = self.load_function(device, &name, constants.as_ref())?;
        let pipeline = device
            .new_compute_pipeline_state_with_function(&func)
            .map_err(|e| MetalKernelError::FailedToCreatePipeline(e.to_string()))?;
        pipelines.insert((name, constants), pipeline.clone());
        Ok(pipeline)
    }
}

mod dequant;
pub use dequant::*;
mod bitwise;
pub use bitwise::*;
mod bnb;
pub use bnb::*;
mod afq;
pub use afq::*;
mod moe;
pub use moe::*;
mod mxfp4;
pub use mxfp4::*;
mod blockwise_fp8;
pub use blockwise_fp8::*;
mod scan;
pub use scan::*;
mod sort;
pub use sort::*;
mod hqq;
pub use hqq::*;
mod elementwise;
pub use elementwise::*;
mod rotary;
pub use rotary::*;
mod attention_sinks;
pub use attention_sinks::*;
mod fp8;
pub use fp8::*;
mod flash_attn;
pub use flash_attn::*;
mod rmsnorm;
pub use rmsnorm::*;
mod logits;
pub use logits::*;
