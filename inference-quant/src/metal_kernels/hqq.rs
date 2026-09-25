// Portions of this file are adapted from Apple's MLX framework
// (https://github.com/ml-explore/mlx)
// Licensed under the Apache License 2.0
// Copyright © 2023 Apple Inc.

use super::*;

// HQQ Bitpacking functions

pub fn call_hqq_pack_8bit(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    input: &Buffer,
    output: &Buffer,
    num_elements: usize,
) -> Result<(), MetalKernelError> {
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    let pipeline = kernels.load_pipeline(device, "pack_8bit")?;
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(encoder, (input, Output::new(output), num_elements));

    let grid_size = MTLSize {
        width: num_elements.div_ceil(256),
        height: 1,
        depth: 1,
    };
    let threadgroup_size = MTLSize {
        width: 256,
        height: 1,
        depth: 1,
    };

    encoder.dispatch_thread_groups(grid_size, threadgroup_size);
    Ok(())
}

pub fn call_hqq_pack_4bit(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    input: &Buffer,
    output: &Buffer,
    height: usize,
    width: usize,
) -> Result<(), MetalKernelError> {
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    let pipeline = kernels.load_pipeline(device, "pack_4bit")?;
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(encoder, (input, Output::new(output), height, width));

    let step = height / 2;
    let grid_size = MTLSize {
        width: step.div_ceil(16),
        height: width.div_ceil(16),
        depth: 1,
    };
    let threadgroup_size = MTLSize {
        width: 16,
        height: 16,
        depth: 1,
    };

    encoder.dispatch_thread_groups(grid_size, threadgroup_size);
    Ok(())
}

#[allow(dead_code)]
pub fn call_hqq_pack_2bit(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    input: &Buffer,
    output: &Buffer,
    height: usize,
    width: usize,
) -> Result<(), MetalKernelError> {
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    let pipeline = kernels.load_pipeline(device, "pack_2bit")?;
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(encoder, (input, Output::new(output), height, width));

    let step = height / 4;
    let grid_size = MTLSize {
        width: step.div_ceil(16),
        height: width.div_ceil(16),
        depth: 1,
    };
    let threadgroup_size = MTLSize {
        width: 16,
        height: 16,
        depth: 1,
    };

    encoder.dispatch_thread_groups(grid_size, threadgroup_size);
    Ok(())
}

#[allow(dead_code)]
pub fn call_hqq_pack_3bit(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    input: &Buffer,
    output: &Buffer,
    height: usize,
    width: usize,
) -> Result<(), MetalKernelError> {
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    let pipeline = kernels.load_pipeline(device, "pack_3bit")?;
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(encoder, (input, Output::new(output), height, width));

    let step = height / 10;
    let grid_size = MTLSize {
        width: step.div_ceil(16),
        height: width.div_ceil(16),
        depth: 1,
    };
    let threadgroup_size = MTLSize {
        width: 16,
        height: 16,
        depth: 1,
    };

    encoder.dispatch_thread_groups(grid_size, threadgroup_size);
    Ok(())
}

#[allow(dead_code)]
pub fn call_hqq_pack_1bit(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    input: &Buffer,
    output: &Buffer,
    height: usize,
    width: usize,
) -> Result<(), MetalKernelError> {
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    let pipeline = kernels.load_pipeline(device, "pack_1bit")?;
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(encoder, (input, Output::new(output), height, width));

    let step = height / 8;
    let grid_size = MTLSize {
        width: step.div_ceil(16),
        height: width.div_ceil(16),
        depth: 1,
    };
    let threadgroup_size = MTLSize {
        width: 16,
        height: 16,
        depth: 1,
    };

    encoder.dispatch_thread_groups(grid_size, threadgroup_size);
    Ok(())
}
