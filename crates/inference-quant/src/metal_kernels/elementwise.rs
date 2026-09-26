// Portions of this file are adapted from Apple's MLX framework
// (https://github.com/ml-explore/mlx)
// Licensed under the Apache License 2.0
// Copyright © 2023 Apple Inc.

use super::*;

#[allow(clippy::too_many_arguments)]
pub fn call_fused_glu(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    ty: DType,
    a: &Buffer,
    b: &Buffer,
    a_offset: usize,
    b_offset: usize,
    n_elements: usize,
    activation: i32,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let name = match ty {
        DType::F32 => "fused_glu_float",
        DType::F16 => "fused_glu_half",
        DType::BF16 => "fused_glu_bfloat",
        other => {
            return Err(MetalKernelError::DTypeMismatch {
                expected: vec![DType::F32, DType::F16, DType::BF16],
                got: other,
            })
        }
    };
    let pipeline = kernels.load_pipeline(device, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            (a, a_offset),
            (b, b_offset),
            Output::new(output),
            n_elements as u32,
            activation
        )
    );

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, n_elements);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_softcap(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    ty: DType,
    input: &Buffer,
    input_offset: usize,
    n_elements: usize,
    cap: f32,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let name = match ty {
        DType::F32 => "softcap_float",
        other => {
            return Err(MetalKernelError::DTypeMismatch {
                expected: vec![DType::F32],
                got: other,
            })
        }
    };
    let pipeline = kernels.load_pipeline(device, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            (input, input_offset),
            Output::new(output),
            n_elements as u32,
            cap
        )
    );

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, n_elements);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}
