// Portions of this file are adapted from Apple's MLX framework
// (https://github.com/ml-explore/mlx)
// Licensed under the Apache License 2.0
// Copyright © 2023 Apple Inc.

use super::*;

#[allow(clippy::too_many_arguments)]
pub fn call_rmsnorm_residual(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    ty: DType,
    x: (&Buffer, usize),
    residual: (&Buffer, usize),
    weight: (&Buffer, usize),
    scale: Option<(&Buffer, usize)>,
    out: &Buffer,
    n_cols: usize,
    n_rows: usize,
    eps: f32,
) -> Result<(), MetalKernelError> {
    let name = match ty {
        DType::F32 => "rmsnorm_residual_f32",
        DType::F16 => "rmsnorm_residual_f16",
        DType::BF16 => "rmsnorm_residual_bf16",
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

    let n_cols_u32 = n_cols as u32;
    let has_scale_u32: u32 = if scale.is_some() { 1 } else { 0 };

    encoder.set_input_buffer(0, Some(x.0), x.1);
    encoder.set_input_buffer(1, Some(residual.0), residual.1);
    encoder.set_input_buffer(2, Some(weight.0), weight.1);
    let (scale_buf, scale_off) = scale.unwrap_or((weight.0, weight.1));
    encoder.set_input_buffer(3, Some(scale_buf), scale_off);
    encoder.set_output_buffer(4, Some(out), 0);
    <u32 as EncoderParam>::set_param(encoder, 5, n_cols_u32);
    <f32 as EncoderParam>::set_param(encoder, 6, eps);
    <u32 as EncoderParam>::set_param(encoder, 7, has_scale_u32);

    let tg_size = 256usize.min(n_cols.next_power_of_two().max(32));
    encoder.set_threadgroup_memory_length(0, tg_size * std::mem::size_of::<f32>());

    let group = MTLSize {
        width: tg_size,
        height: 1,
        depth: 1,
    };
    let grid = MTLSize {
        width: n_rows,
        height: 1,
        depth: 1,
    };
    encoder.dispatch_thread_groups(grid, group);
    Ok(())
}
