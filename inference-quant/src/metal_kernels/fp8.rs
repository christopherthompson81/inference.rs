// Portions of this file are adapted from Apple's MLX framework
// (https://github.com/ml-explore/mlx)
// Licensed under the Apache License 2.0
// Copyright © 2023 Apple Inc.

use super::*;

// ============================================================================
// Scalar FP8 conversion kernels
// ============================================================================

/// Convert FP8 E4M3 tensor to another dtype (F32, F16, BF16)
#[allow(clippy::too_many_arguments)]
pub fn call_fp8_to_dtype(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    out_ty: DType,
    input: &Buffer,
    output: &Buffer,
    num_elements: usize,
) -> Result<(), MetalKernelError> {
    let name = match out_ty {
        DType::F32 => "fp8_to_dtype_float",
        DType::F16 => "fp8_to_dtype_half",
        DType::BF16 => "fp8_to_dtype_bfloat16_t",
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

    set_params!(encoder, (input, Output::new(output), num_elements as u32));

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, num_elements);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

/// Convert tensor (F32, F16, BF16) to FP8 E4M3 with clamping
#[allow(clippy::too_many_arguments)]
pub fn call_dtype_to_fp8(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    in_ty: DType,
    input: &Buffer,
    output: &Buffer,
    num_elements: usize,
) -> Result<(), MetalKernelError> {
    let name = match in_ty {
        DType::F32 => "dtype_to_fp8_float",
        DType::F16 => "dtype_to_fp8_half",
        DType::BF16 => "dtype_to_fp8_bfloat16_t",
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

    set_params!(encoder, (input, Output::new(output), num_elements as u32));

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, num_elements);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

/// Per-tensor FP8 dequantization: output = fp8_weight * scale_inv
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub fn call_fp8_pertensor_dequant(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    out_ty: DType,
    weight: &Buffer,
    scale_inv: &Buffer,
    output: &Buffer,
    num_elements: usize,
) -> Result<(), MetalKernelError> {
    let name = match out_ty {
        DType::F32 => "fp8_pertensor_dequant_float",
        DType::F16 => "fp8_pertensor_dequant_half",
        DType::BF16 => "fp8_pertensor_dequant_bfloat16_t",
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
        (weight, scale_inv, Output::new(output), num_elements as u32)
    );

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, num_elements);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

/// Vector FP8 dequantization: output[i] = fp8_weight[i] * scale[i / 128]
/// Each group of 128 elements shares one scale
#[allow(clippy::too_many_arguments)]
pub fn call_fp8_vector_dequant(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    out_ty: DType,
    weight: &Buffer,
    scale: &Buffer,
    output: &Buffer,
    num_elements: usize,
) -> Result<(), MetalKernelError> {
    let name = match out_ty {
        DType::F32 => "fp8_vector_dequant_float",
        DType::F16 => "fp8_vector_dequant_half",
        DType::BF16 => "fp8_vector_dequant_bfloat16_t",
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
        (weight, scale, Output::new(output), num_elements as u32)
    );

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, num_elements);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}
