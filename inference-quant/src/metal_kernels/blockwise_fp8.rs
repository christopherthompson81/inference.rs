// Portions of this file are adapted from Apple's MLX framework
// (https://github.com/ml-explore/mlx)
// Licensed under the Apache License 2.0
// Copyright © 2023 Apple Inc.

use super::*;

#[allow(clippy::too_many_arguments)]
pub fn call_dequant_blockwise_fp8(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    ty: DType,
    weight: &Buffer,
    scales: &Buffer,
    output: &Buffer,
    weight_height: u32,
    weight_width: u32,
    weight_row_stride: u32,
    scale_stride: u32,
    block_size_y: u32,
    block_size_x: u32,
) -> Result<(), MetalKernelError> {
    #[repr(C)]
    struct DequantParams {
        weight_height: u32,
        weight_width: u32,
        weight_row_stride: u32,
        scale_stride: u32,
        block_size_y: u32,
        block_size_x: u32,
    }

    let name = match ty {
        DType::F32 => "dequant_fp8_blockwise_float",
        DType::BF16 => "dequant_fp8_blockwise_bfloat16_t",
        DType::F16 => "dequant_fp8_blockwise_half",
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

    let dequant_params = DequantParams {
        weight_height,
        weight_width,
        weight_row_stride,
        scale_stride,
        block_size_y,
        block_size_x,
    };

    impl EncoderParam for &DequantParams {
        fn set_param(encoder: &ComputeCommandEncoderRef, position: usize, data: Self) {
            encoder.set_bytes_raw(
                position,
                core::mem::size_of_val(data),
                data as *const _ as *const c_void,
            );
        }
    }

    set_params!(
        encoder,
        (weight, scales, Output::new(output), &dequant_params)
    );

    let tg = MTLSize {
        width: 32,
        height: 32,
        depth: 1,
    };
    let blocks = MTLSize {
        width: weight_width.div_ceil(dequant_params.block_size_x) as usize,
        height: weight_height.div_ceil(dequant_params.block_size_y) as usize,
        depth: 1,
    };

    encoder.dispatch_thread_groups(blocks, tg);
    Ok(())
}
