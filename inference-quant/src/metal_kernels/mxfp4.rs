// Portions of this file are adapted from Apple's MLX framework
// (https://github.com/ml-explore/mlx)
// Licensed under the Apache License 2.0
// Copyright © 2023 Apple Inc.

use super::*;

#[allow(clippy::too_many_arguments)]
pub fn call_mxfp4_matmul(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    ty: DType,
    x: (&Buffer, usize),
    w: (&Buffer, usize),
    scales: (&Buffer, usize),
    bias: (&Buffer, usize),
    out: &Buffer,
    m: usize,
    n: usize,
    k: usize,
    has_bias: bool,
) -> Result<(), MetalKernelError> {
    let name = match ty {
        DType::F16 => "mxfp4_matmul_f16",
        DType::BF16 => "mxfp4_matmul_bf16",
        other => {
            return Err(MetalKernelError::DTypeMismatch {
                expected: vec![DType::F16, DType::BF16],
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
            x,
            w,
            scales,
            bias,
            Output::new(out),
            m as i32,
            n as i32,
            k as i32,
            has_bias as i32
        )
    );

    // 8 simdgroups * 32 = 256 threads. Grid.x tiles N in blocks of 8.
    let group_dims = MTLSize {
        width: 256,
        height: 1,
        depth: 1,
    };
    let grid_dims = MTLSize {
        width: n.div_ceil(8),
        height: m,
        depth: 1,
    };
    encoder.dispatch_thread_groups(grid_dims, group_dims);
    Ok(())
}

/// Optimized vecmat kernel for MXFP4 decode (M <= 4).
/// All 256 threads collaborate on K-reduction for 4 output columns.
#[allow(clippy::too_many_arguments)]
pub fn call_mxfp4_vecmat(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    ty: DType,
    x: (&Buffer, usize),
    w: (&Buffer, usize),
    scales: (&Buffer, usize),
    bias: (&Buffer, usize),
    out: &Buffer,
    m: usize,
    n: usize,
    k: usize,
    has_bias: bool,
) -> Result<(), MetalKernelError> {
    let name = match ty {
        DType::F16 => "mxfp4_vecmat_f16",
        DType::BF16 => "mxfp4_vecmat_bf16",
        other => {
            return Err(MetalKernelError::DTypeMismatch {
                expected: vec![DType::F16, DType::BF16],
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
            x,
            w,
            scales,
            bias,
            Output::new(out),
            m as i32,
            n as i32,
            k as i32,
            has_bias as i32
        )
    );

    // 256 threads, grid.x tiles N in blocks of 4 (kVecmatCols)
    let group_dims = MTLSize {
        width: 256,
        height: 1,
        depth: 1,
    };
    let grid_dims = MTLSize {
        width: n.div_ceil(4),
        height: m,
        depth: 1,
    };
    encoder.dispatch_thread_groups(grid_dims, group_dims);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_mxfp4_moe_gemm(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    ty: DType,
    x: (&Buffer, usize),
    w: (&Buffer, usize),
    scales: (&Buffer, usize),
    biases: (&Buffer, usize),
    indices: (&Buffer, usize),
    out: &Buffer,
    num_tokens: usize,
    topk: usize,
    num_experts: usize,
    n: usize,
    k: usize,
    has_bias: bool,
    input_has_topk_dim: bool,
    reuse_topk: bool,
) -> Result<(), MetalKernelError> {
    let name = match (reuse_topk, ty) {
        (true, DType::F16) => "mxfp4_moe_gemm_reuse_f16",
        (true, DType::BF16) => "mxfp4_moe_gemm_reuse_bf16",
        (false, DType::F16) => "mxfp4_moe_gemm_split_f16",
        (false, DType::BF16) => "mxfp4_moe_gemm_split_bf16",
        (_, other) => {
            return Err(MetalKernelError::DTypeMismatch {
                expected: vec![DType::F16, DType::BF16],
                got: other,
            })
        }
    };
    let pipeline = kernels.load_pipeline(device, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    if reuse_topk {
        set_params!(
            encoder,
            (
                x,
                w,
                scales,
                biases,
                indices,
                Output::new(out),
                num_tokens as i32,
                topk as i32,
                num_experts as i32,
                n as i32,
                k as i32,
                has_bias as i32
            )
        );
    } else {
        set_params!(
            encoder,
            (
                x,
                w,
                scales,
                biases,
                indices,
                Output::new(out),
                num_tokens as i32,
                topk as i32,
                num_experts as i32,
                n as i32,
                k as i32,
                has_bias as i32,
                input_has_topk_dim as i32
            )
        );
    }

    let group_dims = MTLSize {
        width: 256,
        height: 1,
        depth: 1,
    };
    let grid_dims = MTLSize {
        width: n.div_ceil(8),
        height: num_tokens,
        depth: if reuse_topk { 1 } else { topk },
    };
    encoder.dispatch_thread_groups(grid_dims, group_dims);
    Ok(())
}
