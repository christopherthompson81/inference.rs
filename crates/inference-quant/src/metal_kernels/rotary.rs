// Portions of this file are adapted from Apple's MLX framework
// (https://github.com/ml-explore/mlx)
// Licensed under the Apache License 2.0
// Copyright © 2023 Apple Inc.

use super::*;

#[allow(clippy::too_many_arguments)]
pub fn call_rotary(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    ty: DType,
    src: &Buffer,
    cos: &Buffer,
    sin: &Buffer,
    src_offset: usize,
    cos_offset: usize,
    sin_offset: usize,
    batch: usize,
    heads: usize,
    seq_len: usize,
    head_dim: usize,
    rot_dim: usize,
    cache_rows: usize,
    is_neox: bool,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    call_rotary_q(
        device, ep, kernels, ty, src, cos, sin, src_offset, cos_offset, sin_offset, batch, heads,
        seq_len, head_dim, rot_dim, cache_rows, is_neox, output,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn call_rotary_q(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    ty: DType,
    q: &Buffer,
    cos: &Buffer,
    sin: &Buffer,
    q_offset: usize,
    cos_offset: usize,
    sin_offset: usize,
    batch: usize,
    q_heads: usize,
    seq_len: usize,
    head_dim: usize,
    rot_dim: usize,
    cache_rows: usize,
    is_neox: bool,
    q_out: &Buffer,
) -> Result<(), MetalKernelError> {
    let name = match ty {
        DType::F32 => "rotary_q_float",
        DType::F16 => "rotary_q_half",
        DType::BF16 => "rotary_q_bfloat",
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
            (q, q_offset),
            (cos, cos_offset),
            (sin, sin_offset),
            Output::new(q_out),
            batch as u32,
            q_heads as u32,
            seq_len as u32,
            head_dim as u32,
            rot_dim as u32,
            cache_rows as u32,
            is_neox
        )
    );

    let n_elements = batch * q_heads * seq_len * head_dim;
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, n_elements);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_rotary_qk(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    ty: DType,
    q: &Buffer,
    k: &Buffer,
    cos: &Buffer,
    sin: &Buffer,
    q_offset: usize,
    k_offset: usize,
    cos_offset: usize,
    sin_offset: usize,
    batch: usize,
    q_heads: usize,
    k_heads: usize,
    seq_len: usize,
    head_dim: usize,
    rot_dim: usize,
    cache_rows: usize,
    is_neox: bool,
    q_out: &Buffer,
    k_out: &Buffer,
) -> Result<(), MetalKernelError> {
    let name = match ty {
        DType::F32 => "rotary_qk_float",
        DType::F16 => "rotary_qk_half",
        DType::BF16 => "rotary_qk_bfloat",
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
            (q, q_offset),
            (k, k_offset),
            (cos, cos_offset),
            (sin, sin_offset),
            Output::new(q_out),
            Output::new(k_out),
            batch as u32,
            q_heads as u32,
            k_heads as u32,
            seq_len as u32,
            head_dim as u32,
            rot_dim as u32,
            cache_rows as u32,
            is_neox
        )
    );

    let n_elements = batch * (q_heads + k_heads) * seq_len * head_dim;
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, n_elements);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_rotary_q_positions(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    ty: DType,
    q: &Buffer,
    cos: &Buffer,
    sin: &Buffer,
    positions: &Buffer,
    q_offset: usize,
    cos_offset: usize,
    sin_offset: usize,
    positions_offset: usize,
    batch: usize,
    q_heads: usize,
    seq_len: usize,
    head_dim: usize,
    rot_dim: usize,
    is_neox: bool,
    q_out: &Buffer,
) -> Result<(), MetalKernelError> {
    let name = match ty {
        DType::F32 => "rotary_q_positions_float",
        DType::F16 => "rotary_q_positions_half",
        DType::BF16 => "rotary_q_positions_bfloat",
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
            (q, q_offset),
            (cos, cos_offset),
            (sin, sin_offset),
            (positions, positions_offset),
            Output::new(q_out),
            batch as u32,
            q_heads as u32,
            seq_len as u32,
            head_dim as u32,
            rot_dim as u32,
            is_neox
        )
    );

    let n_elements = batch * q_heads * seq_len * head_dim;
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, n_elements);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_rotary_qk_positions(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    ty: DType,
    q: &Buffer,
    k: &Buffer,
    cos: &Buffer,
    sin: &Buffer,
    positions: &Buffer,
    q_offset: usize,
    k_offset: usize,
    cos_offset: usize,
    sin_offset: usize,
    positions_offset: usize,
    batch: usize,
    q_heads: usize,
    k_heads: usize,
    seq_len: usize,
    head_dim: usize,
    rot_dim: usize,
    is_neox: bool,
    q_out: &Buffer,
    k_out: &Buffer,
) -> Result<(), MetalKernelError> {
    let name = match ty {
        DType::F32 => "rotary_qk_positions_float",
        DType::F16 => "rotary_qk_positions_half",
        DType::BF16 => "rotary_qk_positions_bfloat",
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
            (q, q_offset),
            (k, k_offset),
            (cos, cos_offset),
            (sin, sin_offset),
            (positions, positions_offset),
            Output::new(q_out),
            Output::new(k_out),
            batch as u32,
            q_heads as u32,
            k_heads as u32,
            seq_len as u32,
            head_dim as u32,
            rot_dim as u32,
            is_neox
        )
    );

    let n_elements = batch * (q_heads + k_heads) * seq_len * head_dim;
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, n_elements);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}
