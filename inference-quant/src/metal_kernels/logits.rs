// Portions of this file are adapted from Apple's MLX framework
// (https://github.com/ml-explore/mlx)
// Licensed under the Apache License 2.0
// Copyright © 2023 Apple Inc.

use super::*;

/// Two-stage top-k + softmax stats over a logits row. `packed_out` is laid
/// out as `[top_values (k), top_indices_as_f32 (k), denom (1), max (1)]`.
/// `input_dtype` selects between the F32 and BF16 stage-1 kernels.
#[allow(clippy::too_many_arguments)]
pub fn call_topk_logits_packed(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    input_dtype: DType,
    input: &Buffer,
    block_values: &Buffer,
    block_indices: &Buffer,
    block_maxes: &Buffer,
    block_sums: &Buffer,
    packed_out: &Buffer,
    ncols: usize,
    k: usize,
    chunk_size: usize,
    inv_temperature: f32,
) -> Result<(), MetalKernelError> {
    let nblocks = ncols.div_ceil(chunk_size);

    let stage1_name = match input_dtype {
        DType::F32 => "topk_logits_stage1_f32",
        DType::BF16 => "topk_logits_stage1_bf16",
        DType::F16 => "topk_logits_stage1_f16",
        other => {
            return Err(MetalKernelError::DTypeMismatch {
                expected: vec![DType::F32, DType::F16, DType::BF16],
                got: other,
            })
        }
    };
    let stage1 = kernels.load_pipeline(device, stage1_name)?;
    let stage2 = kernels.load_pipeline(device, "topk_logits_stage2_packed_f32")?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();

    encoder.set_compute_pipeline_state(&stage1);
    encoder.set_input_buffer(0, Some(input), 0);
    encoder.set_output_buffer(1, Some(block_values), 0);
    encoder.set_output_buffer(2, Some(block_indices), 0);
    encoder.set_output_buffer(3, Some(block_maxes), 0);
    encoder.set_output_buffer(4, Some(block_sums), 0);
    <i32 as EncoderParam>::set_param(encoder, 5, ncols as i32);
    <i32 as EncoderParam>::set_param(encoder, 6, k as i32);
    <i32 as EncoderParam>::set_param(encoder, 7, chunk_size as i32);
    <f32 as EncoderParam>::set_param(encoder, 8, inv_temperature);
    encoder.set_threadgroup_memory_length(0, chunk_size);
    let group = MTLSize {
        width: 1024,
        height: 1,
        depth: 1,
    };
    let grid = MTLSize {
        width: nblocks,
        height: 1,
        depth: 1,
    };
    encoder.dispatch_thread_groups(grid, group);

    encoder.set_compute_pipeline_state(&stage2);
    encoder.set_input_buffer(0, Some(block_values), 0);
    encoder.set_input_buffer(1, Some(block_indices), 0);
    encoder.set_input_buffer(2, Some(block_maxes), 0);
    encoder.set_input_buffer(3, Some(block_sums), 0);
    encoder.set_output_buffer(4, Some(packed_out), 0);
    <i32 as EncoderParam>::set_param(encoder, 5, nblocks as i32);
    <i32 as EncoderParam>::set_param(encoder, 6, k as i32);
    encoder.set_threadgroup_memory_length(0, (nblocks * k).max(1));
    encoder.dispatch_thread_groups(
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        group,
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_copy_logits(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: DType,
    src: &Buffer,
    src_offset: usize,
    dst: &Buffer,
    n: usize,
) -> Result<(), MetalKernelError> {
    let name = match dtype {
        DType::F32 => "copy_f32",
        DType::BF16 => "copy_bf16",
        DType::F16 => "copy_f16",
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
    encoder.set_input_buffer(0, Some(src), src_offset);
    encoder.set_output_buffer(1, Some(dst), 0);
    <i32 as EncoderParam>::set_param(encoder, 2, n as i32);
    let (groups, gsize) = linear_split(&pipeline, n);
    encoder.dispatch_thread_groups(groups, gsize);
    Ok(())
}

/// In-place sparse penalties: each (token_id, count) pair updates one logit.
#[allow(clippy::too_many_arguments)]
pub fn call_apply_sparse_penalties(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: DType,
    logits: &Buffer,
    token_ids: &Buffer,
    counts: &Buffer,
    n: usize,
    n_tokens: usize,
    frequency_penalty: f32,
    presence_penalty: f32,
    repetition_penalty: f32,
) -> Result<(), MetalKernelError> {
    if n_tokens == 0 {
        return Ok(());
    }
    let name = match dtype {
        DType::F32 => "apply_sparse_penalties_f32",
        DType::BF16 => "apply_sparse_penalties_bf16",
        DType::F16 => "apply_sparse_penalties_f16",
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
    encoder.set_input_buffer(0, Some(logits), 0);
    encoder.set_input_buffer(1, Some(token_ids), 0);
    encoder.set_input_buffer(2, Some(counts), 0);
    <i32 as EncoderParam>::set_param(encoder, 3, n as i32);
    <i32 as EncoderParam>::set_param(encoder, 4, n_tokens as i32);
    <f32 as EncoderParam>::set_param(encoder, 5, frequency_penalty);
    <f32 as EncoderParam>::set_param(encoder, 6, presence_penalty);
    <f32 as EncoderParam>::set_param(encoder, 7, repetition_penalty);

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, n_tokens);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}
