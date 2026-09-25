// Portions of this file are adapted from Apple's MLX framework
// (https://github.com/ml-explore/mlx)
// Licensed under the Apache License 2.0
// Copyright © 2023 Apple Inc.

use super::*;

/// Fused topk-weighted reduce: collapses
/// `[num_tokens * topk, hidden]` (per-assignment outputs in row-contiguous
/// (token, slot) order) into `[num_tokens, hidden]`, with each slot scaled by
/// `topk_weights[token*topk + slot]`. Accumulation is in fp32. Replaces the
/// `reshape -> to_dtype(F32) -> broadcast_mul -> sum -> to_dtype` tail.
#[allow(clippy::too_many_arguments)]
pub fn call_moe_weighted_reduce_flat(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    ty: DType,
    inputs: &Buffer,
    inputs_offset: usize,
    topk_weights: &Buffer,
    topk_weights_offset: usize,
    out: &Buffer,
    num_tokens: usize,
    hidden: usize,
    topk: usize,
) -> Result<(), MetalKernelError> {
    let type_string = match ty {
        DType::F32 => "float",
        DType::BF16 => "bfloat",
        DType::F16 => "half",
        other => {
            return Err(MetalKernelError::DTypeMismatch {
                expected: vec![DType::F32, DType::F16, DType::BF16],
                got: other,
            })
        }
    };
    let name = format!("moe_weighted_reduce_flat_{type_string}");
    let pipeline = kernels.load_pipeline(device, &name)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_input_buffer(0, Some(inputs), inputs_offset);
    encoder.set_input_buffer(1, Some(topk_weights), topk_weights_offset);
    encoder.set_output_buffer(2, Some(out), 0);
    <i32 as EncoderParam>::set_param(encoder, 3, num_tokens as i32);
    <i32 as EncoderParam>::set_param(encoder, 4, hidden as i32);
    <i32 as EncoderParam>::set_param(encoder, 5, topk as i32);

    // 2D grid over (hidden, num_tokens). Each thread handles one output cell.
    // Threadgroup chosen to keep h-dim wide for coalesced writes.
    let tg_x: usize = 64;
    let tg_y: usize = 4;
    let group_dims = MTLSize {
        width: tg_x,
        height: tg_y,
        depth: 1,
    };
    let grid_dims = MTLSize {
        width: hidden.div_ceil(tg_x) * tg_x,
        height: num_tokens.div_ceil(tg_y) * tg_y,
        depth: 1,
    };
    encoder.dispatch_thread_groups(
        MTLSize {
            width: grid_dims.width / tg_x,
            height: grid_dims.height / tg_y,
            depth: 1,
        },
        group_dims,
    );
    Ok(())
}
