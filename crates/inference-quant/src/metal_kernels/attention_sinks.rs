// Portions of this file are adapted from Apple's MLX framework
// (https://github.com/ml-explore/mlx)
// Licensed under the Apache License 2.0
// Copyright © 2023 Apple Inc.

use super::*;

// ============================================================================
// Softmax with sinks kernel
// ============================================================================

#[allow(clippy::too_many_arguments)]
pub fn call_softmax_with_sinks(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    ty: DType,
    logits: &Buffer,
    logits_offset: usize,
    sinks: &Buffer,
    sinks_offset: usize,
    output: &Buffer,
    num_heads: u32,
    q_len: u32,
    k_len: u32,
    total_rows: usize,
) -> Result<(), MetalKernelError> {
    let name = match ty {
        DType::F32 => "softmax_with_sinks_float",
        DType::F16 => "softmax_with_sinks_half",
        DType::BF16 => "softmax_with_sinks_bfloat",
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

    // Choose thread group size based on k_len
    let threads_per_group: usize = if k_len <= 64 {
        64
    } else if k_len <= 128 {
        128
    } else if k_len <= 256 {
        256
    } else {
        512
    };

    // Shared memory: s_max(1) + s_sum(1) + warp_scratch(threads_per_group / 32)
    let num_simdgroups = threads_per_group.div_ceil(32);
    let shared_mem_size = (2 + num_simdgroups) * std::mem::size_of::<f32>();
    encoder.set_threadgroup_memory_length(0, shared_mem_size);

    set_params!(
        encoder,
        (
            (logits, logits_offset),
            (sinks, sinks_offset),
            Output::new(output),
            num_heads,
            q_len,
            k_len
        )
    );

    let thread_groups_count = MTLSize {
        width: total_rows,
        height: 1,
        depth: 1,
    };
    let thread_group_size = MTLSize {
        width: threads_per_group,
        height: 1,
        depth: 1,
    };
    encoder.dispatch_thread_groups(thread_groups_count, thread_group_size);
    Ok(())
}

// ============================================================================
// SDPA with sinks (fused attention) kernels
// ============================================================================

fn sdpa_with_sinks_dtype_name(ty: DType) -> Result<&'static str, MetalKernelError> {
    match ty {
        DType::F32 => Ok("float"),
        DType::F16 => Ok("half"),
        DType::BF16 => Ok("bfloat16_t"),
        other => Err(MetalKernelError::DTypeMismatch {
            expected: vec![DType::F32, DType::F16, DType::BF16],
            got: other,
        }),
    }
}

/// Fused SDPA with sinks for decode (q_len == 1, single-pass).
/// Dispatches `sdpa_vector_with_sinks` Metal kernel.
#[allow(clippy::too_many_arguments)]
pub fn call_sdpa_vector_with_sinks(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    ty: DType,
    q_buffer: &Buffer,
    q_offset: usize,
    k_buffer: &Buffer,
    k_offset: usize,
    v_buffer: &Buffer,
    v_offset: usize,
    sinks_buffer: &Buffer,
    sinks_offset: usize,
    output: &Buffer,
    head_dim: usize,
    gqa_factor: i32,
    n: i32, // k_len
    k_stride: usize,
    v_stride: usize,
    scale: f32,
    b: usize, // batch * num_heads
) -> Result<(), MetalKernelError> {
    let type_name = sdpa_with_sinks_dtype_name(ty)?;
    let name = format!("sdpa_vector_with_sinks_{type_name}_{head_dim}");

    let pipeline = kernels.load_pipeline(device, &name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    let k_stride = k_stride as u64;
    let v_stride = v_stride as u64;

    set_params!(
        encoder,
        (
            (q_buffer, q_offset),
            (k_buffer, k_offset),
            (v_buffer, v_offset),
            (sinks_buffer, sinks_offset),
            Output::new(output),
            gqa_factor,
            n,
            k_stride,
            v_stride,
            scale
        )
    );

    let grid_dims = MTLSize {
        width: 1,
        height: b,
        depth: 1,
    };
    let group_dims = MTLSize {
        width: 1024, // 32 simdgroups * 32 threads
        height: 1,
        depth: 1,
    };
    encoder.dispatch_thread_groups(grid_dims, group_dims);
    Ok(())
}

/// Fused SDPA with sinks for decode, two-pass variant (k_len >= 1024).
#[allow(clippy::too_many_arguments)]
pub fn call_sdpa_vector_with_sinks_2pass(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    ty: DType,
    q_buffer: &Buffer,
    q_offset: usize,
    k_buffer: &Buffer,
    k_offset: usize,
    v_buffer: &Buffer,
    v_offset: usize,
    sinks_buffer: &Buffer,
    sinks_offset: usize,
    output: &Buffer,
    intermediate: &Buffer,
    sums: &Buffer,
    maxs: &Buffer,
    head_dim: usize,
    gqa_factor: i32,
    n: i32, // k_len
    k_stride: usize,
    v_stride: usize,
    scale: f32,
    b: usize, // batch * num_heads
) -> Result<(), MetalKernelError> {
    let type_name = sdpa_with_sinks_dtype_name(ty)?;
    let blocks: u64 = 32;

    // Pass 1: compute partial outputs per block
    {
        let name = format!("sdpa_vector_with_sinks_2pass_1_{type_name}_{head_dim}");
        let pipeline = kernels.load_pipeline(device, &name)?;

        let encoder = ep.encoder();
        let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);

        let k_stride = k_stride as u64;
        let v_stride = v_stride as u64;

        set_params!(
            encoder,
            (
                (q_buffer, q_offset),
                (k_buffer, k_offset),
                (v_buffer, v_offset),
                Output::new(intermediate),
                Output::new(sums),
                Output::new(maxs),
                gqa_factor,
                n,
                k_stride,
                v_stride,
                scale
            )
        );

        let grid_dims = MTLSize {
            width: 1,
            height: b,
            depth: blocks as usize,
        };
        let group_dims = MTLSize {
            width: 8 * 32, // BN=8 simdgroups * 32 threads
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(grid_dims, group_dims);
    }

    // Pass 2: reduce across blocks, integrate sinks
    {
        let name = format!("sdpa_vector_with_sinks_2pass_2_{type_name}_{head_dim}");
        let pipeline = kernels.load_pipeline(device, &name)?;

        let encoder = ep.encoder();
        let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);

        set_params!(
            encoder,
            (
                intermediate,
                sums,
                maxs,
                (sinks_buffer, sinks_offset),
                Output::new(output)
            )
        );

        let grid_dims = MTLSize {
            width: 1,
            height: b,
            depth: 1,
        };
        let group_dims = MTLSize {
            width: 1024, // 32 * 32
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(grid_dims, group_dims);
    }

    Ok(())
}

/// Fused flash attention with sinks for prefill (q_len > 1).
/// Dispatches `flash_attn_sinks_kernel` Metal kernel.
#[allow(clippy::too_many_arguments)]
pub fn call_flash_attn_sinks_prefill(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    ty: DType,
    q_buffer: &Buffer,
    q_offset: usize,
    k_buffer: &Buffer,
    k_offset: usize,
    v_buffer: &Buffer,
    v_offset: usize,
    sinks_buffer: &Buffer,
    sinks_offset: usize,
    output: &Buffer,
    scale: f32,
    batch_size: usize,
    q_len: usize,
    k_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    window_size: usize,
) -> Result<(), MetalKernelError> {
    let type_name = sdpa_with_sinks_dtype_name(ty)?;

    let br: usize = 8; // simdgroups per threadgroup
    let bc: usize = match head_dim {
        64 => 64,
        80 | 96 | 112 | 128 => 32,
        192 | 256 => 16,
        _ => {
            return Err(MetalKernelError::CompilationError(format!(
                "flash_attn_sinks: unsupported head_dim={head_dim}"
            )))
        }
    };

    let name = format!("flash_attn_sinks_{type_name}_hd{head_dim}_br{br}_bc{bc}");
    let pipeline = kernels.load_pipeline(device, &name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    // Shared memory: k_smem[BC * D_PAD] + v_smem[BC * D_PAD] in float32
    let d_pad = head_dim.div_ceil(32) * 32;
    let shared_mem_size = 2 * bc * d_pad * std::mem::size_of::<f32>();
    encoder.set_threadgroup_memory_length(0, shared_mem_size);

    let q_len_i32 = q_len as i32;
    let k_len_i32 = k_len as i32;
    let num_heads_i32 = num_heads as i32;
    let num_kv_heads_i32 = num_kv_heads as i32;
    let window_size_i32 = window_size as i32;

    set_params!(
        encoder,
        (
            (q_buffer, q_offset),
            (k_buffer, k_offset),
            (v_buffer, v_offset),
            (sinks_buffer, sinks_offset),
            Output::new(output),
            scale,
            q_len_i32,
            k_len_i32,
            num_heads_i32,
            num_kv_heads_i32,
            window_size_i32
        )
    );

    let grid_dims = MTLSize {
        width: num_heads,
        height: batch_size,
        depth: q_len.div_ceil(br),
    };
    let group_dims = MTLSize {
        width: br * 32,
        height: 1,
        depth: 1,
    };
    encoder.dispatch_thread_groups(grid_dims, group_dims);
    Ok(())
}

/// Dispatches `flash_attn_sinks_varlen_kernel` Metal kernel.
#[allow(clippy::too_many_arguments)]
pub fn call_flash_attn_sinks_varlen_prefill(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    ty: DType,
    q_buffer: &Buffer,
    q_offset: usize,
    k_buffer: &Buffer,
    k_offset: usize,
    v_buffer: &Buffer,
    v_offset: usize,
    sinks_buffer: &Buffer,
    sinks_offset: usize,
    output: &Buffer,
    cu_seqlens_q_buffer: &Buffer,
    cu_seqlens_q_offset: usize,
    cu_seqlens_k_buffer: &Buffer,
    cu_seqlens_k_offset: usize,
    scale: f32,
    batch_size: usize,
    max_q_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    window_size: usize,
) -> Result<(), MetalKernelError> {
    let type_name = sdpa_with_sinks_dtype_name(ty)?;

    let br: usize = 8;
    let bc: usize = match head_dim {
        64 => 64,
        80 | 96 | 112 | 128 => 32,
        192 | 256 => 16,
        _ => {
            return Err(MetalKernelError::CompilationError(format!(
                "flash_attn_sinks_varlen: unsupported head_dim={head_dim}"
            )))
        }
    };

    let name = format!("flash_attn_sinks_varlen_{type_name}_hd{head_dim}_br{br}_bc{bc}");
    let pipeline = kernels.load_pipeline(device, &name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    let d_pad = head_dim.div_ceil(32) * 32;
    let shared_mem_size = 2 * bc * d_pad * std::mem::size_of::<f32>();
    encoder.set_threadgroup_memory_length(0, shared_mem_size);

    let max_q_len_i32 = max_q_len as i32;
    let num_heads_i32 = num_heads as i32;
    let num_kv_heads_i32 = num_kv_heads as i32;
    let window_size_i32 = window_size as i32;

    set_params!(
        encoder,
        (
            (q_buffer, q_offset),
            (k_buffer, k_offset),
            (v_buffer, v_offset),
            (sinks_buffer, sinks_offset),
            Output::new(output),
            (cu_seqlens_q_buffer, cu_seqlens_q_offset),
            (cu_seqlens_k_buffer, cu_seqlens_k_offset),
            scale,
            max_q_len_i32,
            num_heads_i32,
            num_kv_heads_i32,
            window_size_i32
        )
    );

    let grid_dims = MTLSize {
        width: num_heads,
        height: batch_size,
        depth: max_q_len.div_ceil(br),
    };
    let group_dims = MTLSize {
        width: br * 32,
        height: 1,
        depth: 1,
    };
    encoder.dispatch_thread_groups(grid_dims, group_dims);
    Ok(())
}
