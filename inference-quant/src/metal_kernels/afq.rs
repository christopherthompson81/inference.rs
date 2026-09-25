// Portions of this file are adapted from Apple's MLX framework
// (https://github.com/ml-explore/mlx)
// Licensed under the Apache License 2.0
// Copyright © 2023 Apple Inc.

use super::*;

#[allow(clippy::too_many_arguments)]
pub fn call_affine_quantize(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    full_ty: DType,
    input: &Buffer,
    input_offset: usize,
    input_dims: &[usize],
    input_strides: &[usize],
    output: &Buffer,
    output_dims: &[usize],
    scales: &Buffer,
    biases: &Buffer,
    dequantize: bool,
    group_size: usize,
    bits: usize,
) -> Result<(), MetalKernelError> {
    let type_string = match full_ty {
        DType::F32 => "float",
        DType::BF16 => "bfloat16_t",
        DType::F16 => "float16_t",
        other => {
            return Err(MetalKernelError::DTypeMismatch {
                expected: vec![DType::F32, DType::F16, DType::BF16],
                got: other,
            })
        }
    };
    let kernel_func = if dequantize {
        "affine_dequantize"
    } else {
        "affine_quantize"
    };
    let name = format!("{kernel_func}_{type_string}_gs_{group_size}_b_{bits}");

    let pipeline = kernels.load_pipeline(device, &name)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    // Treat uint32 as uint8 in kernel
    let uint8_per_uint32 = 4;
    let simd_size = 32;
    let packs_per_int = match bits {
        3 => 8,
        6 => 4,
        40 => 2, // mxfp4: 2 FP4 values per byte
        _ => 8 / bits,
    };
    let per_thread = if dequantize {
        packs_per_int
    } else {
        group_size / simd_size
    };
    let nthreads = if dequantize {
        output_dims.iter().product::<usize>() / packs_per_int
    } else {
        input_dims.iter().product::<usize>() / per_thread
    };

    let thread_group_size = (pipeline.max_total_threads_per_threadgroup() as usize).min(nthreads);
    let group_dims = MTLSize {
        width: thread_group_size as usize,
        height: 1,
        depth: 1,
    };
    let use_2d = nthreads > u32::MAX as usize;
    let mut grid_shape = input_dims.to_vec();
    if dequantize {
        *grid_shape.last_mut().unwrap() *= uint8_per_uint32;
    } else {
        *grid_shape.last_mut().unwrap() /= per_thread;
    }
    let grid_dims = if use_2d {
        get_2d_grid_dims(&grid_shape, input_strides)
    } else {
        MTLSize {
            width: nthreads,
            height: 1,
            depth: 1,
        }
    };

    if dequantize {
        set_params!(
            encoder,
            ((input, input_offset), scales, biases, Output::new(output))
        );
    } else {
        set_params!(
            encoder,
            (
                (input, input_offset),
                Output::new(output),
                Output::new(scales),
                Output::new(biases)
            )
        );
    }

    encoder.dispatch_threads(grid_dims, group_dims);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_afq_embedding(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    ty: DType,
    w: &Buffer,
    w_offset: usize,
    scales: &Buffer,
    scales_offset: usize,
    biases: &Buffer,
    biases_offset: usize,
    ids: &Buffer,
    ids_offset: usize,
    out: &Buffer,
    num_ids: usize,
    hidden_size: usize,
    bits: usize,
    group_size: usize,
) -> Result<(), MetalKernelError> {
    let type_string = match ty {
        DType::F32 => "float",
        DType::BF16 => "bfloat16_t",
        DType::F16 => "float16_t",
        other => {
            return Err(MetalKernelError::DTypeMismatch {
                expected: vec![DType::F32, DType::F16, DType::BF16],
                got: other,
            })
        }
    };
    let name = format!("affine_embedding_{type_string}_gs_{group_size}_b_{bits}");
    let pipeline = kernels.load_pipeline(device, &name)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    let pack_factor = match bits {
        3 => 8,
        6 => 4,
        40 => 2,
        _ => 8 / bits,
    };
    let nthreads = num_ids * hidden_size / pack_factor;
    if nthreads == 0 {
        return Ok(());
    }
    let thread_group_size = (pipeline.max_total_threads_per_threadgroup() as usize).min(nthreads);
    let group_dims = MTLSize {
        width: thread_group_size,
        height: 1,
        depth: 1,
    };
    let grid_dims = MTLSize {
        width: nthreads,
        height: 1,
        depth: 1,
    };
    let num_ids = num_ids as i32;
    let hidden_size = hidden_size as i32;

    set_params!(
        encoder,
        (
            (w, w_offset),
            (scales, scales_offset),
            (biases, biases_offset),
            (ids, ids_offset),
            Output::new(out),
            num_ids,
            hidden_size
        )
    );

    encoder.dispatch_threads(grid_dims, group_dims);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_afq_qmm(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    ty: DType,
    x: &Buffer,
    x_offset: usize,
    x_shape: &[usize],
    x_stride: &[usize],
    w: &Buffer,
    w_shape: &[usize],
    w_stride: &[usize],
    scales: &Buffer,
    s_stride: &[usize],
    biases: &Buffer,
    b_stride: &[usize],
    out: &Buffer,
    out_shape: &[usize],
    gather_lhs_rhs_indices: Option<(&Buffer, &Buffer)>,
    gather_lhs_shape: Option<&[usize]>,
    gather_lhs_rhs_strides: Option<(&[usize], &[usize])>,
    transpose: bool,
    bits: usize,
    group_size: usize,
) -> Result<(), MetalKernelError> {
    let gather = gather_lhs_rhs_indices.is_some();

    let batched = !gather && w_shape.len() > 2;

    let d = x_shape[x_shape.len() - 1];
    let o = out_shape[out_shape.len() - 1];
    // For the unbatched W case, avoid `adjust_matrix_offsets`
    // for a small performance gain.
    let b = if batched || gather {
        x_shape[x_shape.len() - 2]
    } else {
        x_shape.iter().product::<usize>() / d
    };
    let n = if batched || gather {
        out_shape.iter().product::<usize>() / b / o
    } else {
        1
    };

    let mut name = if gather {
        "bs_".to_string()
    } else {
        "".to_string()
    };
    let mut matrix = false;
    let mut aligned = false;
    let mut quad = false;

    // Batch-size cutoff for qmv* vs qmm_t dispatch at small M.
    let qmv_limit: usize = if transpose {
        match device.device_type() {
            MetalDeviceType::Ultra => {
                if d <= 2048 && o <= 2048 {
                    32
                } else if d <= 4096 && o <= 4096 {
                    18
                } else {
                    12
                }
            }
            _ => {
                if d <= 2048 && o <= 2048 {
                    18
                } else if d <= 4096 && o <= 4096 {
                    12
                } else {
                    10
                }
            }
        }
    } else {
        4
    };

    let (group_dims, grid_dims) = if transpose {
        if b < qmv_limit && (d == 128 || d == 64) && bits.is_power_of_two() {
            name.push_str("qmv_quad");
            let quads_per_simd = 8;
            let results_per_simdgroup = 8;
            let bo = quads_per_simd * results_per_simdgroup;
            let simdgroup_size = 32;
            quad = true;
            let group_dims = MTLSize {
                width: simdgroup_size as usize,
                height: 1,
                depth: 1,
            };
            let grid_dims = MTLSize {
                width: o.div_ceil(bo),
                height: b,
                depth: n,
            };
            (group_dims, grid_dims)
        } else if b < qmv_limit && o.is_multiple_of(8) && d.is_multiple_of(512) && d >= 512 {
            name.push_str("qmv_fast");
            let bo = 8;
            let bd = 32;
            let group_dims = MTLSize {
                width: bd,
                height: 2,
                depth: 1,
            };
            let grid_dims = MTLSize {
                width: (o / bo),
                height: b,
                depth: n,
            };
            (group_dims, grid_dims)
        } else if b < qmv_limit {
            name.push_str("qmv");
            let bo = 8;
            let bd = 32;
            let group_dims = MTLSize {
                width: bd,
                height: 2,
                depth: 1,
            };
            let grid_dims = MTLSize {
                width: o.div_ceil(bo),
                height: b,
                depth: n,
            };
            (group_dims, grid_dims)
        } else {
            // Prefer the BM=64 BN=32 tile when prefill is large enough.
            let wn = 2;
            let wm = 2;
            let use_tile_64_32 = b >= 64 && o.is_multiple_of(32);
            let use_tile_64_64 = b >= 64 && o.is_multiple_of(64);
            let (bm, bn) = if use_tile_64_32 {
                (64, 32)
            } else if use_tile_64_64 {
                (64, 64)
            } else {
                (32, 32)
            };
            name.push_str("qmm_t");
            let group_dims = MTLSize {
                width: 32,
                height: wn as usize,
                depth: wm as usize,
            };
            let grid_dims = MTLSize {
                width: o.div_ceil(bn),
                height: b.div_ceil(bm),
                depth: n,
            };
            matrix = true;
            aligned = true;
            (group_dims, grid_dims)
        }
    } else {
        /*if b < 4 && d >= 1024 {
            todo!("qvm_split_k");
        } else */
        if b < qmv_limit {
            name.push_str("qvm");
            let bo = 64;
            let bd = 32;
            let group_dims = MTLSize {
                width: bd,
                height: 2,
                depth: 1,
            };
            let grid_dims = MTLSize {
                width: (o / bo),
                height: b,
                depth: n,
            };
            (group_dims, grid_dims)
        } else {
            name.push_str("qmm_n");
            let wn = 2;
            let wm = 2;
            let bm = 32;
            let bn = 32;
            let group_dims = MTLSize {
                width: 32,
                height: wn as usize,
                depth: wm as usize,
            };
            let grid_dims = MTLSize {
                width: (o / bn),
                height: b.div_ceil(bm),
                depth: n,
            };
            matrix = true;
            if !o.is_multiple_of(bn) {
                panic!("output size should be divisible by {bn} but received {o}.");
            }
            (group_dims, grid_dims)
        }
    };

    let aligned_n = if o.is_multiple_of(32) {
        "true"
    } else {
        "false"
    };

    let type_string = match ty {
        DType::F32 => "float",
        DType::BF16 => "bfloat16_t",
        DType::F16 => "float16_t",
        other => {
            return Err(MetalKernelError::DTypeMismatch {
                expected: vec![DType::F32, DType::F16, DType::BF16],
                got: other,
            })
        }
    };

    name = format!("{name}_{type_string}_gs_{group_size}_b_{bits}");
    if quad {
        name.push_str(&format!("_d_{d}"));
    }
    if aligned {
        name.push_str(&format!("_alN_{aligned_n}"));
    }
    if !gather {
        name.push_str(&format!("_batch_{}", batched as usize));
    }
    if matrix && aligned && !batched && !gather && transpose {
        if b >= 64 && o.is_multiple_of(32) {
            name.push_str("_t_64_32_32");
        } else if b >= 64 && o.is_multiple_of(64) {
            name.push_str("_t_64_64_32");
        }
    }

    let pipeline = kernels.load_pipeline(device, &name)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    encoder.set_input_buffer(0, Some(w), 0);
    encoder.set_input_buffer(1, Some(scales), 0);
    encoder.set_input_buffer(2, Some(biases), 0);
    encoder.set_input_buffer(3, Some(x), x_offset);
    encoder.set_output_buffer(4, Some(out), 0);
    <i32 as EncoderParam>::set_param(encoder, 5, d as i32);
    <i32 as EncoderParam>::set_param(encoder, 6, o as i32);

    let mut offset = 7;
    if matrix {
        <i32 as EncoderParam>::set_param(encoder, 7, b as i32);
        offset += 1;
    }

    let x_batch_ndims = x_shape.len() - 2;
    let w_batch_ndims = w_shape.len() - 2;

    if batched || gather {
        <i32 as EncoderParam>::set_param(encoder, offset, x_batch_ndims as i32);
        <&[i32] as EncoderParam>::set_param(
            encoder,
            offset + 1,
            &x_shape.iter().map(|x| *x as i32).collect::<Vec<_>>(),
        );
        <&[i64] as EncoderParam>::set_param(
            encoder,
            offset + 2,
            &x_stride.iter().map(|x| *x as i64).collect::<Vec<_>>(),
        );
        <i32 as EncoderParam>::set_param(encoder, offset + 3, w_batch_ndims as i32);
        <&[i32] as EncoderParam>::set_param(
            encoder,
            offset + 4,
            &w_shape.iter().map(|x| *x as i32).collect::<Vec<_>>(),
        );
        <&[i64] as EncoderParam>::set_param(
            encoder,
            offset + 5,
            &w_stride.iter().map(|x| *x as i64).collect::<Vec<_>>(),
        );
        <&[i64] as EncoderParam>::set_param(
            encoder,
            offset + 6,
            &s_stride.iter().map(|x| *x as i64).collect::<Vec<_>>(),
        );
        <&[i64] as EncoderParam>::set_param(
            encoder,
            offset + 7,
            &b_stride.iter().map(|x| *x as i64).collect::<Vec<_>>(),
        );
    }
    if gather {
        let (lhs_indices, rhs_indices) = gather_lhs_rhs_indices.unwrap();
        let batch_shape = gather_lhs_shape.unwrap();
        let batch_ndims = batch_shape.len();
        let (lhs_strides, rhs_strides) = gather_lhs_rhs_strides.unwrap();

        <i32 as EncoderParam>::set_param(encoder, offset + 8, batch_ndims as i32);
        <&[i32] as EncoderParam>::set_param(
            encoder,
            offset + 9,
            &batch_shape.iter().map(|x| *x as i32).collect::<Vec<_>>(),
        );
        encoder.set_input_buffer(offset + 10, Some(lhs_indices), 0);
        encoder.set_input_buffer(offset + 11, Some(rhs_indices), 0);
        <&[i64] as EncoderParam>::set_param(
            encoder,
            offset + 12,
            &lhs_strides.iter().map(|x| *x as i64).collect::<Vec<_>>(),
        );
        <&[i64] as EncoderParam>::set_param(
            encoder,
            offset + 13,
            &rhs_strides.iter().map(|x| *x as i64).collect::<Vec<_>>(),
        );
    }

    encoder.dispatch_thread_groups(grid_dims, group_dims);
    Ok(())
}

/// Split-K AFQ qmm_t. Writes `[split_k, m, n]` to `out`; caller sums over
/// the leading split_k dim to get the final `[m, n]`.
#[allow(clippy::too_many_arguments)]
pub fn call_afq_qmm_splitk(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    ty: DType,
    x: &Buffer,
    x_offset: usize,
    w: &Buffer,
    scales: &Buffer,
    biases: &Buffer,
    out: &Buffer,
    m: usize,
    n: usize,
    k: usize,
    split_k: usize,
    bits: usize,
    group_size: usize,
) -> Result<(), MetalKernelError> {
    assert!(split_k >= 2, "call_afq_qmm_splitk requires split_k >= 2");
    assert_eq!(
        k % (split_k * group_size),
        0,
        "K ({k}) must be divisible by split_k * group_size ({})",
        split_k * group_size
    );
    let k_partition_size = k / split_k;
    let split_k_partition_stride = (m * n) as i32;

    let type_string = match ty {
        DType::F32 => "float",
        DType::BF16 => "bfloat16_t",
        DType::F16 => "float16_t",
        other => {
            return Err(MetalKernelError::DTypeMismatch {
                expected: vec![DType::F32, DType::F16, DType::BF16],
                got: other,
            })
        }
    };
    let aligned = if n.is_multiple_of(32) {
        "true"
    } else {
        "false"
    };
    let name = format!("qmm_t_splitk_{type_string}_gs_{group_size}_b_{bits}_alN_{aligned}");
    let pipeline = kernels.load_pipeline(device, &name)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    encoder.set_input_buffer(0, Some(w), 0);
    encoder.set_input_buffer(1, Some(scales), 0);
    encoder.set_input_buffer(2, Some(biases), 0);
    encoder.set_input_buffer(3, Some(x), x_offset);
    encoder.set_output_buffer(4, Some(out), 0);
    <i32 as EncoderParam>::set_param(encoder, 5, k as i32);
    <i32 as EncoderParam>::set_param(encoder, 6, n as i32);
    <i32 as EncoderParam>::set_param(encoder, 7, m as i32);
    <i32 as EncoderParam>::set_param(encoder, 8, k_partition_size as i32);
    <i32 as EncoderParam>::set_param(encoder, 9, split_k_partition_stride);

    let group_dims = MTLSize {
        width: 32,
        height: 2,
        depth: 2,
    };
    let grid_dims = MTLSize {
        width: n.div_ceil(32),
        height: m.div_ceil(32),
        depth: split_k,
    };
    encoder.dispatch_thread_groups(grid_dims, group_dims);
    Ok(())
}

// Tile presets for the MLX-ported sorted-MoE gather GEMM. Must match the
// instantiations in quantized.metal: (BM, BN, BK, WM, WN).
const AFQ_GATHER_RHS_BN: usize = 32;

const AFQ_GATHER_RHS_BK: usize = 32;

fn pick_afq_gather_rhs_tile(m: usize) -> (usize, usize, usize, usize, usize) {
    // BM=32 is a sweet spot on M3-class Apple GPUs: BM=64 regresses on
    // 26B-A4B-class MoE prefill (register pressure / occupancy), BM=16 leaves
    // arithmetic intensity on the table for large M.
    if m >= 128 {
        (32, AFQ_GATHER_RHS_BN, AFQ_GATHER_RHS_BK, 1, 2)
    } else {
        (16, AFQ_GATHER_RHS_BN, AFQ_GATHER_RHS_BK, 1, 2)
    }
}

/// Sorted-MoE tiled grouped GEMM (ported from MLX `affine_gather_qmm_rhs`).
/// Caller must pre-sort `x` and `indices` by expert id; rows for the same
/// expert must be contiguous. Output `y` has the same row order as `x` and
/// must be unsorted by the caller using the inverse permutation.
#[allow(clippy::too_many_arguments)]
pub fn call_afq_gather_qmm_rhs(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    ty: DType,
    x: &Buffer,
    x_offset: usize,
    w: &Buffer,
    scales: &Buffer,
    biases: &Buffer,
    indices: &Buffer,
    out: &Buffer,
    m: usize,
    n: usize,
    k: usize,
    bits: usize,
    group_size: usize,
) -> Result<(), MetalKernelError> {
    let type_string = match ty {
        DType::F32 => "float",
        DType::BF16 => "bfloat16_t",
        DType::F16 => "float16_t",
        other => {
            return Err(MetalKernelError::DTypeMismatch {
                expected: vec![DType::F32, DType::F16, DType::BF16],
                got: other,
            })
        }
    };

    let (bm, bn, bk, wm, wn) = pick_afq_gather_rhs_tile(m);
    let align_m = m.is_multiple_of(bm);
    let align_n = n.is_multiple_of(bn);
    let align_k = k.is_multiple_of(bk);
    let am = if align_m { "t" } else { "n" };
    let an = if align_n { "t" } else { "n" };
    let ak = if align_k { "t" } else { "n" };
    let name = format!(
        "affine_gather_qmm_rhs_{type_string}_gs_{group_size}_b_{bits}_bm_{bm}_bn_{bn}_bk_{bk}_wm_{wm}_wn_{wn}_t_true_alM_{am}_alN_{an}_alK_{ak}",
    );

    let pipeline = kernels.load_pipeline(device, &name)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    encoder.set_input_buffer(0, Some(x), x_offset);
    encoder.set_input_buffer(1, Some(w), 0);
    encoder.set_input_buffer(2, Some(scales), 0);
    encoder.set_input_buffer(3, Some(biases), 0);
    encoder.set_input_buffer(4, Some(indices), 0);
    encoder.set_output_buffer(5, Some(out), 0);
    <i32 as EncoderParam>::set_param(encoder, 6, m as i32);
    <i32 as EncoderParam>::set_param(encoder, 7, n as i32);
    <i32 as EncoderParam>::set_param(encoder, 8, k as i32);

    let group_dims = MTLSize {
        width: 32,
        height: wn,
        depth: wm,
    };
    let grid_dims = MTLSize {
        width: n.div_ceil(bn),
        height: m.div_ceil(bm),
        depth: 1,
    };
    encoder.dispatch_thread_groups(grid_dims, group_dims);
    Ok(())
}

/// Fused gate+up variant of `call_afq_gather_qmm_rhs`. Computes
/// `y = activation(gate_proj(x)) * up_proj(x)` in one launch. `act_idx` maps
/// to the same `ACT` codes used by `qmm_t_gate_up`:
/// 0=Silu/Swish, 1=Gelu(tanh approx), 2=Gelu(erf approx), 3=Relu.
#[allow(clippy::too_many_arguments)]
pub fn call_afq_gather_qmm_rhs_gate_up(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    ty: DType,
    x: &Buffer,
    x_offset: usize,
    w_gate: &Buffer,
    scales_gate: &Buffer,
    biases_gate: &Buffer,
    w_up: &Buffer,
    scales_up: &Buffer,
    biases_up: &Buffer,
    indices: &Buffer,
    out: &Buffer,
    m: usize,
    n: usize,
    k: usize,
    bits: usize,
    group_size: usize,
    act_idx: usize,
) -> Result<(), MetalKernelError> {
    let type_string = match ty {
        DType::F32 => "float",
        DType::BF16 => "bfloat16_t",
        DType::F16 => "float16_t",
        other => {
            return Err(MetalKernelError::DTypeMismatch {
                expected: vec![DType::F32, DType::F16, DType::BF16],
                got: other,
            })
        }
    };

    // Reuse the same tile picker but only BM=16/32 are instantiated for the
    // fused kernel (BM=64 isn't a win and would double instantiations).
    let (bm, bn, bk, wm, wn) = if m >= 128 {
        (32, 32, 32, 1, 2)
    } else {
        (16, 32, 32, 1, 2)
    };
    let align_m = m.is_multiple_of(bm);
    let align_n = n.is_multiple_of(bn);
    let align_k = k.is_multiple_of(bk);
    let am = if align_m { "t" } else { "n" };
    let an = if align_n { "t" } else { "n" };
    let ak = if align_k { "t" } else { "n" };
    let name = format!(
        "affine_gather_qmm_rhs_gate_up_{type_string}_gs_{group_size}_b_{bits}_act_{act_idx}_bm_{bm}_bn_{bn}_bk_{bk}_wm_{wm}_wn_{wn}_alM_{am}_alN_{an}_alK_{ak}",
    );

    let pipeline = kernels.load_pipeline(device, &name)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    encoder.set_input_buffer(0, Some(x), x_offset);
    encoder.set_input_buffer(1, Some(w_gate), 0);
    encoder.set_input_buffer(2, Some(scales_gate), 0);
    encoder.set_input_buffer(3, Some(biases_gate), 0);
    encoder.set_input_buffer(4, Some(w_up), 0);
    encoder.set_input_buffer(5, Some(scales_up), 0);
    encoder.set_input_buffer(6, Some(biases_up), 0);
    encoder.set_input_buffer(7, Some(indices), 0);
    encoder.set_output_buffer(8, Some(out), 0);
    <i32 as EncoderParam>::set_param(encoder, 9, m as i32);
    <i32 as EncoderParam>::set_param(encoder, 10, n as i32);
    <i32 as EncoderParam>::set_param(encoder, 11, k as i32);

    let group_dims = MTLSize {
        width: 32,
        height: wn,
        depth: wm,
    };
    let grid_dims = MTLSize {
        width: n.div_ceil(bn),
        height: m.div_ceil(bm),
        depth: 1,
    };
    encoder.dispatch_thread_groups(grid_dims, group_dims);
    Ok(())
}

/// Fused gate+up qmm_t with GLU activation, all in one dispatch.
/// `act_code`: 0=silu, 1=gelu(tanh), 2=gelu(erf), 3=relu.
#[allow(clippy::too_many_arguments)]
pub fn call_afq_qmm_gate_up(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    ty: DType,
    x: (&Buffer, usize),
    w_gate: &Buffer,
    scales_gate: &Buffer,
    biases_gate: &Buffer,
    w_up: &Buffer,
    scales_up: &Buffer,
    biases_up: &Buffer,
    out: &Buffer,
    m: usize,
    n: usize,
    k: usize,
    bits: usize,
    group_size: usize,
    act_code: u32,
) -> Result<(), MetalKernelError> {
    let type_string = match ty {
        DType::F32 => "float",
        DType::BF16 => "bfloat16_t",
        DType::F16 => "float16_t",
        other => {
            return Err(MetalKernelError::DTypeMismatch {
                expected: vec![DType::F32, DType::F16, DType::BF16],
                got: other,
            })
        }
    };
    let aligned = if n.is_multiple_of(32) {
        "true"
    } else {
        "false"
    };
    let name = format!(
        "qmm_t_gate_up_{type_string}_gs_{group_size}_b_{bits}_act_{act_code}_alN_{aligned}"
    );
    let pipeline = kernels.load_pipeline(device, &name)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    encoder.set_input_buffer(0, Some(w_gate), 0);
    encoder.set_input_buffer(1, Some(scales_gate), 0);
    encoder.set_input_buffer(2, Some(biases_gate), 0);
    encoder.set_input_buffer(3, Some(w_up), 0);
    encoder.set_input_buffer(4, Some(scales_up), 0);
    encoder.set_input_buffer(5, Some(biases_up), 0);
    encoder.set_input_buffer(6, Some(x.0), x.1);
    encoder.set_output_buffer(7, Some(out), 0);
    <i32 as EncoderParam>::set_param(encoder, 8, k as i32);
    <i32 as EncoderParam>::set_param(encoder, 9, n as i32);
    <i32 as EncoderParam>::set_param(encoder, 10, m as i32);

    let group = MTLSize {
        width: 32,
        height: 2,
        depth: 2,
    };
    let grid = MTLSize {
        width: n.div_ceil(32),
        height: m.div_ceil(32),
        depth: 1,
    };
    encoder.dispatch_thread_groups(grid, group);
    Ok(())
}

/// Fused QKV qmm_t: single dispatch producing three outputs from three
/// weight matrices. `n_q` and `n_k` must be multiples of 32 (tile width)
/// so no threadgroup straddles two matrices.
#[allow(clippy::too_many_arguments)]
pub fn call_afq_qmm_qkv(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    ty: DType,
    x: (&Buffer, usize),
    w_q: &Buffer,
    scales_q: &Buffer,
    biases_q: &Buffer,
    w_k: &Buffer,
    scales_k: &Buffer,
    biases_k: &Buffer,
    w_v: &Buffer,
    scales_v: &Buffer,
    biases_v: &Buffer,
    q_out: &Buffer,
    k_out: &Buffer,
    v_out: &Buffer,
    m: usize,
    n_q: usize,
    n_k: usize,
    n_v: usize,
    k: usize,
    bits: usize,
    group_size: usize,
) -> Result<(), MetalKernelError> {
    let type_string = match ty {
        DType::F32 => "float",
        DType::BF16 => "bfloat16_t",
        DType::F16 => "float16_t",
        other => {
            return Err(MetalKernelError::DTypeMismatch {
                expected: vec![DType::F32, DType::F16, DType::BF16],
                got: other,
            })
        }
    };
    let name = format!("qmm_t_qkv_{type_string}_gs_{group_size}_b_{bits}");
    let pipeline = kernels.load_pipeline(device, &name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    encoder.set_input_buffer(0, Some(w_q), 0);
    encoder.set_input_buffer(1, Some(scales_q), 0);
    encoder.set_input_buffer(2, Some(biases_q), 0);
    encoder.set_input_buffer(3, Some(w_k), 0);
    encoder.set_input_buffer(4, Some(scales_k), 0);
    encoder.set_input_buffer(5, Some(biases_k), 0);
    encoder.set_input_buffer(6, Some(w_v), 0);
    encoder.set_input_buffer(7, Some(scales_v), 0);
    encoder.set_input_buffer(8, Some(biases_v), 0);
    encoder.set_input_buffer(9, Some(x.0), x.1);
    encoder.set_output_buffer(10, Some(q_out), 0);
    encoder.set_output_buffer(11, Some(k_out), 0);
    encoder.set_output_buffer(12, Some(v_out), 0);
    <i32 as EncoderParam>::set_param(encoder, 13, k as i32);
    <i32 as EncoderParam>::set_param(encoder, 14, n_q as i32);
    <i32 as EncoderParam>::set_param(encoder, 15, n_k as i32);
    <i32 as EncoderParam>::set_param(encoder, 16, n_v as i32);
    <i32 as EncoderParam>::set_param(encoder, 17, m as i32);

    let total_n = n_q + n_k + n_v;
    let group = MTLSize {
        width: 32,
        height: 2,
        depth: 2,
    };
    let grid = MTLSize {
        width: total_n.div_ceil(32),
        height: m.div_ceil(32),
        depth: 1,
    };
    encoder.dispatch_thread_groups(grid, group);
    Ok(())
}
