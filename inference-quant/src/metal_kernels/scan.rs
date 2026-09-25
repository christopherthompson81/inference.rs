// Portions of this file are adapted from Apple's MLX framework
// (https://github.com/ml-explore/mlx)
// Licensed under the Apache License 2.0
// Copyright © 2023 Apple Inc.

use super::*;

#[allow(dead_code)]
pub enum ScanType {
    Sum,
    Prod,
    Max,
    Min,
    LogAddExp,
}

#[allow(clippy::too_many_arguments)]
pub fn call_scan(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    ty: DType,
    op: ScanType,
    xs: &Buffer,
    xs_offset: usize,
    axis: usize,
    shape: &[usize],
    strides: &[usize],
    reverse: bool,
    inclusive: bool,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let contiguous = strides[axis] == 1;
    let mut name = if contiguous {
        "contig_".to_string()
    } else {
        "strided_".to_string()
    };
    name.push_str("scan_");
    if reverse {
        name.push_str("reverse_");
    }
    if inclusive {
        name.push_str("inclusive_");
    } else {
        name.push_str("exclusive_");
    }
    match op {
        ScanType::Sum => name.push_str("sum_"),
        ScanType::Prod => name.push_str("prod_"),
        ScanType::Max => name.push_str("max_"),
        ScanType::Min => name.push_str("min_"),
        ScanType::LogAddExp => name.push_str("logaddexp_"),
    }

    let type_name = match ty {
        DType::F32 => "float32",
        DType::BF16 => "bfloat16",
        DType::F16 => "float16",
        DType::U8 => "uint8",
        DType::I16 => "int16",
        DType::I32 => "int32",
        DType::I64 => "int64",
        other => {
            return Err(MetalKernelError::DTypeMismatch {
                expected: vec![
                    DType::F32,
                    DType::F16,
                    DType::BF16,
                    DType::U8,
                    DType::I16,
                    DType::I32,
                    DType::I64,
                ],
                got: other,
            })
        }
    };
    name.push_str(&format!("{type_name}_{type_name}"));

    let pipeline = kernels.load_pipeline(device, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    if contiguous {
        encoder.set_input_buffer(0, Some(xs), xs_offset);
        encoder.set_output_buffer(1, Some(output), 0);

        let size = shape[axis];
        encoder.set_bytes_raw(
            2,
            std::mem::size_of::<usize>(),
            &size as *const usize as *const _,
        );

        // Compute the thread grid
        let n_reads = if ty.size_in_bytes() <= 4 { 4 } else { 2 };
        let simd_size = 32;
        let elements_per_simd = n_reads * simd_size;
        let mut thread_group_size = pipeline.max_total_threads_per_threadgroup();
        if size <= n_reads * 1024 {
            thread_group_size = size.div_ceil(elements_per_simd) * simd_size;
        } else if size <= n_reads * 2048 {
            thread_group_size = (size / 2).div_ceil(elements_per_simd) * simd_size;
        }
        thread_group_size = thread_group_size.min(pipeline.max_total_threads_per_threadgroup());
        let tmp_grid_dims = get_2d_grid_dims_divisor(shape, strides, size);

        let grid_dims = MTLSize {
            width: thread_group_size,
            height: tmp_grid_dims.width,
            depth: tmp_grid_dims.height,
        };
        let group_dims = MTLSize {
            width: thread_group_size,
            height: 1,
            depth: 1,
        };

        encoder.dispatch_threads(grid_dims, group_dims);
    } else {
        let size = shape[axis];
        let stride = strides[size];
        let _bm = 32;
        let bn = 32;
        let stride_blocks = stride.div_ceil(bn);

        encoder.set_input_buffer(0, Some(xs), xs_offset);
        encoder.set_output_buffer(1, Some(output), 0);

        encoder.set_bytes_raw(
            2,
            std::mem::size_of::<usize>(),
            &size as *const usize as *const _,
        );
        encoder.set_bytes_raw(
            3,
            std::mem::size_of::<usize>(),
            &stride as *const usize as *const _,
        );
        encoder.set_bytes_raw(
            4,
            std::mem::size_of::<usize>(),
            &stride_blocks as *const usize as *const _,
        );

        // Compute the thread grid
        let n_reads = if ty.size_in_bytes() <= 4 { 4 } else { 2 };
        let n_simdgroups = bn / n_reads;
        let thread_group_size = n_simdgroups * 32;
        let mut tmp_grid_dims = get_2d_grid_dims_divisor(shape, strides, size * stride);
        if tmp_grid_dims.width <= (u32::MAX as usize) / stride_blocks {
            tmp_grid_dims.width *= stride_blocks;
        } else {
            tmp_grid_dims.height *= stride_blocks;
        }

        let grid_dims = MTLSize {
            width: thread_group_size,
            height: tmp_grid_dims.width,
            depth: tmp_grid_dims.height,
        };
        let group_dims = MTLSize {
            width: thread_group_size,
            height: 1,
            depth: 1,
        };

        encoder.dispatch_threads(grid_dims, group_dims);
    }

    Ok(())
}
