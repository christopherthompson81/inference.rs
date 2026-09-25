// Portions of this file are adapted from Apple's MLX framework
// (https://github.com/ml-explore/mlx)
// Licensed under the Apache License 2.0
// Copyright © 2023 Apple Inc.

use super::*;

/// Immutable parameters that fully describe a single `sort` / `argsort`
/// operation.  Bundling them keeps the public API readable as the kernel
/// signatures grow.
///
/// All slice references point into caller‑owned data – the struct itself
/// owns **no** memory.
#[derive(Debug)]
pub struct SortArgs<'a> {
    /// Axis along which the values are sorted.
    pub axis: usize,
    /// Shape of the input tensor.
    pub shape: &'a [usize],
    /// Strides (element‑wise) for the input tensor.
    pub strides: &'a [usize],
    /// Shape of the output tensor.
    pub out_shape: &'a [usize],
    /// Strides (element‑wise) for the output tensor.
    pub out_strides: &'a [usize],
    /// Whether the input tensor is already contiguous in memory.
    pub in_contiguous: bool,
    /// Element type of the input tensor.
    pub in_ty: DType,
    /// Element type of the output tensor (`u32` for `argsort`).
    pub out_ty: DType,
    /// GPU buffer holding the input tensor elements.
    pub src: &'a Buffer,
    /// Element offset into `src` where the region to be sorted begins.
    pub src_offset: usize,
    /// GPU buffer that will receive the sorted values / indices.
    pub dst: &'a Buffer,
    pub bn: usize,
    pub tn: usize,
    pub n_blocks: usize,
}

/// Scratch buffers that can be reused between consecutive multi‑block sort
/// calls.  Providing them avoids the internal allocations performed each
/// time by `call_multi_block_sort`.
#[derive(Debug)]
pub struct MultiBlockSortCache {
    dev_vals_ping: Arc<Buffer>,
    dev_vals_pong: Arc<Buffer>,
    dev_idxs_ping: Arc<Buffer>,
    dev_idxs_pong: Arc<Buffer>,
    block_partitions: Arc<Buffer>,
}

// --------------------------------------------------------------------------
// Simple LRU cache for scratch buffers used by multi‑block sort / argsort
// --------------------------------------------------------------------------

use std::collections::VecDeque;

/// Uniquely identifies a scratch‑buffer layout.
///
/// We key only on the dimensions that impact required buffer sizes.  If two
/// calls have the same `(rows, cols, dtype_size, blocks)` tuple they can
/// safely reuse the same scratch buffers.
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
struct CacheKey {
    n_rows: usize,
    size_sorted_axis: usize,
    dtype_size: usize,
    n_blocks: usize,
}

/// The buffers that can be reused between kernel launches.
#[derive(Debug)]
struct CachedBuffers {
    dev_vals_ping: Arc<Buffer>,
    dev_vals_pong: Arc<Buffer>,
    dev_idxs_ping: Arc<Buffer>,
    dev_idxs_pong: Arc<Buffer>,
    block_partitions: Arc<Buffer>,
}

/// Thread‑safe LRU cache with a fixed maximum number of entries.
///
/// *   The cache is **thread‑safe** (internally uses a `RwLock`).
/// *   Reuses scratch buffers across launches that share the same
///     (`rows`, `cols`, `dtype`, `blocks`) tuple.
pub struct SortScratchCache {
    cap: usize,
    map: RwLock<HashMap<CacheKey, CachedBuffers>>,
    order: RwLock<VecDeque<CacheKey>>, // most‑recent access at the back
}

impl SortScratchCache {
    /// Create a new cache that holds at most `cap` distinct buffer sets.
    pub fn new(cap: usize) -> Self {
        Self {
            cap,
            map: RwLock::new(HashMap::new()),
            order: RwLock::new(VecDeque::new()),
        }
    }

    /// Retrieve a set of scratch buffers (creating them if absent) and return
    /// a `MultiBlockSortCache::External` pointing to them.
    ///
    /// The borrow lasts for `'a` (caller’s scope) and is safe because the
    /// buffers live inside `self`.
    pub fn checkout(
        &self,
        metal_device: &MetalDevice,
        n_rows: usize,
        size_sorted_axis: usize,
        dtype: DType,
        n_blocks: usize,
    ) -> MultiBlockSortCache {
        let key = CacheKey {
            n_rows,
            size_sorted_axis,
            dtype_size: dtype.size_in_bytes(),
            n_blocks,
        };

        // Fast path – try read‑lock first
        if let Some(buffers) = self.map.read().unwrap().get(&key) {
            // Touch LRU order (needs write lock)
            self.touch_key(key);
            return MultiBlockSortCache {
                dev_vals_ping: buffers.dev_vals_ping.clone(),
                dev_vals_pong: buffers.dev_vals_pong.clone(),
                dev_idxs_ping: buffers.dev_idxs_ping.clone(),
                dev_idxs_pong: buffers.dev_idxs_pong.clone(),
                block_partitions: buffers.block_partitions.clone(),
            };
        }

        // Slow path – allocate new buffers
        let mut map_guard = self.map.write().unwrap();
        let mut order_guard = self.order.write().unwrap();

        // Evict least‑recently used if we’re at capacity
        if map_guard.len() == self.cap {
            if let Some(oldest) = order_guard.pop_front() {
                map_guard.remove(&oldest);
            }
        }

        // Allocate fresh buffers
        let elem_vals = n_rows * size_sorted_axis;
        let cached = CachedBuffers {
            dev_vals_ping: metal_device
                .new_buffer(elem_vals, dtype, "cache_vals_ping")
                .expect("alloc dev_vals_ping"),
            dev_vals_pong: metal_device
                .new_buffer(elem_vals, dtype, "cache_vals_pong")
                .expect("alloc dev_vals_pong"),
            dev_idxs_ping: metal_device
                .new_buffer(elem_vals, DType::U32, "cache_idxs_ping")
                .expect("alloc dev_idxs_ping"),
            dev_idxs_pong: metal_device
                .new_buffer(elem_vals, DType::U32, "cache_idxs_pong")
                .expect("alloc dev_idxs_pong"),
            block_partitions: metal_device
                .new_buffer(n_rows * (n_blocks + 1), DType::U32, "cache_partitions")
                .expect("alloc block_partitions"),
        };

        map_guard.insert(key, cached);
        order_guard.push_back(key);

        // SAFETY: we must re‑borrow from the fresh entry since `map_guard`
        // holds ownership of the buffers.
        let buffers = map_guard.get(&key).unwrap();
        MultiBlockSortCache {
            dev_vals_ping: buffers.dev_vals_ping.clone(),
            dev_vals_pong: buffers.dev_vals_pong.clone(),
            dev_idxs_ping: buffers.dev_idxs_ping.clone(),
            dev_idxs_pong: buffers.dev_idxs_pong.clone(),
            block_partitions: buffers.block_partitions.clone(),
        }
    }

    /// Move `key` to the back of the LRU order list.
    fn touch_key(&self, key: CacheKey) {
        let mut order = self.order.write().unwrap();
        if let Some(pos) = order.iter().position(|k| *k == key) {
            order.remove(pos);
        }
        order.push_back(key);
    }
}

/// How the copy kernel should behave.
#[derive(Copy, Clone)]
enum CopyType {
    /// The last axis is contiguous – we can treat the tensor as a 1‑D slice.
    Vector,
    /// Arbitrary layout – we need to jump using the supplied strides.
    General,
}

fn type_to_name(dt: DType) -> &'static str {
    match dt {
        DType::F32 => "float32",
        DType::F16 => "float16",
        DType::BF16 => "bfloat16",
        DType::I32 => "int32",
        DType::I64 => "int64",
        DType::U32 => "uint32",
        DType::U8 => "uint8",
        _ => "unknown",
    }
}

#[allow(clippy::too_many_arguments)]
fn call_copy_gpu_inplace(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    ty: DType,
    shape: &[usize],
    src_strides: &[usize],
    dst_strides: &[usize],
    src: &Buffer,
    src_offset: usize,
    dst: &Buffer,
    dst_offset: usize,
    copy_type: CopyType,
) -> Result<(), MetalKernelError> {
    // === Constants & helpers =================================================
    const MAX_COPY_SPECIALIZED_DIMS: usize = 3;

    /// https://github.com/ml-explore/mlx/blob/eebe73001affcb424171e9d49657e508f70a9201/mlx/backend/metal/utils.h#L87
    #[inline]
    fn work_per_thread_for_dtype(dt: DType) -> usize {
        let wpt = 8 / dt.size_in_bytes().max(1);
        wpt.max(1)
    }

    #[inline]
    fn ceil_div(x: usize, y: usize) -> usize {
        x.div_ceil(y)
    }

    // === Sanity checks =======================================================
    if shape.is_empty() {
        // Nothing to do – mimic early‑return in the C++ version.
        return Ok(());
    }
    assert!(
        shape.len() == src_strides.len() && shape.len() == dst_strides.len(),
        "shape/stride rank mismatch in call_copy_gpu_inplace"
    );

    // === Derived sizes / flags ==============================================
    let elem_count: usize = shape.iter().product();
    let large = match copy_type {
        CopyType::General => {
            // Allow negative strides – original code switches to 32‑bit indexing once strides
            // are flattened, but here we only care about the element count threshold.
            elem_count > i32::MAX as usize
        }
        CopyType::Vector => elem_count > u32::MAX as usize,
    };
    let work_per_thread = match copy_type {
        CopyType::Vector => work_per_thread_for_dtype(ty),
        CopyType::General => {
            if shape.len() > MAX_COPY_SPECIALIZED_DIMS {
                if large {
                    4
                } else {
                    2
                }
            } else {
                1
            }
        }
    };

    // === Kernel name construction ===========================================
    let mut kernel_name = match copy_type {
        CopyType::Vector => {
            if large {
                "v2".to_owned()
            } else {
                "v".to_owned()
            }
        }
        CopyType::General => "g".to_owned(),
    };

    if matches!(copy_type, CopyType::General) {
        if shape.len() <= MAX_COPY_SPECIALIZED_DIMS {
            kernel_name.push_str(&shape.len().to_string());
        } else {
            kernel_name.push_str(&format!("n{work_per_thread}"));
        }
        if large {
            kernel_name.push_str("large");
        }
    }

    // The original MLX kernels encode source/destination dtypes in the mangled
    // name, even when they are the same.
    kernel_name.push_str(&format!("_copy{}{}", type_to_name(ty), type_to_name(ty)));

    // === Pipeline & encoder ==================================================
    let pipeline = kernels.load_pipeline(device, &kernel_name)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    // === Buffers (slots 0/1) =================================================
    let byte_offset_src = src_offset * ty.size_in_bytes();
    let byte_offset_dst = dst_offset * ty.size_in_bytes();
    encoder.set_input_buffer(0, Some(src), byte_offset_src);
    encoder.set_output_buffer(1, Some(dst), byte_offset_dst);

    // === Specialisation for each CopyType ===================================
    match copy_type {
        // ---------------------------------------------------------------------
        CopyType::Vector => {
            // Slot 2 – total elements (32‑ or 64‑bit depending on `large`)
            if large {
                <i64 as EncoderParam>::set_param(encoder, 2, elem_count as i64);
            } else {
                <i32 as EncoderParam>::set_param(encoder, 2, elem_count as i32);
            }

            // Grid
            let nthreads = ceil_div(elem_count, work_per_thread);
            let tg_size = pipeline.max_total_threads_per_threadgroup();
            let tg_size = tg_size.min(nthreads);

            let group_dims = MTLSize {
                width: tg_size,
                height: 1,
                depth: 1,
            };
            let grid_dims = if large {
                // 64‑bit indexing path – fall back to 2‑D tiling helper.
                get_2d_grid_dims_divisor(shape, dst_strides, work_per_thread)
            } else {
                MTLSize {
                    width: nthreads,
                    height: 1,
                    depth: 1,
                }
            };
            encoder.dispatch_threads(grid_dims, group_dims);
        }

        // ---------------------------------------------------------------------
        CopyType::General => {
            let ndim = shape.len() as i32;

            // ---- Shape / stride descriptors ---------------------------------
            if shape.len() > 3 {
                let shape_i32: Vec<i32> = shape.iter().map(|&x| x as i32).collect();
                encoder.set_bytes_raw(
                    2,
                    std::mem::size_of::<i32>() * shape_i32.len(),
                    shape_i32.as_ptr() as *const _,
                );
            }
            // Strides – always required (slot 3)
            let strides_in_i64: Vec<i64> = src_strides.iter().map(|&x| x as i64).collect();
            encoder.set_bytes_raw(
                3,
                std::mem::size_of::<i64>() * strides_in_i64.len(),
                strides_in_i64.as_ptr() as *const _,
            );

            // If the kernel is the generic “gn*” variant we also pass `ndim`
            if shape.len() > MAX_COPY_SPECIALIZED_DIMS {
                <i32 as EncoderParam>::set_param(encoder, 5, ndim);
            }

            // ---- Thread/work‑grid ------------------------------------------
            let mut dim0 = *shape.last().unwrap_or(&1);
            let dim1 = if shape.len() > 1 {
                shape[shape.len() - 2]
            } else {
                1
            };
            let rest = elem_count / (dim0 * dim1);
            if shape.len() > MAX_COPY_SPECIALIZED_DIMS {
                dim0 = ceil_div(dim0, work_per_thread);
            }

            assert_eq!(
                pipeline.max_total_threads_per_threadgroup(),
                1024,
                "copy kernels expect a 1024-lane threadgroup"
            );
            let group_dims = get_block_dims(dim0, dim1, rest, 10);
            let grid_dims = MTLSize {
                width: dim0,
                height: dim1,
                depth: rest,
            };
            encoder.dispatch_threads(grid_dims, group_dims);
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn call_single_block_sort<'a>(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    args: &SortArgs<'a>,
    bn: usize,
    tn: usize,
    argsort: bool,
) -> Result<(), MetalKernelError> {
    // --- destructure helper -------------------------------------------------
    let axis = args.axis;
    let shape = args.shape;
    let strides = args.strides;
    let out_strides = args.out_strides;
    let in_contiguous = args.in_contiguous;
    let in_ty = args.in_ty;
    let out_ty = args.out_ty;
    let src = args.src;
    let src_offset = args.src_offset;
    let dst = args.dst;
    let n_rows = shape.iter().product::<usize>() / shape[axis];

    let mut in_nc_str = strides.iter().map(|x| *x as i64).collect::<Vec<_>>();
    in_nc_str.remove(axis);

    let mut out_nc_str = out_strides.iter().map(|x| *x as i64).collect::<Vec<_>>();
    out_nc_str.remove(axis);

    let mut nc_shape = shape.iter().map(|x| *x as i32).collect::<Vec<_>>();
    nc_shape.remove(axis);

    let nc_dim = nc_shape.len();

    let size_sorted_axis = shape[axis];
    let in_stride_sorted_axis = strides[axis];
    let out_stride_sorted_axis = out_strides[axis];

    macro_rules! check_strides {
        ($strides:expr, $sort_stride:expr) => {{
            let min_stride = $strides.iter().min().unwrap();
            let max_stride = $strides.iter().max().unwrap();
            $sort_stride == *min_stride || $sort_stride == *max_stride
        }};
    }
    let contiguous = in_contiguous
        && check_strides!(strides, in_stride_sorted_axis)
        && check_strides!(out_strides, out_stride_sorted_axis);

    use std::fmt::Write as _;
    let mut name = String::new();
    if contiguous {
        name.push('c');
    } else {
        name.push_str("nc");
    }
    if argsort {
        name.push_str("arg");
    }
    write!(
        &mut name,
        "_block_sort_{}_{}_bn{}_tn{}",
        type_to_name(in_ty),
        type_to_name(out_ty),
        bn,
        tn
    )
    .unwrap();

    let pipeline = kernels.load_pipeline(device, &name)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    // Buffers
    encoder.set_input_buffer(0, Some(src), src_offset);
    encoder.set_output_buffer(1, Some(dst), 0);

    // Scalar params
    <i32 as EncoderParam>::set_param(encoder, 2, size_sorted_axis as i32);
    <i32 as EncoderParam>::set_param(encoder, 3, in_stride_sorted_axis as i32);
    <i32 as EncoderParam>::set_param(encoder, 4, out_stride_sorted_axis as i32);

    if contiguous {
        let mut in_stride_segment_axis = i32::MAX;
        let mut out_stride_segment_axis = i32::MAX;
        for i in 0..in_nc_str.len() {
            if nc_shape[i] == 1 {
                continue;
            }
            if in_nc_str[i] > i32::MAX as i64 || out_nc_str[i] > i32::MAX as i64 {
                panic!(
                    "Stride too large in single_block_sort in {} out {}",
                    in_nc_str[i], out_nc_str[i]
                );
            }
            in_stride_segment_axis = in_stride_segment_axis.min(in_nc_str[i] as i32);
            out_stride_segment_axis = out_stride_segment_axis.min(out_nc_str[i] as i32);
        }

        <i32 as EncoderParam>::set_param(encoder, 5, in_stride_segment_axis);
        <i32 as EncoderParam>::set_param(encoder, 6, out_stride_segment_axis);
    } else {
        <i32 as EncoderParam>::set_param(encoder, 5, nc_dim as i32);
        if nc_shape.is_empty() {
            let shape = 0i32;
            let stride = 0i64;

            <i32 as EncoderParam>::set_param(encoder, 6, shape);
            <i64 as EncoderParam>::set_param(encoder, 7, stride);
            <i64 as EncoderParam>::set_param(encoder, 8, stride);
        } else {
            encoder.set_bytes_raw(
                6,
                std::mem::size_of::<i32>() * nc_shape.len(),
                nc_shape.as_ptr() as *const _,
            );
            encoder.set_bytes_raw(
                7,
                std::mem::size_of::<i64>() * in_nc_str.len(),
                in_nc_str.as_ptr() as *const _,
            );
            encoder.set_bytes_raw(
                8,
                std::mem::size_of::<i64>() * out_nc_str.len(),
                out_nc_str.as_ptr() as *const _,
            );
        }
    }

    // Dispatch
    let group_dims = MTLSize {
        width: bn,
        height: 1,
        depth: 1,
    };
    let grid_dims = MTLSize {
        width: 1,
        height: n_rows,
        depth: 1,
    };
    encoder.dispatch_thread_groups(grid_dims, group_dims);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn call_multi_block_sort<'a>(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    args: &SortArgs<'a>,
    bn: usize,
    tn: usize,
    n_blocks: usize,
    argsort: bool,
    cache: &MultiBlockSortCache,
) -> Result<(), MetalKernelError> {
    // --- destructure helper -------------------------------------------------
    let axis = args.axis;
    let shape = args.shape;
    let strides = args.strides;
    let out_strides = args.out_strides;
    let in_ty = args.in_ty;
    let out_ty = args.out_ty;
    let src = args.src;
    let src_offset = args.src_offset;
    let dst = args.dst;
    let n_rows = shape.iter().product::<usize>() / shape[axis];

    let mut nc_str = strides.iter().map(|x| *x as i64).collect::<Vec<_>>();
    nc_str.remove(axis);

    let mut nc_shape = shape.iter().map(|x| *x as i32).collect::<Vec<_>>();
    nc_shape.remove(axis);

    let nc_dim = nc_shape.len();

    if nc_dim == 0 {
        nc_shape = vec![0];
        nc_str = vec![1];
    }

    let size_sorted_axis = shape[axis];
    let stride_sorted_axis = strides[axis];

    // ------------------------------------------------------------------
    // Acquire scratch buffers (either cached or freshly allocated)
    // ------------------------------------------------------------------

    // Scratch buffers supplied by the caller (from SortScratchCache)
    let dev_vals_0 = cache.dev_vals_ping.clone();
    let dev_vals_1 = cache.dev_vals_pong.clone();
    let dev_idxs_0 = cache.dev_idxs_ping.clone();
    let dev_idxs_1 = cache.dev_idxs_pong.clone();
    let block_partitions = cache.block_partitions.clone();

    // Do blockwise sort
    {
        let name = format!(
            "sort_mbsort_{}_{}_bn{bn}_tn{tn}",
            type_to_name(in_ty),
            type_to_name(DType::U32)
        );

        let pipeline = kernels.load_pipeline(device, &name)?;

        let encoder = ep.encoder();
        let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);

        encoder.set_input_buffer(0, Some(src), src_offset);
        encoder.set_output_buffer(1, Some(&*dev_vals_0), 0);
        encoder.set_output_buffer(2, Some(&*dev_idxs_0), 0);
        <i32 as EncoderParam>::set_param(encoder, 3, size_sorted_axis as i32);
        <i32 as EncoderParam>::set_param(encoder, 4, stride_sorted_axis as i32);
        <i32 as EncoderParam>::set_param(encoder, 5, nc_dim as i32);
        encoder.set_bytes_raw(
            6,
            std::mem::size_of::<i32>() * nc_shape.len(),
            nc_shape.as_ptr() as *const _,
        );
        encoder.set_bytes_raw(
            7,
            std::mem::size_of::<i64>() * nc_str.len(),
            nc_str.as_ptr() as *const _,
        );

        // Dispatch
        let group_dims = MTLSize {
            width: bn,
            height: 1,
            depth: 1,
        };
        let grid_dims = MTLSize {
            width: n_blocks,
            height: n_rows,
            depth: 1,
        };
        encoder.dispatch_thread_groups(grid_dims, group_dims);
    }

    // Do merges
    let mut ping = false;
    #[allow(unused_assignments)]
    let mut dev_vals_in = dev_vals_0.clone();
    #[allow(unused_assignments)]
    let mut dev_idxs_in = dev_idxs_0.clone();
    let mut dev_vals_out = dev_vals_1.clone();
    let mut dev_idxs_out = dev_idxs_1.clone();

    let n_thread_per_group = std::cmp::min(n_blocks + 1, 1024);

    let mut merge_tiles = 2;
    while (merge_tiles / 2) < n_blocks {
        dev_vals_in = if ping {
            dev_vals_1.clone()
        } else {
            dev_vals_0.clone()
        };
        dev_idxs_in = if ping {
            dev_idxs_1.clone()
        } else {
            dev_idxs_0.clone()
        };
        dev_vals_out = if ping {
            dev_vals_0.clone()
        } else {
            dev_vals_1.clone()
        };
        dev_idxs_out = if ping {
            dev_idxs_0.clone()
        } else {
            dev_idxs_1.clone()
        };
        ping = !ping;

        // Do partition
        {
            let name = format!(
                "partition_mbsort_{}_{}_bn{bn}_tn{tn}",
                type_to_name(in_ty),
                type_to_name(DType::U32)
            );

            let pipeline = kernels.load_pipeline(device, &name)?;

            let encoder = ep.encoder();
            let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
            encoder.set_compute_pipeline_state(&pipeline);

            encoder.set_output_buffer(0, Some(&*block_partitions), 0);
            encoder.set_input_buffer(1, Some(&*dev_vals_in), 0);
            encoder.set_input_buffer(2, Some(&*dev_idxs_in), 0);
            <i32 as EncoderParam>::set_param(encoder, 3, size_sorted_axis as i32);
            <i32 as EncoderParam>::set_param(encoder, 4, merge_tiles as i32);
            <i32 as EncoderParam>::set_param(encoder, 5, n_blocks as i32);

            // Dispatch
            let group_dims = MTLSize {
                width: n_thread_per_group as usize,
                height: 1,
                depth: 1,
            };
            let grid_dims = MTLSize {
                width: 1,
                height: n_rows,
                depth: 1,
            };
            encoder.dispatch_thread_groups(grid_dims, group_dims);
        }

        // Do merge
        {
            let name = format!(
                "merge_mbsort_{}_{}_bn{bn}_tn{tn}",
                type_to_name(in_ty),
                type_to_name(DType::U32)
            );

            let pipeline = kernels.load_pipeline(device, &name)?;

            let encoder = ep.encoder();
            let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
            encoder.set_compute_pipeline_state(&pipeline);

            encoder.set_input_buffer(0, Some(&*block_partitions), 0);
            encoder.set_input_buffer(1, Some(&*dev_vals_in), 0);
            encoder.set_input_buffer(2, Some(&*dev_idxs_in), 0);
            encoder.set_output_buffer(3, Some(&*dev_vals_out), 0);
            encoder.set_output_buffer(4, Some(&*dev_idxs_out), 0);
            <i32 as EncoderParam>::set_param(encoder, 5, size_sorted_axis as i32);
            <i32 as EncoderParam>::set_param(encoder, 6, merge_tiles as i32);
            <i32 as EncoderParam>::set_param(encoder, 7, n_blocks as i32);

            // Dispatch
            let group_dims = MTLSize {
                width: bn,
                height: 1,
                depth: 1,
            };
            let grid_dims = MTLSize {
                width: n_blocks,
                height: n_rows,
                depth: 1,
            };
            encoder.dispatch_thread_groups(grid_dims, group_dims);
        }

        merge_tiles *= 2;
    }

    // Copy outputs with appropriate strides
    let mut strides = out_strides.to_vec();
    for strides_ax in strides.iter_mut().skip(axis + 1) {
        *strides_ax *= args.out_shape[axis];
    }
    strides[axis] = 1;

    let copy_src = if argsort { dev_idxs_out } else { dev_vals_out };

    call_copy_gpu_inplace(
        device,
        ep,
        kernels,
        out_ty,
        args.out_shape,
        &strides,
        out_strides,
        &copy_src, // deref Arc
        0,
        dst,
        0,
        if axis == shape.len() - 1 {
            CopyType::Vector
        } else {
            CopyType::General
        },
    )?;

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn call_block_sort<'a>(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    args: &SortArgs<'a>,
    argsort: bool,
    cache: &MultiBlockSortCache,
) -> Result<(), MetalKernelError> {
    // --- destructure helper -------------------------------------------------
    let bn = args.bn;
    let tn = args.tn;
    let n_blocks = args.n_blocks;

    if n_blocks > 1 {
        // multi‑block path
        call_multi_block_sort(device, ep, kernels, args, bn, tn, n_blocks, argsort, cache)
    } else {
        // single‑block path
        call_single_block_sort(device, ep, kernels, args, bn, tn, argsort)
    }
}

/// Sorts values along the last axis of a contiguous tensor.
pub fn call_sort<'a>(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    args: &SortArgs<'a>,
    cache: &MultiBlockSortCache,
) -> Result<(), MetalKernelError> {
    call_block_sort(device, ep, kernels, args, /* argsort = */ false, cache)
}

/// Returns the indices that would sort `src` along the last axis.
pub fn call_argsort<'a>(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    args: &SortArgs<'a>,
    cache: &MultiBlockSortCache,
) -> Result<(), MetalKernelError> {
    call_block_sort(device, ep, kernels, args, /* argsort = */ true, cache)
}
