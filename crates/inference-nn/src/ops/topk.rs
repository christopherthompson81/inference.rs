use super::*;

// ============================================================================
// Optimized parallel topk for CUDA
// Uses a dedicated kernel that's much faster than full sort for small k
// Single kernel call writes both values and indices - no post-processing needed
// ============================================================================

#[cfg(feature = "cuda")]
#[allow(clippy::cast_possible_truncation)]
pub(super) fn cuda_topk(input: &Tensor, k: usize) -> Result<TopKOutput> {
    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::CudaStorageSlice;
    use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
    use std::ffi::c_void;

    let input = final_logits_row(input)?;
    let dims = input.dims();
    let ncols = *dims
        .last()
        .ok_or_else(|| candle_core::Error::Msg("empty dims".to_string()))?;
    let nrows = (input.elem_count() / ncols) as i32;
    let ncols_i32 = ncols as i32;
    let k_i32 = k as i32;

    // Output shapes
    let mut out_dims = dims.to_vec();
    *out_dims.last_mut().unwrap() = k;
    let out_elem_count = nrows as usize * k;

    let (storage, _layout) = input.storage_and_layout();
    let storage = match &*storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("cuda_topk requires CUDA tensor"),
    };
    let dev = storage.device();
    let stream = dev.cuda_stream();
    let stream_raw = stream.cu_stream() as i64;

    let (src_ptr, _src_guard) = match &storage.slice {
        CudaStorageSlice::BF16(inp) => inp.device_ptr(&stream),
        CudaStorageSlice::F16(inp) => inp.device_ptr(&stream),
        CudaStorageSlice::F32(inp) => inp.device_ptr(&stream),
        _ => candle_core::bail!("cuda_topk only supports BF16/F16/F32"),
    };
    let src_ptr = src_ptr as *const c_void;

    // Allocate both output buffers
    let mut indices_dst = unsafe { dev.alloc::<u32>(out_elem_count) }?;
    let (indices_ptr, indices_guard) = indices_dst.device_ptr_mut(&stream);

    let (values_tensor, indices_tensor) = match input.dtype() {
        DType::BF16 => {
            let mut values_dst = unsafe { dev.alloc::<half::bf16>(out_elem_count) }?;
            let (values_ptr, values_guard) = values_dst.device_ptr_mut(&stream);

            unsafe {
                ffi::topk_bf16(
                    src_ptr,
                    values_ptr as *mut c_void,
                    indices_ptr as *mut c_void,
                    nrows,
                    ncols_i32,
                    k_i32,
                    stream_raw,
                );
            }

            drop(values_guard);
            drop(indices_guard);

            let values_storage = candle_core::cuda_backend::CudaStorage {
                slice: CudaStorageSlice::BF16(values_dst),
                device: dev.clone(),
            };
            let indices_storage = candle_core::cuda_backend::CudaStorage {
                slice: CudaStorageSlice::U32(indices_dst),
                device: dev.clone(),
            };

            let values_tensor = Tensor::from((
                candle_core::Storage::Cuda(values_storage),
                Shape::from_dims(&out_dims),
            ));
            let indices_tensor = Tensor::from((
                candle_core::Storage::Cuda(indices_storage),
                Shape::from_dims(&out_dims),
            ));
            (values_tensor, indices_tensor)
        }
        DType::F16 => {
            let mut values_dst = unsafe { dev.alloc::<half::f16>(out_elem_count) }?;
            let (values_ptr, values_guard) = values_dst.device_ptr_mut(&stream);

            unsafe {
                ffi::topk_f16(
                    src_ptr,
                    values_ptr as *mut c_void,
                    indices_ptr as *mut c_void,
                    nrows,
                    ncols_i32,
                    k_i32,
                    stream_raw,
                );
            }

            drop(values_guard);
            drop(indices_guard);

            let values_storage = candle_core::cuda_backend::CudaStorage {
                slice: CudaStorageSlice::F16(values_dst),
                device: dev.clone(),
            };
            let indices_storage = candle_core::cuda_backend::CudaStorage {
                slice: CudaStorageSlice::U32(indices_dst),
                device: dev.clone(),
            };

            let values_tensor = Tensor::from((
                candle_core::Storage::Cuda(values_storage),
                Shape::from_dims(&out_dims),
            ));
            let indices_tensor = Tensor::from((
                candle_core::Storage::Cuda(indices_storage),
                Shape::from_dims(&out_dims),
            ));
            (values_tensor, indices_tensor)
        }
        DType::F32 => {
            let mut values_dst = unsafe { dev.alloc::<f32>(out_elem_count) }?;
            let (values_ptr, values_guard) = values_dst.device_ptr_mut(&stream);

            unsafe {
                ffi::topk_f32(
                    src_ptr,
                    values_ptr as *mut c_void,
                    indices_ptr as *mut c_void,
                    nrows,
                    ncols_i32,
                    k_i32,
                    stream_raw,
                );
            }

            drop(values_guard);
            drop(indices_guard);

            let values_storage = candle_core::cuda_backend::CudaStorage {
                slice: CudaStorageSlice::F32(values_dst),
                device: dev.clone(),
            };
            let indices_storage = candle_core::cuda_backend::CudaStorage {
                slice: CudaStorageSlice::U32(indices_dst),
                device: dev.clone(),
            };

            let values_tensor = Tensor::from((
                candle_core::Storage::Cuda(values_storage),
                Shape::from_dims(&out_dims),
            ));
            let indices_tensor = Tensor::from((
                candle_core::Storage::Cuda(indices_storage),
                Shape::from_dims(&out_dims),
            ));
            (values_tensor, indices_tensor)
        }
        dt => candle_core::bail!("cuda_topk unsupported dtype: {:?}", dt),
    };

    Ok(TopKOutput {
        values: values_tensor,
        indices: indices_tensor,
    })
}

// Rows up to this many columns sort in one launch from shared memory (4 bytes per padded column, under 48 KiB)
const ARGSORT_ROWS_MAX_COLS: usize = 8192;
// `argsort_rows` dtype codes, matched by the switch in sort.cu
#[cfg(feature = "cuda")]
const ARGSORT_DTYPE_U8: i32 = 0;
#[cfg(feature = "cuda")]
const ARGSORT_DTYPE_U32: i32 = 1;
#[cfg(feature = "cuda")]
const ARGSORT_DTYPE_I64: i32 = 2;
#[cfg(feature = "cuda")]
const ARGSORT_DTYPE_BF16: i32 = 3;
#[cfg(feature = "cuda")]
const ARGSORT_DTYPE_F16: i32 = 4;
#[cfg(feature = "cuda")]
const ARGSORT_DTYPE_F32: i32 = 5;
#[cfg(feature = "cuda")]
const ARGSORT_DTYPE_F64: i32 = 6;

#[derive(Debug, Clone)]
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
struct ArgSort {
    asc: bool,
    last_dim: usize,
    inplace: bool,
}

impl candle_core::CustomOp1 for ArgSort {
    fn name(&self) -> &'static str {
        "argsort"
    }

    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &candle_core::Layout,
    ) -> Result<(candle_core::CpuStorage, candle_core::Shape)> {
        candle_core::bail!("argsort: CPU tensors sort through candle's arg_sort_last_dim")
    }

    #[allow(clippy::cast_possible_truncation)]
    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        storage: &candle_core::CudaStorage,
        layout: &candle_core::Layout,
    ) -> Result<(candle_core::CudaStorage, candle_core::Shape)> {
        use candle_core::backend::BackendStorage;
        use candle_core::cuda_backend::CudaStorageSlice;
        use candle_core::cuda_backend::cudarc::driver::DevicePtr;

        let dev = storage.device();
        let elem_count = layout.shape().elem_count();
        if elem_count == 0 {
            let dst = unsafe { dev.alloc::<u32>(0) }?;
            let dst = candle_core::cuda_backend::CudaStorage {
                slice: CudaStorageSlice::U32(dst),
                device: dev.clone(),
            };
            return Ok((dst, layout.shape().clone()));
        }
        let ncols = self.last_dim as i32;
        let nrows = elem_count as i32 / ncols;
        let dst = unsafe { dev.alloc::<u32>(elem_count) }?;

        use std::ffi::c_void;

        let (src, _src_guard) = match &storage.slice {
            CudaStorageSlice::U8(inp) => inp.device_ptr(inp.stream()),
            CudaStorageSlice::U32(inp) => inp.device_ptr(inp.stream()),
            CudaStorageSlice::I64(inp) => inp.device_ptr(inp.stream()),
            CudaStorageSlice::BF16(inp) => inp.device_ptr(inp.stream()),
            CudaStorageSlice::F16(inp) => inp.device_ptr(inp.stream()),
            CudaStorageSlice::F32(inp) => inp.device_ptr(inp.stream()),
            CudaStorageSlice::F64(inp) => inp.device_ptr(inp.stream()),
            _ => candle_core::bail!("Unexpected dtype in asort"),
        };
        let src_offset = layout.start_offset() * storage.dtype().size_in_bytes();
        let src_ptr = (src as usize + src_offset) as *const c_void;
        let (dst_ptr, dst_guard) = dst.device_ptr(dst.stream());
        let dst_ptr = dst_ptr as *mut c_void;
        let stream = dev.cuda_stream().cu_stream() as i64;
        if !self.inplace && self.last_dim <= ARGSORT_ROWS_MAX_COLS {
            let dtype = match storage.dtype() {
                candle_core::DType::U8 => ARGSORT_DTYPE_U8,
                candle_core::DType::U32 => ARGSORT_DTYPE_U32,
                candle_core::DType::I64 => ARGSORT_DTYPE_I64,
                candle_core::DType::BF16 => ARGSORT_DTYPE_BF16,
                candle_core::DType::F16 => ARGSORT_DTYPE_F16,
                candle_core::DType::F32 => ARGSORT_DTYPE_F32,
                candle_core::DType::F64 => ARGSORT_DTYPE_F64,
                _ => unreachable!("dtype matched above"),
            };
            let status = unsafe {
                ffi::argsort_rows(src_ptr, dst_ptr, nrows, ncols, dtype, self.asc, stream)
            };
            if status != 0 {
                candle_core::bail!("argsort_rows rejected dtype code {dtype}");
            }
            drop(dst_guard);
            let dst_ret = candle_core::cuda_backend::CudaStorage {
                slice: CudaStorageSlice::U32(dst),
                device: dev.clone(),
            };
            return Ok((dst_ret, layout.shape().clone()));
        }
        unsafe {
            if self.asc {
                match storage.dtype() {
                    candle_core::DType::U8 => {
                        ffi::asort_asc_u8(src_ptr, dst_ptr, nrows, ncols, self.inplace, stream)
                    }
                    candle_core::DType::U32 => {
                        ffi::asort_asc_u32(src_ptr, dst_ptr, nrows, ncols, self.inplace, stream)
                    }
                    candle_core::DType::I64 => {
                        ffi::asort_asc_i64(src_ptr, dst_ptr, nrows, ncols, self.inplace, stream)
                    }
                    candle_core::DType::BF16 => {
                        ffi::asort_asc_bf16(src_ptr, dst_ptr, nrows, ncols, self.inplace, stream)
                    }
                    candle_core::DType::F16 => {
                        ffi::asort_asc_f16(src_ptr, dst_ptr, nrows, ncols, self.inplace, stream)
                    }
                    candle_core::DType::F32 => {
                        ffi::asort_asc_f32(src_ptr, dst_ptr, nrows, ncols, self.inplace, stream)
                    }
                    candle_core::DType::F64 => {
                        ffi::asort_asc_f64(src_ptr, dst_ptr, nrows, ncols, self.inplace, stream)
                    }
                    _ => candle_core::bail!("Unexpected dtype in asort"),
                }
            } else {
                match storage.dtype() {
                    candle_core::DType::U8 => {
                        ffi::asort_desc_u8(src_ptr, dst_ptr, nrows, ncols, self.inplace, stream)
                    }
                    candle_core::DType::U32 => {
                        ffi::asort_desc_u32(src_ptr, dst_ptr, nrows, ncols, self.inplace, stream)
                    }
                    candle_core::DType::I64 => {
                        ffi::asort_desc_i64(src_ptr, dst_ptr, nrows, ncols, self.inplace, stream)
                    }
                    candle_core::DType::BF16 => {
                        ffi::asort_desc_bf16(src_ptr, dst_ptr, nrows, ncols, self.inplace, stream)
                    }
                    candle_core::DType::F16 => {
                        ffi::asort_desc_f16(src_ptr, dst_ptr, nrows, ncols, self.inplace, stream)
                    }
                    candle_core::DType::F32 => {
                        ffi::asort_desc_f32(src_ptr, dst_ptr, nrows, ncols, self.inplace, stream)
                    }
                    candle_core::DType::F64 => {
                        ffi::asort_desc_f64(src_ptr, dst_ptr, nrows, ncols, self.inplace, stream)
                    }
                    _ => candle_core::bail!("Unexpected dtype in asort"),
                }
            }
        }
        drop(dst_guard);
        let dst_ret = candle_core::cuda_backend::CudaStorage {
            slice: CudaStorageSlice::U32(dst),
            device: dev.clone(),
        };
        Ok((dst_ret, layout.shape().clone()))
    }
}

pub trait ArgSortOp {
    fn arg_sort(&self, asc: bool) -> Result<Tensor>;
    fn sort(&self, asc: bool) -> Result<(Tensor, Tensor)>;
}

impl ArgSortOp for Tensor {
    /// Returns the indices that sort the tensor along the last dimension.
    ///
    /// If `asc` is `true`, sorting is in ascending order. Otherwise sorting is performed in
    /// descending order. The sort is unstable so there is no guarantees on the final order when it
    /// comes to ties.
    fn arg_sort(&self, asc: bool) -> Result<Tensor> {
        if !self.device().is_cuda() {
            return self.arg_sort_last_dim(asc);
        }
        if !self.is_contiguous() {
            return Err(candle_core::Error::RequiresContiguous { op: "arg_sort" });
        }
        let last_dim = match self.dims().last() {
            Some(last_dim) => *last_dim,
            None => candle_core::bail!("empty last-dim in arg-sort"),
        };
        // No need for a backward pass for arg sort.
        self.apply_op1_no_bwd(&ArgSort {
            asc,
            last_dim,
            inplace: false,
        })
    }

    /// Sorts the tensor along the last dimension, returns the sorted tensor together with the
    /// sorted indexes.
    ///
    /// If `asc` is `true`, sorting is in ascending order. Otherwise sorting is performed in
    /// descending order. The sort is unstable so there is no guarantees on the final order when it
    /// comes to ties.
    fn sort(&self, asc: bool) -> Result<(Tensor, Tensor)> {
        if !self.device().is_cuda() {
            return self.sort_last_dim(asc);
        }
        if !self.is_contiguous() {
            return Err(candle_core::Error::RequiresContiguous { op: "arg_sort" });
        }
        let last_dim = match self.dims().last() {
            Some(last_dim) => *last_dim,
            None => candle_core::bail!("empty last-dim in arg-sort"),
        };
        // candle's bf16 gather needs sm_80, so bf16 keeps the in-place sort
        if last_dim <= ARGSORT_ROWS_MAX_COLS && self.dtype() != DType::BF16 {
            let indices = self.arg_sort(asc)?;
            return Ok((self.gather(&indices, D::Minus1)?, indices));
        }
        let sorted = self.copy()?;

        let asort = sorted.apply_op1_no_bwd(&ArgSort {
            asc,
            last_dim,
            inplace: true,
        })?;

        Ok((sorted, asort))
    }
}

pub struct TopKOutput {
    pub values: Tensor,
    pub indices: Tensor,
}

pub struct TopKLogitsPackedOutput {
    /// Each row is packed as `[values; indices_as_f32; softmax_denominator; global_max]`.
    pub packed: Tensor,
    pub k: usize,
    pub(super) _workspace: Vec<Tensor>,
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use candle_core::{DType, Device, Result, Tensor};

    use super::ArgSortOp;

    // Distinct keys (a stride coprime to the length) so the unstable sort has one answer; unsigned dtypes shift up.
    #[allow(clippy::cast_precision_loss)]
    fn distinct_rows(rows: usize, cols: usize, dtype: DType) -> Result<Tensor> {
        const STRIDE: usize = 7919;
        let values = (0..rows * cols)
            .map(|i| ((i % cols) * STRIDE % cols + (i / cols) * 3) as f32 - cols as f32 / 2.0)
            .collect::<Vec<_>>();
        let keys = Tensor::from_vec(values, (rows, cols), &Device::Cpu)?;
        let keys = if matches!(dtype, DType::U8 | DType::U32) {
            keys.affine(1.0, (cols / 2) as f64)?
        } else {
            keys
        };
        keys.to_dtype(dtype)
    }

    #[test]
    fn cuda_sort_matches_the_cpu_on_both_kernels() -> Result<()> {
        skip_without_cuda!();
        let device = Device::new_cuda(0)?;
        // Up to 8192 columns the shared-memory rows kernel runs (1500 loops past 1024 threads); past it the global one
        let cases = [
            (3usize, 5usize, DType::F32),
            (1500, 4, DType::F32),
            (8192, 2, DType::F32),
            (8193, 2, DType::F32),
            (20_000, 1, DType::F32),
            (1500, 3, DType::I64),
            (9000, 1, DType::I64),
            (300, 2, DType::F64),
            (9000, 1, DType::F64),
            (9000, 1, DType::U32),
            (200, 3, DType::BF16),
            (200, 3, DType::F16),
            (200, 2, DType::U8),
        ];
        for (cols, rows, dtype) in cases {
            // one extra leading row, narrowed off, puts the CUDA view at a nonzero offset
            let keys = distinct_rows(rows + 1, cols, dtype)?;
            let cpu = keys.narrow(0, 1, rows)?;
            let cuda = keys.to_device(&device)?.narrow(0, 1, rows)?;
            for asc in [true, false] {
                let expected = cpu.arg_sort_last_dim(asc)?.to_vec2::<u32>()?;
                assert_eq!(
                    cuda.arg_sort(asc)?.to_vec2::<u32>()?,
                    expected,
                    "{cols} {dtype:?} {asc}"
                );
                let (values, indices) = cuda.sort(asc)?;
                assert_eq!(
                    indices.to_vec2::<u32>()?,
                    expected,
                    "sort {cols} {dtype:?} {asc}"
                );
                let (expected_values, _) = cpu.sort_last_dim(asc)?;
                assert_eq!(
                    values.to_dtype(DType::F64)?.to_vec2::<f64>()?,
                    expected_values.to_dtype(DType::F64)?.to_vec2::<f64>()?,
                    "sort values {cols} {dtype:?} {asc}"
                );
            }
        }
        Ok(())
    }

    // Three columns pad to four, so the padding has to sort past every value in either direction and dtype.
    #[test]
    fn cuda_sort_pads_a_row_with_negatives() -> Result<()> {
        skip_without_cuda!();
        let device = Device::new_cuda(0)?;
        let row = Tensor::new(&[[-3f32, -1., -2.]], &device)?;
        let (desc, desc_ids) = row.sort(false)?;
        assert_eq!(desc.to_vec2::<f32>()?, [[-1., -2., -3.]]);
        assert_eq!(desc_ids.to_vec2::<u32>()?, [[1, 2, 0]]);
        let (asc, asc_ids) = row.sort(true)?;
        assert_eq!(asc.to_vec2::<f32>()?, [[-3., -2., -1.]]);
        assert_eq!(asc_ids.to_vec2::<u32>()?, [[0, 2, 1]]);
        for dtype in [DType::BF16, DType::F16] {
            let row = Tensor::new(&[[-3f32, 1., -2.]], &device)?.to_dtype(dtype)?;
            assert_eq!(
                row.arg_sort(false)?.to_vec2::<u32>()?,
                [[1, 2, 0]],
                "{dtype:?}"
            );
            let row = Tensor::new(&[[3f32, -1., 2.]], &device)?.to_dtype(dtype)?;
            assert_eq!(
                row.arg_sort(true)?.to_vec2::<u32>()?,
                [[1, 2, 0]],
                "{dtype:?}"
            );
        }
        Ok(())
    }
}
