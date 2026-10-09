use crate::{Result, Tensor};
use rayon::prelude::*;

#[derive(Debug, Clone, Copy)]
struct ArgSort {
    asc: bool,
    last_dim: usize,
}

impl ArgSort {
    fn asort<T: crate::WithDType>(&self, vs: &[T], layout: &crate::Layout) -> Result<Vec<u32>> {
        let vs = match layout.contiguous_offsets() {
            None => crate::bail!("input has to be contiguous"),
            Some((o1, o2)) => &vs[o1..o2],
        };
        #[allow(clippy::uninit_vec)]
        // Safety: indexes are set later in the parallelized section.
        let mut sort_indexes = unsafe {
            let el_count = layout.shape().elem_count();
            let mut v = Vec::with_capacity(el_count);
            v.set_len(el_count);
            v
        };
        if self.asc {
            sort_indexes
                .par_chunks_exact_mut(self.last_dim)
                .zip(vs.par_chunks_exact(self.last_dim))
                .for_each(|(indexes, vs)| {
                    indexes
                        .iter_mut()
                        .enumerate()
                        .for_each(|(i, v)| *v = i as u32);
                    indexes.sort_by(|&i, &j| {
                        vs[i as usize]
                            .partial_cmp(&vs[j as usize])
                            .unwrap_or(std::cmp::Ordering::Greater)
                    })
                });
        } else {
            sort_indexes
                .par_chunks_exact_mut(self.last_dim)
                .zip(vs.par_chunks_exact(self.last_dim))
                .for_each(|(indexes, vs)| {
                    indexes
                        .iter_mut()
                        .enumerate()
                        .for_each(|(i, v)| *v = i as u32);
                    indexes.sort_by(|&j, &i| {
                        vs[i as usize]
                            .partial_cmp(&vs[j as usize])
                            .unwrap_or(std::cmp::Ordering::Greater)
                    })
                });
        }
        Ok(sort_indexes)
    }
}

impl crate::CustomOp1 for ArgSort {
    fn name(&self) -> &'static str {
        "argsort"
    }

    fn cpu_fwd(
        &self,
        storage: &crate::CpuStorage,
        layout: &crate::Layout,
    ) -> Result<(crate::CpuStorage, crate::Shape)> {
        let sort_indexes = match storage {
            crate::CpuStorage::U8(vs) => self.asort(vs, layout)?,
            crate::CpuStorage::U32(vs) => self.asort(vs, layout)?,
            crate::CpuStorage::I16(vs) => self.asort(vs, layout)?,
            crate::CpuStorage::I32(vs) => self.asort(vs, layout)?,
            crate::CpuStorage::I64(vs) => self.asort(vs, layout)?,
            crate::CpuStorage::BF16(vs) => self.asort(vs, layout)?,
            crate::CpuStorage::F16(vs) => self.asort(vs, layout)?,
            crate::CpuStorage::F32(vs) => self.asort(vs, layout)?,
            crate::CpuStorage::F64(vs) => self.asort(vs, layout)?,
            crate::CpuStorage::F8E4M3(vs) => self.asort(vs, layout)?,
            // Dummy types don't support sorting
            crate::CpuStorage::F6E2M3(_) => {
                return Err(
                    crate::Error::UnsupportedDTypeForOp(crate::DType::F6E2M3, "argsort").bt(),
                )
            }
            crate::CpuStorage::F6E3M2(_) => {
                return Err(
                    crate::Error::UnsupportedDTypeForOp(crate::DType::F6E3M2, "argsort").bt(),
                )
            }
            crate::CpuStorage::F4(_) => {
                return Err(crate::Error::UnsupportedDTypeForOp(crate::DType::F4, "argsort").bt())
            }
            crate::CpuStorage::F8E8M0(_) => {
                return Err(
                    crate::Error::UnsupportedDTypeForOp(crate::DType::F8E8M0, "argsort").bt(),
                )
            }
        };
        let sort_indexes = crate::CpuStorage::U32(sort_indexes);
        Ok((sort_indexes, layout.shape().into()))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        storage: &crate::MetalStorage,
        layout: &crate::Layout,
    ) -> Result<(crate::MetalStorage, crate::Shape)> {
        use crate::backend::BackendStorage;
        use crate::DType;

        let name = {
            if self.asc {
                match storage.dtype() {
                    DType::BF16 => "asort_asc_bf16",
                    DType::F16 => "asort_asc_f16",
                    DType::F32 => "asort_asc_f32",
                    DType::F64 => "asort_asc_f64",
                    DType::U8 => "asort_asc_u8",
                    DType::U32 => "asort_asc_u32",
                    DType::I16 => "asort_asc_i16",
                    DType::I32 => "asort_asc_i32",
                    DType::I64 => "asort_asc_i64",
                    DType::F8E4M3 => crate::bail!("Metal device does not yet support F8E4M3."),
                    DType::F6E2M3 | DType::F6E3M2 | DType::F4 | DType::F8E8M0 => {
                        return Err(
                            crate::Error::UnsupportedDTypeForOp(storage.dtype(), "argsort").bt(),
                        )
                    }
                }
            } else {
                match storage.dtype() {
                    DType::BF16 => "asort_desc_bf16",
                    DType::F16 => "asort_desc_f16",
                    DType::F32 => "asort_desc_f32",
                    DType::F64 => "asort_desc_f64",
                    DType::U8 => "asort_desc_u8",
                    DType::U32 => "asort_desc_u32",
                    DType::I16 => "asort_desc_i16",
                    DType::I32 => "asort_desc_i32",
                    DType::I64 => "asort_desc_i64",
                    DType::F8E4M3 => crate::bail!("Metal device does not yet support F8E4M3."),
                    DType::F6E2M3 | DType::F6E3M2 | DType::F4 | DType::F8E8M0 => {
                        return Err(
                            crate::Error::UnsupportedDTypeForOp(storage.dtype(), "argsort").bt(),
                        )
                    }
                }
            }
        };
        let device = storage.device();
        let kernels = device.kernels();
        let command_encoder = device.command_encoder()?;
        let el = layout.shape().elem_count();
        let ncols = self.last_dim;
        let nrows = el / ncols;
        let src = crate::metal_backend::buffer_o(storage.buffer(), layout, storage.dtype());
        let dst = device
            .new_buffer_builder()
            .with_size_for(el, DType::U32)
            .with_label("asort")
            .build()?;
        let ncols_pad = ncols.next_power_of_two();
        if ncols_pad > METAL_MAX_THREADGROUP {
            metal_merge_arg_sort(
                device,
                &command_encoder,
                storage.dtype(),
                nrows,
                ncols,
                src,
                &dst,
            )?;
            let dst = crate::MetalStorage::new(dst, device.clone(), el, DType::U32);
            return Ok((dst, layout.shape().clone()));
        }
        candle_metal_kernels::call_arg_sort(
            device.metal_device(),
            &command_encoder,
            kernels,
            name,
            nrows,
            ncols,
            ncols_pad,
            src,
            &dst,
        )
        .map_err(crate::Error::wrap)?;
        let dst = crate::MetalStorage::new(dst, device.clone(), el, DType::U32);
        Ok((dst, layout.shape().clone()))
    }
}

// Threads in a Metal threadgroup; candle's bitonic argsort runs a row in one, a thread per padded column
#[cfg(feature = "metal")]
const METAL_MAX_THREADGROUP: usize = 1024;
#[cfg(feature = "metal")]
const MLX_SORT_TN: usize = 8;

// MLX's ascending merge sort, as candle's `call_mlx_arg_sort`, whose multi-block path always merges as float32 and
// copies the indices out with the key dtype's width.
#[cfg(feature = "metal")]
fn metal_merge_arg_sort(
    device: &crate::MetalDevice,
    ep: &candle_metal_kernels::metal::CommandsGuard<'_>,
    dtype: crate::DType,
    nrows: usize,
    ncols: usize,
    src: candle_metal_kernels::BufferOffset,
    dst: &candle_metal_kernels::metal::Buffer,
) -> Result<()> {
    use crate::DType;
    use candle_metal_kernels::source::Source;
    use candle_metal_kernels::utils::EncoderProvider;
    use candle_metal_kernels::{set_params, Output, RESOURCE_OPTIONS};
    use objc2_metal::MTLSize;

    let (dtype_str, mlx_dtype) = match dtype {
        DType::U8 => ("uint8", candle_metal_kernels::DType::U8),
        DType::U32 => ("uint32", candle_metal_kernels::DType::U32),
        DType::I64 => ("int64", candle_metal_kernels::DType::I64),
        DType::F16 => ("float16", candle_metal_kernels::DType::F16),
        DType::BF16 => ("bfloat16", candle_metal_kernels::DType::BF16),
        DType::F32 => ("float32", candle_metal_kernels::DType::F32),
        other => {
            crate::bail!(
                "metal argsort past {METAL_MAX_THREADGROUP} columns does not support {other:?}"
            )
        }
    };
    let metal = device.metal_device();
    let kernels = device.kernels();
    let tn = MLX_SORT_TN;
    let bn = match ncols.div_ceil(tn) {
        257.. if dtype.size_in_bytes() <= 4 => 512,
        129.. => 256,
        _ => 128,
    };
    let nblocks = ncols.div_ceil(bn * tn);
    if nblocks == 1 {
        return candle_metal_kernels::call_mlx_arg_sort(
            metal, ep, kernels, mlx_dtype, nrows, ncols, src, dst,
        )
        .map_err(crate::Error::wrap);
    }
    let mut passes = 0;
    while (1 << passes) < nblocks {
        passes += 1;
    }
    let el = nrows * ncols;
    let new_buffer = |bytes: usize| {
        metal
            .new_buffer(bytes, RESOURCE_OPTIONS)
            .map_err(crate::Error::wrap)
    };
    let vals = [
        new_buffer(el * dtype.size_in_bytes())?,
        new_buffer(el * dtype.size_in_bytes())?,
    ];
    // pass p merges from idxs[p % 2] into idxs[(p + 1) % 2], so the last pass lands in dst
    let scratch = new_buffer(el * DType::U32.size_in_bytes())?;
    let idxs = if passes % 2 == 0 {
        [dst, &scratch]
    } else {
        [&scratch, dst]
    };
    let partitions = new_buffer(nrows * (nblocks + 1) * DType::U32.size_in_bytes())?;

    let guard = ep.encoder();
    let encoder: &candle_metal_kernels::metal::ComputeCommandEncoder = guard.as_ref();
    let load = |name: String| {
        kernels
            .load_pipeline(metal, Source::MlxSort, name)
            .map_err(crate::Error::wrap)
    };
    let grid = |width, height| MTLSize {
        width,
        height,
        depth: 1,
    };

    encoder.set_compute_pipeline_state(&load(format!(
        "sort_mbsort_{dtype_str}_uint32_bn{bn}_tn{tn}"
    ))?);
    set_params!(
        encoder,
        (
            &src,
            Output::new(&vals[0]),
            Output::new(idxs[0]),
            ncols as i32,
            1i32,
            1i32,
            nrows as i32,
            ncols as i64
        )
    );
    encoder.dispatch_thread_groups(grid(nblocks, nrows), grid(bn, 1));

    let partition = load(format!("partition_mbsort_{dtype_str}_uint32_bn{bn}_tn{tn}"))?;
    let merge = load(format!("merge_mbsort_{dtype_str}_uint32_bn{bn}_tn{tn}"))?;
    let partition_threads = (nblocks + 1).min(METAL_MAX_THREADGROUP);
    for pass in 0..passes {
        let (from, to) = (pass % 2, (pass + 1) % 2);
        let merge_tiles = 2i32 << pass;
        encoder.set_compute_pipeline_state(&partition);
        set_params!(
            encoder,
            (
                Output::new(&partitions),
                &vals[from],
                idxs[from],
                ncols as i32,
                merge_tiles,
                nblocks as i32
            )
        );
        encoder.dispatch_thread_groups(grid(1, nrows), grid(partition_threads, 1));
        encoder.set_compute_pipeline_state(&merge);
        set_params!(
            encoder,
            (
                &partitions,
                &vals[from],
                idxs[from],
                Output::new(&vals[to]),
                Output::new(idxs[to]),
                ncols as i32,
                merge_tiles,
                nblocks as i32
            )
        );
        encoder.dispatch_thread_groups(grid(nblocks, nrows), grid(bn, 1));
    }
    Ok(())
}

impl Tensor {
    /// Returns the indices that sort the tensor along the last dimension.
    ///
    /// If `asc` is `true`, sorting is in ascending order. Otherwise sorting is performed in
    /// descending order. The sort is unstable so there is no guarantees on the final order when it
    /// comes to ties.
    pub fn arg_sort_last_dim(&self, asc: bool) -> Result<Tensor> {
        if !self.is_contiguous() {
            return Err(crate::Error::RequiresContiguous {
                op: "arg_sort_last_dim",
            });
        }
        let last_dim = match self.dims().last() {
            None => crate::bail!("empty last-dim in arg-sort"),
            Some(last_dim) => *last_dim,
        };
        #[cfg(feature = "metal")]
        if !asc && self.device().is_metal() && last_dim.next_power_of_two() > METAL_MAX_THREADGROUP
        {
            let ascending = self.apply_op1_no_bwd(&ArgSort {
                asc: true,
                last_dim,
            })?;
            let reversed = Tensor::from_iter((0..last_dim as u32).rev(), self.device())?;
            return ascending.index_select(&reversed, crate::D::Minus1);
        }
        // No need for a backward pass for arg sort.
        self.apply_op1_no_bwd(&ArgSort { asc, last_dim })
    }

    /// Sorts the tensor along the last dimension, returns the sorted tensor together with the
    /// sorted indexes.
    ///
    /// If `asc` is `true`, sorting is in ascending order. Otherwise sorting is performed in
    /// descending order. The sort is unstable so there is no guarantees on the final order when it
    /// comes to ties.
    pub fn sort_last_dim(&self, asc: bool) -> Result<(Tensor, Tensor)> {
        if !self.is_contiguous() {
            return Err(crate::Error::RequiresContiguous {
                op: "sort_last_dim",
            });
        }
        let asort = self.arg_sort_last_dim(asc)?;
        let sorted = self.gather(&asort, crate::D::Minus1)?;
        Ok((sorted, asort))
    }
}

#[cfg(all(test, feature = "metal"))]
mod metal_tests {
    use crate::{DType, Device, Result, Tensor};

    #[test]
    fn metal_arg_sort_last_dim_matches_cpu() -> Result<()> {
        let metal = Device::new_metal(0)?;
        // 1025 is the first bitonic overflow, 4096 one MLX block, 10_000 and 70_000 even and odd merge pass counts
        for &ncols in &[7usize, 1024, 1025, 4096, 10_000, 70_000] {
            let vals: Vec<f32> = (0..2 * ncols)
                .map(|i| ((i as u64 * 2_654_435_761) % 251) as f32)
                .collect();
            let keys = Tensor::from_vec(vals, (2, ncols), &Device::Cpu)?;
            for dtype in [
                DType::F32,
                DType::U32,
                DType::BF16,
                DType::F16,
                DType::U8,
                DType::I64,
            ] {
                let cpu = keys.to_dtype(dtype)?;
                let gpu = cpu.to_device(&metal)?;
                for asc in [true, false] {
                    let want = cpu.arg_sort_last_dim(asc)?;
                    let got = gpu.arg_sort_last_dim(asc)?.to_device(&Device::Cpu)?;
                    // ties may order differently, so compare the keys each permutation gathers
                    let sorted =
                        |idx: &Tensor| cpu.gather(idx, 1)?.to_dtype(DType::F32)?.to_vec2::<f32>();
                    assert_eq!(
                        sorted(&got)?,
                        sorted(&want)?,
                        "ncols {ncols} {dtype:?} asc {asc}"
                    );
                }
            }
        }
        Ok(())
    }
}
