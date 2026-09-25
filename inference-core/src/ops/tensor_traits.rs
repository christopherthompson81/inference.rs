use super::*;

pub trait TopKLastDimOp {
    /// Topk in the last dim. `values` retains a gradient but `indices` has none w.r.t self.
    /// This expects a contiguous tensor.
    /// Note: this implements torch.topk with sorted=True.
    fn topk(&self, topk: usize) -> Result<TopKOutput>;

    /// Topk in the last dim. `values` retains a gradient but `indices` has none w.r.t self.
    /// This expects a contiguous tensor.
    /// Note: this implements torch.topk with sorted=False.
    fn topk_unsorted(&self, topk: usize) -> Result<TopKOutput>;
}

impl TopKLastDimOp for Tensor {
    fn topk(&self, topk: usize) -> Result<TopKOutput> {
        // Use optimized parallel topk kernel on CUDA
        // Single kernel call, no post-processing overhead
        #[cfg(feature = "cuda")]
        if self.device().is_cuda() {
            return cuda_topk(self, topk);
        }

        // Fallback: full sort (CPU or non-CUDA)
        let (values, sorted_indices) = self.sort_last_dim(false)?;

        let topk_indices = sorted_indices.narrow(D::Minus1, 0, topk)?.contiguous()?;
        let topk_values = values.narrow(D::Minus1, 0, topk)?.contiguous()?;
        Ok(TopKOutput {
            values: topk_values,
            indices: topk_indices,
        })
    }

    fn topk_unsorted(&self, topk: usize) -> Result<TopKOutput> {
        // Sorted descending
        let TopKOutput { values, indices } = self.topk(topk)?;
        // Reorder the indices ascending
        #[cfg(feature = "cuda")]
        let reorder_indices = indices.arg_sort(true)?;
        #[cfg(not(feature = "cuda"))]
        let reorder_indices = indices.arg_sort_last_dim(true)?;
        let topk_indices_unsorted = indices
            .to_dtype(DType::F32)?
            .gather(&reorder_indices, D::Minus1)?
            .to_dtype(DType::U32)?;
        let topk_values_unsorted = values.gather(&reorder_indices, D::Minus1)?;
        Ok(TopKOutput {
            values: topk_values_unsorted,
            indices: topk_indices_unsorted,
        })
    }
}

pub trait RepeatInterleaveOp {
    fn repeat_interleave<D: Dim>(&self, repeats: usize, dim: D) -> Result<Tensor>;
    fn repeat_interleave_flat(&self, repeats: Vec<u32>) -> Result<Tensor>;
}

impl RepeatInterleaveOp for Tensor {
    fn repeat_interleave<D: Dim>(&self, repeats: usize, dim: D) -> Result<Tensor> {
        let dim = dim.to_index(self.shape(), "repeat_interleave")?;
        let dim_elements = self.dim(dim)?;
        // For metal
        assert!(self.dtype().is_float());
        #[allow(clippy::cast_possible_truncation)]
        let indices = Tensor::new(
            (0..dim_elements)
                .flat_map(|i| vec![i as u32; repeats])
                .collect::<Vec<_>>(),
            self.device(),
        )?;
        self.index_select(&indices, dim)
    }

    fn repeat_interleave_flat(&self, repeats: Vec<u32>) -> Result<Tensor> {
        let xs = self.flatten_all()?;
        if repeats.len() != xs.dim(0)? {
            candle_core::bail!(
                "repeats ({}) must match flattened self length ({})",
                repeats.len(),
                xs.dim(0)?
            );
        }
        #[allow(clippy::cast_possible_truncation)]
        let indices = Tensor::new(
            (0..xs.dim(0)?)
                .flat_map(|i| vec![i as u32; repeats[i] as usize])
                .collect::<Vec<_>>(),
            xs.device(),
        )?;
        xs.index_select(&indices, 0)
    }
}

pub trait SplitOp {
    fn split<D: Dim>(&self, splits: &[usize], dim: D) -> Result<Vec<Tensor>>;
}

impl SplitOp for Tensor {
    fn split<D: Dim>(&self, splits: &[usize], dim: D) -> Result<Vec<Tensor>> {
        let dim = dim.to_index(self.shape(), "split")?;
        let mut split_res = Vec::new();
        let mut index = 0;
        for split in splits {
            split_res.push(self.narrow(dim, index, *split)?);
            index += *split;
        }
        Ok(split_res)
    }
}

#[allow(dead_code)]
pub trait BincountOp {
    fn bincount(&self, minlength: u32) -> Result<Vec<u32>>;
}

#[allow(dead_code)]
fn bincount(values: &[u32], minlength: u32) -> Vec<u32> {
    // let max_val = values.iter().max().copied().unwrap_or(0);
    // let result_len = (max_val + 1).max(minlength);
    // values.iter().fold(
    //     // Start with a histogram vector of zeros.
    //     vec![0u32; result_len as usize],
    //     // For each value, update the histogram.
    //     |mut histogram, &value| {
    //         histogram[value as usize] += 1;
    //         histogram
    //     },
    // )

    use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

    // Early return if there are no values.
    if values.is_empty() {
        return vec![0u32; minlength as usize];
    }

    // Compute the maximum value in parallel.
    // SAFETY: We just checked that values is nonempty above, so max() will return Some.
    // Using expect() for clearer error message if this invariant is somehow violated.
    let max_val = *values
        .par_iter()
        .max()
        .expect("values should be non-empty after empty check");

    // The histogram length must cover all observed values as well as `minlength`.
    let result_len = (max_val + 1).max(minlength) as usize;

    // Build per-thread histograms in parallel.
    // We use unsafe indexing to eliminate bounds checks in the inner loop.
    values
        .par_iter()
        .fold(
            || vec![0u32; result_len],
            |mut local_hist, &v| {
                // SAFETY: v is guaranteed to be <= max_val, so it is in bounds.
                unsafe {
                    *local_hist.get_unchecked_mut(v as usize) += 1;
                }
                local_hist
            },
        )
        // Merge the per-thread histograms in parallel.
        .reduce(
            || vec![0u32; result_len],
            |mut global_hist, local_hist| {
                for i in 0..result_len {
                    // SAFETY: we know local histogram is at least result_len, as is global_hist
                    unsafe {
                        *global_hist.get_unchecked_mut(i) += local_hist.get_unchecked(i);
                    }
                }
                global_hist
            },
        )
}

#[allow(dead_code)]
impl BincountOp for Tensor {
    fn bincount(&self, minlength: u32) -> Result<Vec<u32>> {
        let values = self.to_vec1::<u32>()?;

        Ok(bincount(&values, minlength))
    }
}

// https://github.com/mokeyish/candle-ext/blob/ca4547c803469bd51c00ce5eda2f18dd249c8f10/src/triangular.rs#L21
pub fn apply_triangular(xs: &Tensor, diagonal: isize, upper: bool) -> Result<Tensor> {
    let device = xs.device();
    let (l, s) = xs.dims2()?;
    let mut xs_tri = vec![];
    for i in 0..l as isize {
        for j in 0..s as isize {
            let cond = if upper {
                i + diagonal > j
            } else {
                i + diagonal < j
            };
            xs_tri.push(if cond { 0u8 } else { 1u8 });
        }
    }
    xs * Tensor::from_vec(xs_tri, (l, s), device)?.to_dtype(xs.dtype())?
}
