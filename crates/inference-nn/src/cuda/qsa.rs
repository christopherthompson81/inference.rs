//! QSA's key selection (Qwen3.8-Flash-Next): each query row keeps its top-scoring complete key blocks plus the
//! incomplete tail, as an additive f16 mask over the paged cells.

use inference_tensor::{CpuStorage, CustomOp2, DType, Layout, Result, Shape, Tensor};

/// A query row's position that marks padding: the row keeps cell 0 only, so its output stays finite.
pub const QSA_PADDED_ROW: u32 = u32::MAX;

struct QsaSelect {
    n_kv: usize,
    k: usize,
    ratio: usize,
}

impl QsaSelect {
    fn check(&self, scores: &Layout, positions: &Layout) -> Result<(usize, usize)> {
        let (rows, n_pool) = scores.shape().dims2()?;
        if positions.shape().dims1()? != rows
            || !scores.is_contiguous()
            || !positions.is_contiguous()
        {
            inference_tensor::bail!(
                "QSA selection takes contiguous (rows, pools) scores and (rows,) positions, got {:?} and {:?}",
                scores.shape(),
                positions.shape()
            );
        }
        if self.k == 0 || self.ratio == 0 {
            inference_tensor::bail!("QSA selection needs a nonzero budget and block size");
        }
        Ok((rows, n_pool))
    }

    // The CUDA kernel's semantics, row by row.
    fn row(&self, scores: &[f32], position: u32, out: &mut [half::f16]) {
        let hide = half::f16::NEG_INFINITY;
        if position == QSA_PADDED_ROW {
            out.fill(hide);
            out[0] = half::f16::ZERO;
            return;
        }
        let position = position as usize;
        let complete = (position + 1) / self.ratio;
        let tail = complete * self.ratio;
        let visible = complete.min(scores.len());
        let mut chosen = vec![visible <= self.k; visible];
        if visible > self.k {
            let mut order = (0..visible).collect::<Vec<_>>();
            // NaN ranks lowest, as the kernel's ordering does
            let key = |i: usize| {
                if scores[i].is_nan() {
                    f32::NEG_INFINITY
                } else {
                    scores[i]
                }
            };
            order.sort_by(|&a, &b| key(b).total_cmp(&key(a)).then(a.cmp(&b)));
            for &i in &order[..self.k] {
                chosen[i] = true;
            }
        }
        for (t, cell) in out.iter_mut().enumerate() {
            let picked = t >= tail || chosen.get(t / self.ratio).copied().unwrap_or(false);
            *cell = if t <= position && picked {
                half::f16::ZERO
            } else {
                hide
            };
        }
    }
}

impl CustomOp2 for QsaSelect {
    fn name(&self) -> &'static str {
        "qsa-select-mask"
    }

    fn cpu_fwd(
        &self,
        scores: &CpuStorage,
        scores_l: &Layout,
        positions: &CpuStorage,
        positions_l: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (rows, n_pool) = self.check(scores_l, positions_l)?;
        let (CpuStorage::F32(scores), CpuStorage::U32(positions)) = (scores, positions) else {
            inference_tensor::bail!("QSA selection takes f32 scores and u32 positions");
        };
        let scores = &scores[scores_l.start_offset()..];
        let positions = &positions[positions_l.start_offset()..];
        let mut mask = vec![half::f16::ZERO; rows * self.n_kv];
        for (row, out) in mask.chunks_mut(self.n_kv).enumerate() {
            self.row(
                &scores[row * n_pool..(row + 1) * n_pool],
                positions[row],
                out,
            );
        }
        Ok((CpuStorage::F16(mask), Shape::from((rows, self.n_kv))))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        scores: &inference_tensor::CudaStorage,
        scores_l: &Layout,
        positions: &inference_tensor::CudaStorage,
        positions_l: &Layout,
    ) -> Result<(inference_tensor::CudaStorage, Shape)> {
        use inference_tensor::backend::BackendStorage;
        use inference_tensor::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};

        let (rows, n_pool) = self.check(scores_l, positions_l)?;
        let device = scores.device();
        let stream = device.cuda_stream();
        let scores = scores
            .as_cuda_slice::<f32>()?
            .slice(scores_l.start_offset()..);
        let positions = positions
            .as_cuda_slice::<u32>()?
            .slice(positions_l.start_offset()..);
        let mut mask = unsafe { device.alloc::<half::f16>(rows * self.n_kv)? };
        let (scores_ptr, scores_guard) = scores.device_ptr(&stream);
        let (positions_ptr, positions_guard) = positions.device_ptr(&stream);
        let (mask_ptr, mask_guard) = mask.device_ptr_mut(&stream);
        let int = |value: usize| i32::try_from(value).map_err(inference_tensor::Error::wrap);
        let status = unsafe {
            super::ffi::qsa_select_mask(
                scores_ptr as *const f32,
                positions_ptr as *const u32,
                mask_ptr as *mut core::ffi::c_void,
                int(rows)?,
                int(n_pool)?,
                int(self.n_kv)?,
                int(self.k)?,
                int(self.ratio)?,
                stream.cu_stream() as i64,
            )
        };
        drop(mask_guard);
        drop(positions_guard);
        drop(scores_guard);
        if status != 0 {
            inference_tensor::bail!("qsa_select_mask failed with status {status}");
        }
        Ok((
            inference_tensor::CudaStorage::wrap_cuda_slice(mask, device.clone()),
            Shape::from((rows, self.n_kv)),
        ))
    }
}

/// The `(rows, n_kv)` f16 mask that keeps, for each query row at `positions[row]`, every cell up to it in its tail
/// block and in the `k` complete blocks of `ratio` cells with the highest `scores[row]` (ties to the earlier block).
pub fn qsa_select_mask(
    scores: &Tensor,
    positions: &Tensor,
    n_kv: usize,
    k: usize,
    ratio: usize,
) -> Result<Tensor> {
    if scores.dtype() != DType::F32 || positions.dtype() != DType::U32 {
        inference_tensor::bail!("QSA selection takes f32 scores and u32 positions");
    }
    scores
        .contiguous()?
        .apply_op2_no_bwd(&positions.contiguous()?, &QsaSelect { n_kv, k, ratio })
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
mod tests {
    use super::*;
    use inference_tensor::Device;

    const RATIO: usize = 4;
    const K: usize = 3;
    const N_KV: usize = 64;

    fn kept(mask: &Tensor, row: usize) -> Result<Vec<usize>> {
        Ok(mask
            .get(row)?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?
            .iter()
            .enumerate()
            .filter(|(_, value)| **value == 0.)
            .map(|(t, _)| t)
            .collect())
    }

    #[test]
    fn rows_keep_their_top_blocks_the_tail_and_nothing_ahead() -> Result<()> {
        let n_pool = N_KV / RATIO;
        // block scores 0..n_pool, with blocks 2 and 5 tied at the top
        let mut scores = (0..n_pool).map(|b| b as f32).collect::<Vec<_>>();
        scores[2] = 100.;
        scores[5] = 100.;
        let scores = Tensor::from_vec(scores.repeat(3), (3, n_pool), &Device::Cpu)?;
        // 30 sees 7 complete blocks and a tail of 3; 9 sees 2 (under the budget); the last is padding
        let positions = Tensor::new(&[30u32, 9, QSA_PADDED_ROW], &Device::Cpu)?;
        let mask = qsa_select_mask(&scores, &positions, N_KV, K, RATIO)?;
        let blocks = |bs: &[usize]| {
            bs.iter()
                .flat_map(|b| b * RATIO..(b + 1) * RATIO)
                .collect::<Vec<_>>()
        };
        let mut expected = blocks(&[2, 5, 6]);
        expected.extend(28..=30);
        assert_eq!(kept(&mask, 0)?, expected);
        assert_eq!(kept(&mask, 1)?, (0..=9).collect::<Vec<_>>());
        assert_eq!(kept(&mask, 2)?, vec![0]);
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_selection_matches_the_reference() -> Result<()> {
        let Ok(device) = Device::new_cuda(0) else {
            return Ok(());
        };
        const ROWS: usize = 6;
        const POOLS: usize = 3000;
        const CELLS: usize = POOLS * RATIO + RATIO - 1;
        const BUDGET: usize = 512;
        // coarse scores with many ties, zeros among them, as rectified sums give
        let scores = (0..ROWS * POOLS)
            .map(|i| ((i * 7919) % 37) as f32 * 0.5 - 4.)
            .map(|v: f32| v.max(0.))
            .collect::<Vec<_>>();
        let scores = Tensor::from_vec(scores, (ROWS, POOLS), &Device::Cpu)?;
        let positions = Tensor::new(
            &[0u32, 2050, 2051, 9000, CELLS as u32 - 1, QSA_PADDED_ROW],
            &Device::Cpu,
        )?;
        let expected = qsa_select_mask(&scores, &positions, CELLS, BUDGET, RATIO)?;
        let actual = qsa_select_mask(
            &scores.to_device(&device)?,
            &positions.to_device(&device)?,
            CELLS,
            BUDGET,
            RATIO,
        )?
        .to_device(&Device::Cpu)?;
        for row in 0..ROWS {
            assert_eq!(kept(&actual, row)?, kept(&expected, row)?, "row {row}");
        }
        Ok(())
    }
}
