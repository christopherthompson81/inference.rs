//! QSA's key selection (Qwen3.8-Flash-Next): each query row keeps its top-scoring complete key blocks plus the
//! incomplete tail, as an additive f16 mask over the paged cells or as the list of kept cells, which
//! [`qsa_gather_cells`] copies out of a paged cache.

use inference_tensor::{CpuStorage, CustomOp2, CustomOp3, DType, Layout, Result, Shape, Tensor};

/// A query row's position that marks padding: the row keeps cell 0 only, so its output stays finite.
pub const QSA_PADDED_ROW: u32 = u32::MAX;

#[derive(Clone, Copy)]
enum QsaOutput {
    // an additive f16 mask over this many cells
    Mask(usize),
    // the kept positions, in this many u32 slots
    Cells(usize),
}

struct QsaSelect {
    output: QsaOutput,
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

    // The CUDA kernels' selection: whether each visible block is kept, or `None` when every cell up to `position` is.
    fn chosen(&self, scores: &[f32], position: usize) -> Option<Vec<bool>> {
        let visible = ((position + 1) / self.ratio).min(scores.len());
        if visible <= self.k {
            return None;
        }
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
        let mut chosen = vec![false; visible];
        for &i in &order[..self.k] {
            chosen[i] = true;
        }
        Some(chosen)
    }

    // The mask kernel's semantics, row by row.
    fn row(&self, scores: &[f32], position: u32, out: &mut [half::f16]) {
        let hide = half::f16::NEG_INFINITY;
        if position == QSA_PADDED_ROW {
            out.fill(hide);
            out[0] = half::f16::ZERO;
            return;
        }
        let position = position as usize;
        let tail = (position + 1) / self.ratio * self.ratio;
        let chosen = self.chosen(scores, position);
        for (t, cell) in out.iter_mut().enumerate() {
            let picked = t >= tail
                || chosen
                    .as_ref()
                    .is_none_or(|c| c.get(t / self.ratio).copied().unwrap_or(false));
            *cell = if t <= position && picked {
                half::f16::ZERO
            } else {
                hide
            };
        }
    }

    // The cells kernel's semantics, row by row.
    fn cells_row(&self, scores: &[f32], position: u32, out: &mut [u32]) {
        out.fill(0);
        if position == QSA_PADDED_ROW {
            return;
        }
        let position = position as usize;
        let tail = (position + 1) / self.ratio * self.ratio;
        let kept = match self.chosen(scores, position) {
            None => (0..=position).collect::<Vec<_>>(),
            Some(chosen) => chosen
                .iter()
                .enumerate()
                .filter(|(_, c)| **c)
                .flat_map(|(b, _)| b * self.ratio..(b + 1) * self.ratio)
                .chain(tail..=position)
                .collect(),
        };
        for (slot, &t) in out.iter_mut().zip(&kept) {
            *slot = u32::try_from(t).expect("kept cells lie at or before a u32 position");
        }
        let first = out[0];
        out[kept.len()..].fill(first);
    }
}

impl CustomOp2 for QsaSelect {
    fn name(&self) -> &'static str {
        match self.output {
            QsaOutput::Mask(_) => "qsa-select-mask",
            QsaOutput::Cells(_) => "qsa-select-cells",
        }
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
        let row_scores = |row: usize| &scores[row * n_pool..(row + 1) * n_pool];
        Ok(match self.output {
            QsaOutput::Mask(n_kv) => {
                let mut mask = vec![half::f16::ZERO; rows * n_kv];
                for (row, out) in mask.chunks_mut(n_kv).enumerate() {
                    self.row(row_scores(row), positions[row], out);
                }
                (CpuStorage::F16(mask), Shape::from((rows, n_kv)))
            }
            QsaOutput::Cells(width) => {
                let mut cells = vec![0u32; rows * width];
                for (row, out) in cells.chunks_mut(width).enumerate() {
                    self.cells_row(row_scores(row), positions[row], out);
                }
                (CpuStorage::U32(cells), Shape::from((rows, width)))
            }
        })
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
        let (scores_ptr, scores_guard) = scores.device_ptr(&stream);
        let (positions_ptr, positions_guard) = positions.device_ptr(&stream);
        let int = |value: usize| i32::try_from(value).map_err(inference_tensor::Error::wrap);
        let cu_stream = stream.cu_stream() as i64;
        let (status, storage, shape) = match self.output {
            QsaOutput::Mask(n_kv) => {
                let mut mask = unsafe { device.alloc::<half::f16>(rows * n_kv)? };
                let (mask_ptr, mask_guard) = mask.device_ptr_mut(&stream);
                let status = unsafe {
                    super::ffi::qsa_select_mask(
                        scores_ptr as *const f32,
                        positions_ptr as *const u32,
                        mask_ptr as *mut core::ffi::c_void,
                        int(rows)?,
                        int(n_pool)?,
                        int(n_kv)?,
                        int(self.k)?,
                        int(self.ratio)?,
                        cu_stream,
                    )
                };
                drop(mask_guard);
                (
                    status,
                    inference_tensor::CudaStorage::wrap_cuda_slice(mask, device.clone()),
                    Shape::from((rows, n_kv)),
                )
            }
            QsaOutput::Cells(width) => {
                let mut cells = unsafe { device.alloc::<u32>(rows * width)? };
                let (cells_ptr, cells_guard) = cells.device_ptr_mut(&stream);
                let status = unsafe {
                    super::ffi::qsa_select_cells(
                        scores_ptr as *const f32,
                        positions_ptr as *const u32,
                        cells_ptr as *mut u32,
                        int(rows)?,
                        int(n_pool)?,
                        int(width)?,
                        int(self.k)?,
                        int(self.ratio)?,
                        cu_stream,
                    )
                };
                drop(cells_guard);
                (
                    status,
                    inference_tensor::CudaStorage::wrap_cuda_slice(cells, device.clone()),
                    Shape::from((rows, width)),
                )
            }
        };
        drop(positions_guard);
        drop(scores_guard);
        if status != 0 {
            inference_tensor::bail!("{} failed with status {status}", self.name());
        }
        Ok((storage, shape))
    }
}

fn select(scores: &Tensor, positions: &Tensor, op: QsaSelect) -> Result<Tensor> {
    if scores.dtype() != DType::F32 || positions.dtype() != DType::U32 {
        inference_tensor::bail!("QSA selection takes f32 scores and u32 positions");
    }
    scores
        .contiguous()?
        .apply_op2_no_bwd(&positions.contiguous()?, &op)
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
    let output = QsaOutput::Mask(n_kv);
    select(scores, positions, QsaSelect { output, k, ratio })
}

/// The slots [`qsa_select_cells`] fills per row: `k` blocks of `ratio` cells and the longest tail.
pub fn qsa_cells_width(k: usize, ratio: usize) -> usize {
    k * ratio + ratio - 1
}

/// [`qsa_select_mask`]'s selection as `(rows, qsa_cells_width(k, ratio))` u32 positions: the chosen blocks' cells in
/// order, then the tail, with the slots past the last kept cell repeating the first (a padding row keeps cell 0).
pub fn qsa_select_cells(
    scores: &Tensor,
    positions: &Tensor,
    k: usize,
    ratio: usize,
) -> Result<Tensor> {
    let output = QsaOutput::Cells(qsa_cells_width(k, ratio));
    select(scores, positions, QsaSelect { output, k, ratio })
}

// (pages, heads, block_size, row) of a paged HND cache
type CacheDims = (usize, usize, usize, usize);

struct QsaGather {
    blocks_per_row: usize,
}

impl QsaGather {
    // the cache's dims, the table's width, and (rows, width) of the cells
    fn check(
        &self,
        cache: &Layout,
        table: &Layout,
        cells: &Layout,
    ) -> Result<(CacheDims, usize, usize, usize)> {
        let dims = cache.shape().dims4()?;
        let (table_rows, table_width) = table.shape().dims2()?;
        let (rows, width) = cells.shape().dims2()?;
        if table_rows != rows
            || !cache.is_contiguous()
            || !table.is_contiguous()
            || !cells.is_contiguous()
            || self.blocks_per_row * dims.2 < width
        {
            inference_tensor::bail!(
                "QSA gather takes a contiguous paged cache, a (rows, blocks) table and (rows, cells) cells, got {:?}, \
                 {:?} and {:?}",
                cache.shape(),
                table.shape(),
                cells.shape()
            );
        }
        Ok((dims, table_width, rows, width))
    }

    fn out_shape(
        &self,
        (_, heads, block_size, row): (usize, usize, usize, usize),
        rows: usize,
    ) -> Shape {
        Shape::from((rows * self.blocks_per_row, heads, block_size, row))
    }
}

impl CustomOp3 for QsaGather {
    fn name(&self) -> &'static str {
        "qsa-gather-cells"
    }

    fn cpu_fwd(
        &self,
        cache: &CpuStorage,
        cache_l: &Layout,
        table: &CpuStorage,
        table_l: &Layout,
        cells: &CpuStorage,
        cells_l: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (dims, table_width, rows, width) = self.check(cache_l, table_l, cells_l)?;
        let (_, heads, block_size, row) = dims;
        let (CpuStorage::U32(table), CpuStorage::U32(cells)) = (table, cells) else {
            inference_tensor::bail!("QSA gather takes u32 block tables and cells");
        };
        let table = &table[table_l.start_offset()..];
        let cells = &cells[cells_l.start_offset()..];
        let slots = self.blocks_per_row * block_size;
        // (source, destination) element offsets of every copied cache row
        let copies = (0..rows)
            .flat_map(|r| (0..slots).flat_map(move |slot| (0..heads).map(move |h| (r, slot, h))))
            .map(|(r, slot, h)| {
                let position = cells[r * width + if slot < width { slot } else { 0 }] as usize;
                let page = table[r * table_width + position / block_size] as usize;
                let src = ((page * heads + h) * block_size + position % block_size) * row;
                let out_page = r * self.blocks_per_row + slot / block_size;
                let dst = ((out_page * heads + h) * block_size + slot % block_size) * row;
                (src, dst)
            })
            .collect::<Vec<_>>();
        fn gather<T: Copy + Default>(
            src: &[T],
            copies: &[(usize, usize)],
            row: usize,
            len: usize,
        ) -> Vec<T> {
            let mut out = vec![T::default(); len];
            for &(s, d) in copies {
                out[d..d + row].copy_from_slice(&src[s..s + row]);
            }
            out
        }
        let len = self.out_shape(dims, rows).elem_count();
        let start = cache_l.start_offset();
        let out = match cache {
            CpuStorage::BF16(c) => CpuStorage::BF16(gather(&c[start..], &copies, row, len)),
            CpuStorage::F16(c) => CpuStorage::F16(gather(&c[start..], &copies, row, len)),
            CpuStorage::F32(c) => CpuStorage::F32(gather(&c[start..], &copies, row, len)),
            CpuStorage::U8(c) => CpuStorage::U8(gather(&c[start..], &copies, row, len)),
            _ => inference_tensor::bail!("QSA gather does not support this cache dtype"),
        };
        Ok((out, self.out_shape(dims, rows)))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        cache: &inference_tensor::CudaStorage,
        cache_l: &Layout,
        table: &inference_tensor::CudaStorage,
        table_l: &Layout,
        cells: &inference_tensor::CudaStorage,
        cells_l: &Layout,
    ) -> Result<(inference_tensor::CudaStorage, Shape)> {
        use inference_tensor::backend::BackendStorage;
        use inference_tensor::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};

        let (dims, table_width, rows, width) = self.check(cache_l, table_l, cells_l)?;
        let (_, heads, block_size, row) = dims;
        let shape = self.out_shape(dims, rows);
        let device = cache.device();
        let stream = device.cuda_stream();
        let table = table
            .as_cuda_slice::<u32>()?
            .slice(table_l.start_offset()..);
        let cells = cells
            .as_cuda_slice::<u32>()?
            .slice(cells_l.start_offset()..);
        let (table_ptr, table_guard) = table.device_ptr(&stream);
        let (cells_ptr, cells_guard) = cells.device_ptr(&stream);
        let int = |value: usize| i32::try_from(value).map_err(inference_tensor::Error::wrap);
        macro_rules! launch {
            ($ty:ty) => {{
                let src = cache
                    .as_cuda_slice::<$ty>()?
                    .slice(cache_l.start_offset()..);
                let mut out = unsafe { device.alloc::<$ty>(shape.elem_count())? };
                let (src_ptr, src_guard) = src.device_ptr(&stream);
                let (out_ptr, out_guard) = out.device_ptr_mut(&stream);
                let status = unsafe {
                    super::ffi::qsa_gather_cells(
                        src_ptr as *const core::ffi::c_void,
                        table_ptr as *const u32,
                        cells_ptr as *const u32,
                        out_ptr as *mut core::ffi::c_void,
                        int(rows)?,
                        int(width)?,
                        int(table_width)?,
                        int(self.blocks_per_row)?,
                        int(heads)?,
                        int(block_size)?,
                        int(row * std::mem::size_of::<$ty>())?,
                        stream.cu_stream() as i64,
                    )
                };
                drop(out_guard);
                drop(src_guard);
                if status != 0 {
                    inference_tensor::bail!("qsa_gather_cells failed with status {status}");
                }
                inference_tensor::CudaStorage::wrap_cuda_slice(out, device.clone())
            }};
        }
        let out = match cache.dtype() {
            DType::BF16 => launch!(half::bf16),
            DType::F16 => launch!(half::f16),
            DType::F32 => launch!(f32),
            DType::U8 => launch!(u8),
            DType::F8E4M3 => launch!(float8::F8E4M3),
            dtype => inference_tensor::bail!("QSA gather does not support {dtype:?} caches"),
        };
        drop(cells_guard);
        drop(table_guard);
        Ok((out, shape))
    }
}

/// Copies each row's `cells` (from [`qsa_select_cells`]) out of a paged HND cache `(pages, heads, block_size, row)`
/// through its `table` row, into `blocks_per_row` pages per row of a compact cache in the same layout: row `r`'s
/// cells fill pages `r * blocks_per_row..`, and the slots past the cells repeat the row's first one.
pub fn qsa_gather_cells(
    cache: &Tensor,
    table: &Tensor,
    cells: &Tensor,
    blocks_per_row: usize,
) -> Result<Tensor> {
    if table.dtype() != DType::U32 || cells.dtype() != DType::U32 {
        inference_tensor::bail!("QSA gather takes u32 block tables and cells");
    }
    cache.apply_op3_no_bwd(
        &table.contiguous()?,
        &cells.contiguous()?,
        &QsaGather { blocks_per_row },
    )
}

struct QsaDecodePlan {
    block_size: usize,
    ratio: usize,
    k: usize,
    pools: usize,
}

impl QsaDecodePlan {
    fn width(&self) -> usize {
        self.ratio + 2 + self.pools
    }

    // (rows, table width)
    fn check(&self, table: &Layout, context_lens: &Layout) -> Result<(usize, usize)> {
        let (rows, table_width) = table.shape().dims2()?;
        if context_lens.shape().dims1()? != rows
            || !table.is_contiguous()
            || !context_lens.is_contiguous()
            || self.ratio == 0
            || !self.block_size.is_multiple_of(self.ratio)
        {
            inference_tensor::bail!(
                "a QSA decode plan takes a contiguous (rows, blocks) table and (rows,) context lens, with blocks of \
                 {} tiling {}-cell pages, got {:?} and {:?}",
                self.ratio,
                self.block_size,
                table.shape(),
                context_lens.shape()
            );
        }
        Ok((rows, table_width))
    }

    // The kernel's semantics for one row.
    #[allow(clippy::cast_possible_truncation)]
    fn row(&self, table: &[u32], seen: u32, out: &mut [u32]) {
        let (bs, ratio) = (self.block_size as u32, self.ratio as u32);
        let cell = |t: u32| table[(t / bs) as usize] * bs + t % bs;
        let position = seen - 1;
        let first = position / ratio * ratio;
        for (m, slot) in out[..self.ratio].iter_mut().enumerate() {
            *slot = cell(first + m as u32);
        }
        out[self.ratio] = position;
        out[self.ratio + 1] = if seen / ratio <= self.k as u32 {
            seen
        } else {
            self.k as u32 * ratio + seen % ratio
        };
        let complete = seen / ratio;
        for (j, slot) in out[self.ratio + 2..].iter_mut().enumerate() {
            let j = j as u32;
            *slot = cell(if j < complete {
                j * ratio + ratio - 1
            } else {
                ratio - 1
            });
        }
    }
}

impl CustomOp2 for QsaDecodePlan {
    fn name(&self) -> &'static str {
        "qsa-decode-plan"
    }

    fn cpu_fwd(
        &self,
        table: &CpuStorage,
        table_l: &Layout,
        context_lens: &CpuStorage,
        context_lens_l: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (rows, table_width) = self.check(table_l, context_lens_l)?;
        let (CpuStorage::U32(table), CpuStorage::U32(context_lens)) = (table, context_lens) else {
            inference_tensor::bail!("a QSA decode plan takes u32 block tables and context lens");
        };
        let table = &table[table_l.start_offset()..];
        let context_lens = &context_lens[context_lens_l.start_offset()..];
        let width = self.width();
        let mut plan = vec![0u32; rows * width];
        for (row, out) in plan.chunks_mut(width).enumerate() {
            let row_table = &table[row * table_width..(row + 1) * table_width];
            self.row(row_table, context_lens[row], out);
        }
        Ok((CpuStorage::U32(plan), Shape::from((rows, width))))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        table: &inference_tensor::CudaStorage,
        table_l: &Layout,
        context_lens: &inference_tensor::CudaStorage,
        context_lens_l: &Layout,
    ) -> Result<(inference_tensor::CudaStorage, Shape)> {
        use inference_tensor::backend::BackendStorage;
        use inference_tensor::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};

        let (rows, table_width) = self.check(table_l, context_lens_l)?;
        let width = self.width();
        let device = table.device();
        let stream = device.cuda_stream();
        let table = table
            .as_cuda_slice::<u32>()?
            .slice(table_l.start_offset()..);
        let context_lens = context_lens
            .as_cuda_slice::<u32>()?
            .slice(context_lens_l.start_offset()..);
        let mut plan = unsafe { device.alloc::<u32>(rows * width)? };
        let int = |value: usize| i32::try_from(value).map_err(inference_tensor::Error::wrap);
        let (table_ptr, table_guard) = table.device_ptr(&stream);
        let (lens_ptr, lens_guard) = context_lens.device_ptr(&stream);
        let (plan_ptr, plan_guard) = plan.device_ptr_mut(&stream);
        let status = unsafe {
            super::ffi::qsa_decode_plan(
                table_ptr as *const u32,
                lens_ptr as *const u32,
                plan_ptr as *mut u32,
                int(rows)?,
                int(table_width)?,
                int(self.block_size)?,
                int(self.ratio)?,
                int(self.k)?,
                int(self.pools)?,
                stream.cu_stream() as i64,
            )
        };
        drop(plan_guard);
        drop(lens_guard);
        drop(table_guard);
        if status != 0 {
            inference_tensor::bail!("{} failed with status {status}", self.name());
        }
        Ok((
            inference_tensor::CudaStorage::wrap_cuda_slice(plan, device.clone()),
            Shape::from((rows, width)),
        ))
    }
}

/// One decode step's QSA plan from each row's block `table` row and context length (its tokens, this one included),
/// built on the device so a captured step replays it: per row, `ratio` cells of the block holding its position
/// (pooled every step; until the block completes the pool lands in its future last cell, which that token's own
/// write replaces), the position, the kept count of [`qsa_gather_cells`], then `pools` pool cells, past the row's
/// complete blocks any valid cell.
pub fn qsa_decode_plan(
    table: &Tensor,
    context_lens: &Tensor,
    block_size: usize,
    ratio: usize,
    k: usize,
    pools: usize,
) -> Result<Tensor> {
    if table.dtype() != DType::U32 || context_lens.dtype() != DType::U32 {
        inference_tensor::bail!("a QSA decode plan takes u32 block tables and context lens");
    }
    table.contiguous()?.apply_op2_no_bwd(
        &context_lens.contiguous()?,
        &QsaDecodePlan {
            block_size,
            ratio,
            k,
            pools,
        },
    )
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

    #[test]
    fn a_decode_plan_pools_the_current_block_and_lists_every_pool() -> Result<()> {
        // two rows of 8-cell pages: row 0 at position 13 (pages 5, 2), row 1 at 3 (page 7)
        let table = Tensor::new(&[[5u32, 2], [7, 0]], &Device::Cpu)?;
        let lens = Tensor::new(&[14u32, 4], &Device::Cpu)?;
        let plan = qsa_decode_plan(&table, &lens, 8, RATIO, K, 4)?.to_vec2::<u32>()?;
        // block 3 is cells 12..16 of page 2; three blocks complete, under the budget of 3 so all 14 kept
        assert_eq!(plan[0], vec![20, 21, 22, 23, 13, 14, 43, 47, 19, 43]);
        // a full first block: k blocks keep everything
        assert_eq!(plan[1], vec![56, 57, 58, 59, 3, 4, 59, 59, 59, 59]);
        let lens = Tensor::new(&[16u32, 4], &Device::Cpu)?;
        let plan = qsa_decode_plan(&table, &lens, 8, RATIO, K, 4)?.to_vec2::<u32>()?;
        // four complete blocks are past the budget: k blocks plus an empty tail
        assert_eq!(plan[0][RATIO + 1], (K * RATIO) as u32);
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_decode_plan_matches_the_reference() -> Result<()> {
        let Ok(device) = Device::new_cuda(0) else {
            return Ok(());
        };
        let rows = 5;
        let width = 6;
        let table = (0..rows * width)
            .map(|i| ((i * 7) % 31) as u32)
            .collect::<Vec<_>>();
        let table = Tensor::from_vec(table, (rows, width), &Device::Cpu)?;
        let lens = Tensor::new(&[1u32, 4, 17, 33, 48], &Device::Cpu)?;
        let pools = width * 8 / RATIO;
        let expected = qsa_decode_plan(&table, &lens, 8, RATIO, K, pools)?.to_vec2::<u32>()?;
        let actual = qsa_decode_plan(
            &table.to_device(&device)?,
            &lens.to_device(&device)?,
            8,
            RATIO,
            K,
            pools,
        )?
        .to_vec2::<u32>()?;
        assert_eq!(actual, expected);
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

    #[test]
    fn cells_list_the_masks_kept_cells_in_order() -> Result<()> {
        let n_pool = N_KV / RATIO;
        let mut scores = (0..n_pool)
            .map(|b| ((b * 5) % 7) as f32)
            .collect::<Vec<_>>();
        scores[3] = 100.;
        let scores = Tensor::from_vec(scores.repeat(4), (4, n_pool), &Device::Cpu)?;
        let positions = Tensor::new(&[30u32, 9, QSA_PADDED_ROW, 63], &Device::Cpu)?;
        let mask = qsa_select_mask(&scores, &positions, N_KV, K, RATIO)?;
        let cells = qsa_select_cells(&scores, &positions, K, RATIO)?;
        assert_eq!(cells.dims2()?, (4, qsa_cells_width(K, RATIO)));
        for row in 0..4 {
            let listed = cells.get(row)?.to_vec1::<u32>()?;
            let expected = kept(&mask, row)?;
            let ordered = listed[..expected.len()]
                .iter()
                .map(|&c| c as usize)
                .collect::<Vec<_>>();
            assert_eq!(ordered, expected, "row {row}");
            assert!(
                listed[expected.len()..].iter().all(|&c| c == listed[0]),
                "row {row}"
            );
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_cells_match_the_reference() -> Result<()> {
        let Ok(device) = Device::new_cuda(0) else {
            return Ok(());
        };
        const ROWS: usize = 6;
        const POOLS: usize = 3000;
        const CELLS: usize = POOLS * RATIO + RATIO - 1;
        const BUDGET: usize = 512;
        let scores = (0..ROWS * POOLS)
            .map(|i| ((i * 7919) % 37) as f32 * 0.5 - 4.)
            .map(|v: f32| v.max(0.))
            .collect::<Vec<_>>();
        let scores = Tensor::from_vec(scores, (ROWS, POOLS), &Device::Cpu)?;
        let positions = Tensor::new(
            &[0u32, 2050, 2051, 9000, CELLS as u32 - 1, QSA_PADDED_ROW],
            &Device::Cpu,
        )?;
        let expected = qsa_select_cells(&scores, &positions, BUDGET, RATIO)?.to_vec2::<u32>()?;
        let actual = qsa_select_cells(
            &scores.to_device(&device)?,
            &positions.to_device(&device)?,
            BUDGET,
            RATIO,
        )?
        .to_vec2::<u32>()?;
        for row in 0..ROWS {
            assert_eq!(actual[row], expected[row], "row {row}");
        }
        Ok(())
    }

    // (pages, heads, block_size) of a small paged cache, two sequences over shuffled pages
    const PAGES: usize = 6;
    const HEADS: usize = 2;
    const BLOCK: usize = 4;
    const WIDTH: usize = 5;
    const OUT_PAGES: usize = 2;

    fn gather_inputs(row: usize, device: &Device) -> Result<(Tensor, Tensor, Tensor)> {
        let cache = Tensor::arange(0f32, (PAGES * HEADS * BLOCK * row) as f32, device)?
            .reshape((PAGES, HEADS, BLOCK, row))?;
        let table = Tensor::new(&[[4u32, 1, 3], [0, 5, 2]], device)?;
        let cells = Tensor::new(&[[0u32, 1, 5, 9, 10], [11, 2, 3, 3, 3]], device)?;
        Ok((cache, table, cells))
    }

    #[test]
    fn gathered_cells_read_their_paged_rows() -> Result<()> {
        let row = 3;
        let (cache, table, cells) = gather_inputs(row, &Device::Cpu)?;
        let out = qsa_gather_cells(&cache, &table, &cells, OUT_PAGES)?;
        assert_eq!(out.dims4()?, (2 * OUT_PAGES, HEADS, BLOCK, row));
        let table = table.to_vec2::<u32>()?;
        let cells = cells.to_vec2::<u32>()?;
        for r in 0..2 {
            for slot in 0..OUT_PAGES * BLOCK {
                let position = cells[r][if slot < WIDTH { slot } else { 0 }] as usize;
                let page = table[r][position / BLOCK] as usize;
                for h in 0..HEADS {
                    let got = out
                        .get(r * OUT_PAGES + slot / BLOCK)?
                        .get(h)?
                        .get(slot % BLOCK)?
                        .to_vec1::<f32>()?;
                    let want = cache
                        .get(page)?
                        .get(h)?
                        .get(position % BLOCK)?
                        .to_vec1::<f32>()?;
                    assert_eq!(got, want, "row {r} slot {slot} head {h}");
                }
            }
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_gather_matches_the_reference() -> Result<()> {
        let Ok(device) = Device::new_cuda(0) else {
            return Ok(());
        };
        // 3 f32 a row copies words, 4 copies 16-byte vectors, and 3 bytes copy bytes
        for (row, dtype) in [
            (3, DType::F32),
            (4, DType::F32),
            (8, DType::BF16),
            (3, DType::U8),
        ] {
            let (cache, table, cells) = gather_inputs(row, &Device::Cpu)?;
            let cache = cache.to_dtype(dtype)?;
            let expected = qsa_gather_cells(&cache, &table, &cells, OUT_PAGES)?;
            let actual = qsa_gather_cells(
                &cache.to_device(&device)?,
                &table.to_device(&device)?,
                &cells.to_device(&device)?,
                OUT_PAGES,
            )?
            .to_device(&Device::Cpu)?;
            let flat = |t: &Tensor| t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>();
            assert_eq!(flat(&actual)?, flat(&expected)?, "{row} {dtype:?}");
        }
        Ok(())
    }
}
