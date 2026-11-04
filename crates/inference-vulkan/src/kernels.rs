use std::sync::Arc;

use crate::map::{MAP_BINDINGS, MAP_PUSH_BYTES};
use crate::memory::{Bound, Buffer};
use crate::ops;
use crate::pipeline::Pipeline;
use crate::{Error, Recorder, Result, shaders};

// values a block of the 32-value types holds
const BLOCK_VALUES: usize = 32;
// rows one matvec workgroup takes (the shader's `ROWS`), and the most workgroups one grid dimension holds
const MATVEC_ROWS: usize = 4;
// the matvecs' buffers and push constants, and an int8 block's bytes (8 packed words, then its (d, d * sum q) in the
// same buffer)
const MATVECS_BINDINGS: usize = 4;
const MATVECS_PUSH_BYTES: u32 = 7 * 4;
const Q8_WORD_BYTES: usize = 32;
const Q8_SCALE_BYTES: usize = 8;
const QUANTIZE_BINDINGS: usize = 3;
const QUANTIZE_PUSH_BYTES: u32 = 3 * 4;
const QUANTIZE_THREADS: usize = 64;
const MAX_GROUPS_X: usize = 65_535;
// the most activation rows the matvec kernels take (their MAX_M); more take the tiled matmul, its tile's weight rows
// and tokens
pub const MATVEC_MAX_ROWS: usize = 8;
const MATMUL_Q_TILE_N: usize = 64;
const MATMUL_Q_TILE_M: usize = 64;
const MATMUL_Q_PUSH_BYTES: u32 = 6 * 4;

/// Every kernel's pipeline on one device: its field, its embedded SPIR-V, its storage buffers and push-constant bytes.
macro_rules! kernels {
    ($($name:ident: $spirv:ident, $bindings:expr, $push:expr;)*) => {
        pub(crate) struct Kernels {
            $(pub(crate) $name: Pipeline,)*
        }

        impl Kernels {
            pub(crate) fn new(device: &ash::Device) -> Result<Self> {
                Ok(Self {
                    $($name: Pipeline::new(device, shaders::$spirv, $bindings, $push, stringify!($name))?,)*
                })
            }

            /// # Safety
            /// No submit may still use the pipelines.
            pub(crate) unsafe fn destroy(&self, device: &ash::Device) {
                // SAFETY: the caller's contract
                unsafe {
                    $(self.$name.destroy(device);)*
                }
            }
        }
    };
}

kernels! {
    mat_vecs_q8_0: MUL_MAT_VECS_Q8_0, MATVECS_BINDINGS, MATVECS_PUSH_BYTES;
    mat_vecs_q5_0: MUL_MAT_VECS_Q5_0, MATVECS_BINDINGS, MATVECS_PUSH_BYTES;
    mat_vecs_q4_0: MUL_MAT_VECS_Q4_0, MATVECS_BINDINGS, MATVECS_PUSH_BYTES;
    mat_vecs_q4_1: MUL_MAT_VECS_Q4_1, MATVECS_BINDINGS, MATVECS_PUSH_BYTES;
    mat_vecs_q5_1: MUL_MAT_VECS_Q5_1, MATVECS_BINDINGS, MATVECS_PUSH_BYTES;
    mat_vecs_q4_k: MUL_MAT_VECS_Q4_K, MATVECS_BINDINGS, MATVECS_PUSH_BYTES;
    mat_vecs_q5_k: MUL_MAT_VECS_Q5_K, MATVECS_BINDINGS, MATVECS_PUSH_BYTES;
    mat_vecs_q6_k: MUL_MAT_VECS_Q6_K, MATVECS_BINDINGS, MATVECS_PUSH_BYTES;
    quantize_q8_1: QUANTIZE_Q8_1, QUANTIZE_BINDINGS, QUANTIZE_PUSH_BYTES;
    mul_mm_q8_0: MUL_MM_Q8_0, MATVECS_BINDINGS, MATMUL_Q_PUSH_BYTES;
    mul_mm_q5_0: MUL_MM_Q5_0, MATVECS_BINDINGS, MATMUL_Q_PUSH_BYTES;
    mul_mm_q4_0: MUL_MM_Q4_0, MATVECS_BINDINGS, MATMUL_Q_PUSH_BYTES;
    mul_mm_q4_1: MUL_MM_Q4_1, MATVECS_BINDINGS, MATMUL_Q_PUSH_BYTES;
    mul_mm_q5_1: MUL_MM_Q5_1, MATVECS_BINDINGS, MATMUL_Q_PUSH_BYTES;
    mul_mm_q4_k: MUL_MM_Q4_K, MATVECS_BINDINGS, MATMUL_Q_PUSH_BYTES;
    mul_mm_q5_k: MUL_MM_Q5_K, MATVECS_BINDINGS, MATMUL_Q_PUSH_BYTES;
    mul_mm_q6_k: MUL_MM_Q6_K, MATVECS_BINDINGS, MATMUL_Q_PUSH_BYTES;
    map_f32: MAP_F32, MAP_BINDINGS, MAP_PUSH_BYTES;
    copy_b32: COPY_B32, MAP_BINDINGS, MAP_PUSH_BYTES;
    rows_f32: ROWS_F32, ops::BINDINGS, ops::ROWS_PUSH_BYTES;
    index_b32: INDEX_B32, ops::BINDINGS, ops::INDEX_PUSH_BYTES;
    matmul_f32: MATMUL_F32, ops::BINDINGS, ops::MATMUL_PUSH_BYTES;
    matmul_f32_dot: MATMUL_F32_DOT, ops::BINDINGS, ops::MATMUL_PUSH_BYTES;
    matmul_f32_tiled: MATMUL_F32_TILED, ops::BINDINGS, ops::MATMUL_PUSH_BYTES;
    ssm_conv_step: SSM_CONV_STEP, ops::CONV_STEP_BINDINGS, ops::CONV_STEP_PUSH_BYTES;
    ssm_scan_step: SSM_SCAN_STEP, ops::SCAN_STEP_BINDINGS, ops::SCAN_STEP_PUSH_BYTES;
    ssd_chunk_state: SSD_CHUNK_STATE, ops::SSD_BINDINGS, ops::SCAN_STEP_PUSH_BYTES;
    ssd_state_pass: SSD_STATE_PASS, ops::SSD_BINDINGS, ops::SCAN_STEP_PUSH_BYTES;
    ssd_chunk_out: SSD_CHUNK_OUT, ops::SSD_BINDINGS, ops::SCAN_STEP_PUSH_BYTES;
    moe_mat_vecs_q5_0: MOE_MAT_VECS_Q5_0, ops::MOE_MATVEC_BINDINGS, ops::MOE_MATVEC_PUSH_BYTES;
    moe_sort: MOE_SORT, ops::MOE_SORT_BINDINGS, ops::MOE_SORT_PUSH_BYTES;
    moe_mm_q5_0: MOE_MM_Q5_0, ops::MOE_MM_BINDINGS, ops::MOE_MM_PUSH_BYTES;
    moe_router: MOE_ROUTER, ops::ROUTER_BINDINGS, ops::ROUTER_PUSH_BYTES;
    paged_cache_write: PAGED_CACHE_WRITE, ops::PAGED_WRITE_BINDINGS, ops::PAGED_WRITE_PUSH_BYTES;
    paged_decode: PAGED_DECODE, ops::PAGED_DECODE_BINDINGS, ops::PAGED_DECODE_PUSH_BYTES;
    paged_decode_reduce: PAGED_DECODE_REDUCE, ops::PAGED_REDUCE_BINDINGS, ops::PAGED_REDUCE_PUSH_BYTES;
}

/// A GGUF weight's block format the matvec kernels read.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BlockType {
    /// an f16 scale and 32 int8 values
    Q8_0,
    /// an f16 scale, 32 fifth bits and 32 low nibbles, values offset by 16
    Q5_0,
    /// an f16 scale and 32 nibbles, values offset by 8
    Q4_0,
    /// f16 scale and min, 32 nibbles
    Q4_1,
    /// f16 scale and min, 32 fifth bits and 32 low nibbles
    Q5_1,
    /// 256 values in eight 32-value sub-blocks of nibbles, each with a 6-bit scale and min
    Q4K,
    /// Q4K with a fifth bit a value
    Q5K,
    /// 256 six-bit values, an int8 scale per 16
    Q6K,
}

impl BlockType {
    /// The bytes of one stored block.
    pub fn block_bytes(self) -> usize {
        match self {
            Self::Q8_0 => 34,
            Self::Q5_0 => 22,
            Self::Q4_0 => 18,
            Self::Q4_1 => 20,
            Self::Q5_1 => 24,
            Self::Q4K => 144,
            Self::Q5K => 176,
            Self::Q6K => 210,
        }
    }

    /// The values one stored block holds; a row's length is a multiple of it.
    pub fn block_values(self) -> usize {
        match self {
            Self::Q4K | Self::Q5K | Self::Q6K => 256,
            _ => BLOCK_VALUES,
        }
    }
}

/// `y[y_offset + m * rows..][..rows] = W x[x_offset + m * cols..][..cols]` for each of `x_rows` activation rows, for a
/// `rows x cols` weight of `ty` blocks held in `weights`: up to [`MATVEC_MAX_ROWS`] rows a matvec reading the weight
/// once, more a tiled matmul. The rows are quantized to int8 blocks (as the CPU's quantized matmuls do) into
/// `scratch`, of [`MatVec::scratch_bytes`].
pub struct MatVec<'a> {
    pub ty: BlockType,
    pub weights: &'a Arc<dyn Bound>,
    pub rows: usize,
    pub cols: usize,
    pub x: &'a Arc<Buffer>,
    pub x_offset: usize,
    pub y: &'a Arc<Buffer>,
    pub y_offset: usize,
    pub x_rows: usize,
    pub scratch: &'a Arc<Buffer>,
}

impl MatVec<'_> {
    /// The scratch bytes `x_rows` rows of `cols` take.
    pub fn scratch_bytes(x_rows: usize, cols: usize) -> usize {
        x_rows * cols / BLOCK_VALUES * (Q8_WORD_BYTES + Q8_SCALE_BYTES)
    }
}

fn push_words(values: &[usize]) -> Result<Vec<u8>> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for &v in values {
        let v = u32::try_from(v)
            .map_err(|_| Error::Invalid(format!("matvec constant {v} past u32")))?;
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    Ok(bytes)
}

impl Recorder<'_> {
    // `rows` f32 rows of `cols` from `x` as int8 blocks in `scratch`, giving where their scales start (in vec2s)
    pub(crate) fn quantize_rows(
        &self,
        x: crate::At<'_>,
        rows: usize,
        cols: usize,
        scratch: &Arc<Buffer>,
    ) -> Result<usize> {
        let x_blocks = rows * cols / BLOCK_VALUES;
        if scratch.len() < MatVec::scratch_bytes(rows, cols) || x.0.len() / 4 < x.1 + rows * cols {
            return Err(Error::Invalid(format!(
                "{rows} rows of {cols} quantized into {} bytes",
                scratch.len()
            )));
        }
        let s_off = x_blocks * Q8_WORD_BYTES / Q8_SCALE_BYTES;
        self.context.kernels.quantize_q8_1.dispatch(
            self,
            &[x.0.buffer, scratch.buffer, scratch.buffer],
            &push_words(&[x_blocks, x.1, s_off])?,
            [x_blocks.div_ceil(QUANTIZE_THREADS) as u32, 1, 1],
        );
        self.retain(x.0.clone());
        self.retain(scratch.clone());
        Ok(s_off)
    }

    pub fn mat_vec(&self, op: &MatVec<'_>) -> Result<()> {
        let fits = op.cols.is_multiple_of(op.ty.block_values())
            && op.weights.offset().is_multiple_of(2)
            && op.weights.bytes() >= op.rows * op.cols / op.ty.block_values() * op.ty.block_bytes()
            && op.x.len() / 4 >= op.x_offset + op.x_rows * op.cols
            && op.y.len() / 4 >= op.y_offset + op.x_rows * op.rows
            && op.x_rows > 0;
        if !fits {
            return Err(Error::Invalid(format!(
                "a {}x{} {:?} matvec over {} weight bytes at offset {}",
                op.rows,
                op.cols,
                op.ty,
                op.weights.bytes(),
                op.weights.offset()
            )));
        }
        let scratch = op.scratch;
        if scratch.len() < MatVec::scratch_bytes(op.x_rows, op.cols) {
            return Err(Error::Invalid(format!(
                "a {}-row matvec's scratch of {} bytes",
                op.x_rows,
                scratch.len()
            )));
        }
        let kernels = &self.context.kernels;
        let s_off = self.quantize_rows((op.x, op.x_offset), op.x_rows, op.cols, scratch)?;
        if op.x_rows > MATVEC_MAX_ROWS {
            let pipeline = match op.ty {
                BlockType::Q8_0 => &kernels.mul_mm_q8_0,
                BlockType::Q5_0 => &kernels.mul_mm_q5_0,
                BlockType::Q4_0 => &kernels.mul_mm_q4_0,
                BlockType::Q4_1 => &kernels.mul_mm_q4_1,
                BlockType::Q5_1 => &kernels.mul_mm_q5_1,
                BlockType::Q4K => &kernels.mul_mm_q4_k,
                BlockType::Q5K => &kernels.mul_mm_q5_k,
                BlockType::Q6K => &kernels.mul_mm_q6_k,
            };
            let bytes = push_words(&[
                op.rows,
                op.cols,
                op.weights.offset() as usize / 2,
                op.y_offset,
                op.x_rows,
                s_off,
            ])?;
            pipeline.dispatch(
                self,
                &[
                    op.weights.vk_buffer(),
                    scratch.buffer,
                    op.y.buffer,
                    scratch.buffer,
                ],
                &bytes,
                [
                    op.rows.div_ceil(MATMUL_Q_TILE_N) as u32,
                    op.x_rows.div_ceil(MATMUL_Q_TILE_M) as u32,
                    1,
                ],
            );
            self.work.set(
                self.work
                    .get()
                    .saturating_add((op.rows * op.cols * op.x_rows) as u64),
            );
            self.retain(scratch.clone());
            self.retain(op.weights.clone() as crate::Retained);
            self.retain(op.y.clone());
            return Ok(());
        }
        let groups = op.rows.div_ceil(MATVEC_ROWS);
        let groups_x = groups.min(MAX_GROUPS_X);
        let bytes = push_words(&[
            op.rows,
            op.cols,
            op.weights.offset() as usize / 2,
            op.y_offset,
            groups_x,
            op.x_rows,
            s_off,
        ])?;
        let pipeline = match op.ty {
            BlockType::Q8_0 => &kernels.mat_vecs_q8_0,
            BlockType::Q5_0 => &kernels.mat_vecs_q5_0,
            BlockType::Q4_0 => &kernels.mat_vecs_q4_0,
            BlockType::Q4_1 => &kernels.mat_vecs_q4_1,
            BlockType::Q5_1 => &kernels.mat_vecs_q5_1,
            BlockType::Q4K => &kernels.mat_vecs_q4_k,
            BlockType::Q5K => &kernels.mat_vecs_q5_k,
            BlockType::Q6K => &kernels.mat_vecs_q6_k,
        };
        pipeline.dispatch(
            self,
            &[
                op.weights.vk_buffer(),
                scratch.buffer,
                op.y.buffer,
                scratch.buffer,
            ],
            &bytes,
            [groups_x as u32, groups.div_ceil(groups_x) as u32, 1],
        );
        self.retain(scratch.clone());
        self.retain(op.weights.clone() as crate::Retained);
        self.retain(op.x.clone());
        self.retain(op.y.clone());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Context, ImportedMemory, f32_bytes};
    use std::alloc::{Layout, alloc_zeroed, dealloc};

    const PAGE: usize = 4096;
    const TOLERANCE: f64 = 1e-4;

    // page-aligned host memory, so an import of it covers no one else's pages
    struct Pages {
        ptr: *mut u8,
        layout: Layout,
    }

    impl Pages {
        fn new(len: usize) -> Self {
            let layout = Layout::from_size_align(len.next_multiple_of(PAGE), PAGE).unwrap();
            // SAFETY: the layout is nonzero and page aligned
            let ptr = unsafe { alloc_zeroed(layout) };
            assert!(!ptr.is_null());
            Self { ptr, layout }
        }

        fn bytes(&mut self) -> &mut [u8] {
            // SAFETY: the allocation is `layout.size()` bytes and owned by `self`
            unsafe { std::slice::from_raw_parts_mut(self.ptr, self.layout.size()) }
        }
    }

    impl Drop for Pages {
        fn drop(&mut self) {
            // SAFETY: allocated in `new` with this layout
            unsafe { dealloc(self.ptr, self.layout) };
        }
    }

    fn context() -> Option<Arc<Context>> {
        match Context::new() {
            Ok(context) => Some(context),
            Err(err) => {
                eprintln!("skipping: {err}");
                None
            }
        }
    }

    fn len(ty: BlockType, rows: usize, cols: usize) -> usize {
        rows * cols / BLOCK_VALUES * ty.block_bytes()
    }

    // random blocks of `ty` with small finite scales
    fn weights(ty: BlockType, bytes: &mut [u8], rows: usize, cols: usize) {
        let mut state = 0x2545_f491u32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        let size = ty.block_bytes();
        for block in bytes[..len(ty, rows, cols)].chunks_exact_mut(size) {
            let d = half::f16::from_f32(0.001 + (next() % 100) as f32 * 1e-5);
            block[..2].copy_from_slice(&d.to_le_bytes());
            block[2..].iter_mut().for_each(|q| *q = next() as u8);
        }
    }

    // value `j` of a block, as ggml's dequantize_row reads it
    fn value(ty: BlockType, block: &[u8], j: usize) -> f64 {
        match ty {
            BlockType::Q8_0 => f64::from(block[2 + j] as i8),
            BlockType::Q5_0 => {
                let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
                let byte = block[6 + j % 16];
                let nibble = if j < 16 { byte & 0xf } else { byte >> 4 };
                f64::from(i32::from(nibble) | (((qh >> j) & 1) as i32) << 4) - 16.0
            }
            other => unreachable!(
                "{other:?} is checked against the CPU's dequantization in inference-tensor"
            ),
        }
    }

    fn reference(ty: BlockType, bytes: &[u8], rows: usize, cols: usize, x: &[f32]) -> Vec<f64> {
        let (blocks, size) = (cols / BLOCK_VALUES, ty.block_bytes());
        (0..rows)
            .map(|r| {
                (0..blocks)
                    .map(|b| {
                        let block = &bytes[(r * blocks + b) * size..][..size];
                        let d = f64::from(half::f16::from_le_bytes([block[0], block[1]]).to_f32());
                        let dot: f64 = (0..BLOCK_VALUES)
                            .map(|j| value(ty, block, j) * f64::from(x[b * BLOCK_VALUES + j]))
                            .sum();
                        d * dot
                    })
                    .sum()
            })
            .collect()
    }

    // matvecs (and past 8 rows the tiled matmul) take int8 activations: each output within the rounding bound of its
    // blocks (|w| times half a block's step, summed) of the f32 product, over weight rows and tokens past a
    // workgroup's and activation rows unaligned in their buffer
    #[test]
    fn matvecs_round_only_their_activations() -> Result<()> {
        let Some(context) = context() else {
            return Ok(());
        };
        const ROWS: usize = 333;
        const COLS: usize = 2688;
        const X_AT: usize = 3;
        for (ty, x_rows) in [
            (BlockType::Q8_0, 1),
            (BlockType::Q8_0, 3),
            (BlockType::Q8_0, 70),
            (BlockType::Q5_0, 1),
            (BlockType::Q5_0, 3),
            (BlockType::Q5_0, 70),
        ] {
            let len = len(ty, ROWS, COLS);
            let mut pages = Pages::new(len);
            weights(ty, pages.bytes(), ROWS, COLS);
            let x: Vec<f32> = (0..x_rows * COLS)
                .map(|i| ((i * 37 % 101) as f32 - 50.0) / (25.0 + (i / COLS) as f32))
                .collect();
            let size = ty.block_bytes();
            let blocks = COLS / BLOCK_VALUES;
            let weights: Arc<dyn Bound> =
                Arc::new(Buffer::with_bytes(&context, &pages.bytes()[..len])?);
            let mut xs = Buffer::new(&context, (X_AT + x_rows * COLS) * 4)?;
            xs.write(X_AT * 4, f32_bytes(&x))?;
            let (xs, ys) = (
                Arc::new(xs),
                Arc::new(Buffer::new(&context, x_rows * ROWS * 4)?),
            );
            let scratch = Arc::new(Buffer::new(&context, MatVec::scratch_bytes(x_rows, COLS))?);
            context.submit(|r| {
                r.mat_vec(&MatVec {
                    ty,
                    weights: &weights,
                    rows: ROWS,
                    cols: COLS,
                    x: &xs,
                    x_offset: X_AT,
                    y: &ys,
                    y_offset: 0,
                    x_rows,
                    scratch: &scratch,
                })
            })?;
            let got = ys.read_f32()?;
            for m in 0..x_rows {
                let xm = &x[m * COLS..][..COLS];
                let want = reference(ty, pages.bytes(), ROWS, COLS, xm);
                for (row, &want) in want.iter().enumerate() {
                    let bound: f64 = (0..blocks)
                        .map(|b| {
                            let block = &pages.bytes()[(row * blocks + b) * size..][..size];
                            let d =
                                f64::from(half::f16::from_le_bytes([block[0], block[1]]).to_f32());
                            let xb = &xm[b * BLOCK_VALUES..][..BLOCK_VALUES];
                            let step =
                                f64::from(xb.iter().fold(0f32, |a, v| a.max(v.abs()))) / 127.0;
                            let w: f64 = (0..BLOCK_VALUES)
                                .map(|j| (d * value(ty, block, j)).abs())
                                .sum();
                            w * step / 2.0
                        })
                        .sum();
                    let got = f64::from(got[m * ROWS + row]);
                    assert!(
                        (got - want).abs() <= bound * 1.01 + 1e-5,
                        "{ty:?} x row {m} row {row}: {got} vs {want} (bound {bound})"
                    );
                }
            }
        }
        Ok(())
    }

    // more dispatches than one batch takes, with the caller's handles to the inputs dropped before the GPU finishes
    #[test]
    fn recorded_work_keeps_its_buffers_alive_across_batches() -> Result<()> {
        let Some(context) = context() else {
            return Ok(());
        };
        const ROWS: usize = 64;
        const COLS: usize = 256;
        const DISPATCHES: usize = 3 * crate::stream::DISPATCHES_PER_SUBMIT / 2;
        let ty = BlockType::Q8_0;
        let len = len(ty, ROWS, COLS);
        let mut pages = Pages::new(len);
        weights(ty, pages.bytes(), ROWS, COLS);
        // -1, 0 and 1, which the int8 activations hold exactly
        let x: Vec<f32> = (0..COLS).map(|i| (i % 3) as f32 - 1.0).collect();
        let want = reference(ty, pages.bytes(), ROWS, COLS, &x);
        let ys = Arc::new(Buffer::new(&context, DISPATCHES * ROWS * 4)?);
        let scratch = Arc::new(Buffer::new(&context, MatVec::scratch_bytes(1, COLS))?);
        {
            // SAFETY: `pages` outlives every submit, which `synchronize` below waits for
            let weights: Arc<dyn Bound> =
                Arc::new(unsafe { ImportedMemory::new(&context, &pages.bytes()[..len])? });
            let xs = Arc::new(Buffer::with_f32(&context, &x)?);
            for d in 0..DISPATCHES {
                context.record(|r| {
                    r.mat_vec(&MatVec {
                        ty,
                        weights: &weights,
                        rows: ROWS,
                        cols: COLS,
                        x: &xs,
                        x_offset: 0,
                        y: &ys,
                        y_offset: d * ROWS,
                        x_rows: 1,
                        scratch: &scratch,
                    })
                })?;
            }
        }
        context.synchronize()?;
        for (i, &got) in ys.read_f32()?.iter().enumerate() {
            let want = want[i % ROWS];
            assert!(
                (f64::from(got) - want).abs() <= TOLERANCE * want.abs().max(1.0),
                "{i}: {got} vs {want}"
            );
        }
        Ok(())
    }

    // an lm_head-sized weight (131072 x 2688) read twenty times a submit, per block type and activation rows
    #[test]
    #[ignore = "benchmark"]
    fn matvec_bandwidth() -> Result<()> {
        let Some(context) = context() else {
            return Ok(());
        };
        const ROWS: usize = 131_072;
        const COLS: usize = 2688;
        const REPEATS: usize = 20;
        const SUBMITS: usize = 5;
        for ty in [BlockType::Q8_0, BlockType::Q5_0] {
            let len = len(ty, ROWS, COLS);
            let mut pages = Pages::new(len);
            weights(ty, pages.bytes(), ROWS, COLS);
            let weights: Arc<dyn Bound> =
                Arc::new(Buffer::with_bytes(&context, &pages.bytes()[..len])?);
            for x_rows in [1usize, 2, 3, 4, 8, 64, 512] {
                let mut xs = Buffer::new(&context, x_rows * COLS * 4)?;
                xs.fill(0.5f32.to_bits())?;
                let (xs, ys) = (
                    Arc::new(xs),
                    Arc::new(Buffer::new(&context, x_rows * ROWS * 4)?),
                );
                let scratch = Arc::new(Buffer::new(&context, MatVec::scratch_bytes(x_rows, COLS))?);
                let op = MatVec {
                    ty,
                    weights: &weights,
                    rows: ROWS,
                    cols: COLS,
                    x: &xs,
                    x_offset: 0,
                    y: &ys,
                    y_offset: 0,
                    x_rows,
                    scratch: &scratch,
                };
                // a record a call, as the backend's ops do, so heavy calls split across submits
                let run = || -> Result<()> {
                    for _ in 0..REPEATS {
                        context.record(|r| r.mat_vec(&op))?;
                    }
                    context.synchronize()
                };
                run()?;
                let start = std::time::Instant::now();
                for _ in 0..SUBMITS {
                    run()?;
                }
                let secs = start.elapsed().as_secs_f64();
                let calls = (SUBMITS * REPEATS) as f64;
                println!(
                    "{ty:?} x {x_rows} rows: {:.1} GB/s, {:.2} TOPS, {:.0} us each",
                    calls * len as f64 / secs / 1e9,
                    calls * (2 * x_rows * ROWS * COLS) as f64 / secs / 1e12,
                    secs / calls * 1e6
                );
            }
        }
        Ok(())
    }

    // a prompt step's experts: 512 tokens routed to 6 of 128 experts (1856 x 2688, Q5_0), each pass timed on its own
    #[test]
    #[ignore = "benchmark"]
    fn expert_matmul_throughput() -> Result<()> {
        let Some(context) = context() else {
            return Ok(());
        };
        const EXPERTS: usize = 128;
        const ROWS: usize = 1856;
        const COLS: usize = 2688;
        const TOKENS: usize = 512;
        const K: usize = 6;
        const REPEATS: usize = 10;
        let ty = BlockType::Q5_0;
        let len = EXPERTS * len(ty, ROWS, COLS);
        let mut pages = Pages::new(len);
        weights(ty, pages.bytes(), EXPERTS * ROWS, COLS);
        let weights: Arc<dyn Bound> =
            Arc::new(Buffer::with_bytes(&context, &pages.bytes()[..len])?);
        let slots = TOKENS * K;
        let mut id_bytes = vec![0u8; slots * 4];
        let mut state = 0x9e37_79b9u32;
        for (t, chunk) in id_bytes
            .as_chunks_mut::<{ 4 * K }>()
            .0
            .iter_mut()
            .enumerate()
        {
            for (j, id) in chunk.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                // distinct experts a token: spread from a random start
                let e = ((state >> 8) as usize + j * 21 + t) % EXPERTS;
                id.copy_from_slice(&(e as u32).to_le_bytes());
            }
        }
        let ids = Arc::new(Buffer::with_bytes(&context, &id_bytes)?);
        let x: Vec<f32> = (0..TOKENS * COLS).map(|i| (i % 13) as f32 - 6.0).collect();
        let xs = Arc::new(Buffer::with_f32(&context, &x)?);
        let ys = Arc::new(Buffer::new(&context, slots * ROWS * 4)?);
        let scratch = Arc::new(Buffer::new(&context, MatVec::scratch_bytes(TOKENS, COLS))?);
        let routing = Arc::new(Buffer::new(
            &context,
            crate::MoeMatVec::routing_bytes(slots, EXPERTS),
        )?);
        let op = crate::MoeMatVec {
            weights: &weights,
            experts: EXPERTS,
            rows: ROWS,
            cols: COLS,
            slots,
            x_div: K,
            x: (&xs, 0),
            ids: (&ids, 0),
            y: &ys,
            scratch: &scratch,
            routing: &routing,
        };
        context.submit(|r| r.moe_mat_vec_q5_0(&op))?;
        let start = std::time::Instant::now();
        for _ in 0..REPEATS {
            context.submit(|r| r.moe_mat_vec_q5_0(&op))?;
        }
        let secs = start.elapsed().as_secs_f64() / REPEATS as f64;
        println!(
            "experts: {:.2} ms a pass, {:.2} int8 TOPS",
            secs * 1e3,
            (2 * slots * ROWS * COLS) as f64 / secs / 1e12
        );
        Ok(())
    }
}
