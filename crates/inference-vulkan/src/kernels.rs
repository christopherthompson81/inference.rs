use std::sync::Arc;

use crate::map::{MAP_BINDINGS, MAP_PUSH_BYTES};
use crate::memory::{Bound, HostBuffer};
use crate::ops;
use crate::pipeline::Pipeline;
use crate::{Error, Recorder, Result, shaders};

// values a block of the 32-value types holds
const BLOCK_VALUES: usize = 32;
// rows one matvec workgroup takes (the shader's `ROWS`), and the most workgroups one grid dimension holds
const MATVEC_ROWS: usize = 2;
const MAX_GROUPS_X: usize = 65_535;
const MATVEC_PUSH_BYTES: u32 = 6 * 4;
const MATVEC_BINDINGS: usize = 3;

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
    mat_vec_q8_0: MUL_MAT_VEC_Q8_0, MATVEC_BINDINGS, MATVEC_PUSH_BYTES;
    mat_vec_q5_0: MUL_MAT_VEC_Q5_0, MATVEC_BINDINGS, MATVEC_PUSH_BYTES;
    map_f32: MAP_F32, MAP_BINDINGS, MAP_PUSH_BYTES;
    copy_b32: COPY_B32, MAP_BINDINGS, MAP_PUSH_BYTES;
    rows_f32: ROWS_F32, ops::BINDINGS, ops::ROWS_PUSH_BYTES;
    index_b32: INDEX_B32, ops::BINDINGS, ops::INDEX_PUSH_BYTES;
    matmul_f32: MATMUL_F32, ops::BINDINGS, ops::MATMUL_PUSH_BYTES;
    matmul_f32_dot: MATMUL_F32_DOT, ops::BINDINGS, ops::MATMUL_PUSH_BYTES;
    ssm_conv_step: SSM_CONV_STEP, ops::CONV_STEP_BINDINGS, ops::CONV_STEP_PUSH_BYTES;
    ssm_scan_step: SSM_SCAN_STEP, ops::SCAN_STEP_BINDINGS, ops::SCAN_STEP_PUSH_BYTES;
    moe_mat_vec_q5_0: MOE_MAT_VEC_Q5_0, ops::MOE_MATVEC_BINDINGS, ops::MOE_MATVEC_PUSH_BYTES;
    moe_router: MOE_ROUTER, ops::ROUTER_BINDINGS, ops::ROUTER_PUSH_BYTES;
}

/// A GGUF weight's block format the matvec kernels read.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BlockType {
    /// an f16 scale and 32 int8 values
    Q8_0,
    /// an f16 scale, 32 fifth bits and 32 low nibbles, values offset by 16
    Q5_0,
}

impl BlockType {
    pub fn block_bytes(self) -> usize {
        match self {
            Self::Q8_0 => 34,
            Self::Q5_0 => 22,
        }
    }
}

/// `y[y_offset..][..rows] = W x[x_offset..][..cols]` for a `rows x cols` weight of `ty` blocks held in `weights`.
pub struct MatVec<'a> {
    pub ty: BlockType,
    pub weights: &'a Arc<dyn Bound>,
    pub rows: usize,
    pub cols: usize,
    pub x: &'a Arc<HostBuffer>,
    pub x_offset: usize,
    pub y: &'a Arc<HostBuffer>,
    pub y_offset: usize,
}

impl Recorder<'_> {
    pub fn mat_vec(&self, op: &MatVec<'_>) -> Result<()> {
        let blocks = op.cols / BLOCK_VALUES;
        let fits = op.cols.is_multiple_of(BLOCK_VALUES)
            && op.weights.offset().is_multiple_of(2)
            && op.weights.bytes() >= op.rows * blocks * op.ty.block_bytes()
            && op.x.len() / 4 >= op.x_offset + op.cols
            && op.y.len() / 4 >= op.y_offset + op.rows;
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
        let groups = op.rows.div_ceil(MATVEC_ROWS);
        let groups_x = groups.min(MAX_GROUPS_X);
        let push = [
            op.rows,
            op.cols,
            op.weights.offset() as usize / 2,
            op.x_offset,
            op.y_offset,
            groups_x,
        ]
        .map(|v| {
            u32::try_from(v).map_err(|_| Error::Invalid(format!("matvec constant {v} past u32")))
        });
        let mut bytes = Vec::with_capacity(MATVEC_PUSH_BYTES as usize);
        for value in push {
            bytes.extend_from_slice(&value?.to_le_bytes());
        }
        let pipeline = match op.ty {
            BlockType::Q8_0 => &self.context.kernels.mat_vec_q8_0,
            BlockType::Q5_0 => &self.context.kernels.mat_vec_q5_0,
        };
        pipeline.dispatch(
            self,
            &[op.weights.vk_buffer(), op.x.buffer, op.y.buffer],
            &bytes,
            [groups_x as u32, groups.div_ceil(groups_x) as u32, 1],
        );
        self.retain(op.weights.clone() as crate::Retained);
        self.retain(op.x.clone());
        self.retain(op.y.clone());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Context, ImportedMemory};
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

    // rows past one workgroup's, an odd count, and a weight that starts mid-page as a GGUF tensor's does
    #[test]
    fn matvecs_match_the_cpu() -> Result<()> {
        for ty in [BlockType::Q8_0, BlockType::Q5_0] {
            matvec_matches_the_cpu(ty)?;
        }
        Ok(())
    }

    fn matvec_matches_the_cpu(ty: BlockType) -> Result<()> {
        let Some(context) = context() else {
            return Ok(());
        };
        const ROWS: usize = 333;
        const COLS: usize = 2688;
        const WEIGHT_AT: usize = 96;
        let mut pages = Pages::new(WEIGHT_AT + len(ty, ROWS, COLS));
        let weight_bytes = &mut pages.bytes()[WEIGHT_AT..];
        weights(ty, weight_bytes, ROWS, COLS);
        let x: Vec<f32> = (0..COLS)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 25.0)
            .collect();
        let want = reference(ty, weight_bytes, ROWS, COLS, &x);
        // SAFETY: `pages` outlives the import and is not written while it lives
        let weights: Arc<dyn Bound> =
            Arc::new(unsafe { ImportedMemory::new(&context, &pages.bytes()[WEIGHT_AT..])? });
        let mut xs = HostBuffer::new(&context, (COLS + 8) * 4)?;
        xs.as_f32_mut()[8..8 + COLS].copy_from_slice(&x);
        let (xs, ys) = (
            Arc::new(xs),
            Arc::new(HostBuffer::new(&context, (ROWS + 4) * 4)?),
        );
        context.submit(|r| {
            r.mat_vec(&MatVec {
                ty,
                weights: &weights,
                rows: ROWS,
                cols: COLS,
                x: &xs,
                x_offset: 8,
                y: &ys,
                y_offset: 4,
            })
        })?;
        for (row, (&got, &want)) in ys.as_f32()[4..4 + ROWS].iter().zip(&want).enumerate() {
            assert!(
                (f64::from(got) - want).abs() <= TOLERANCE * want.abs().max(1.0),
                "{ty:?} row {row}: {got} vs {want}"
            );
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
        let x: Vec<f32> = (0..COLS).map(|i| (i % 7) as f32 - 3.0).collect();
        let want = reference(ty, pages.bytes(), ROWS, COLS, &x);
        let ys = Arc::new(HostBuffer::new(&context, DISPATCHES * ROWS * 4)?);
        {
            // SAFETY: `pages` outlives every submit, which `synchronize` below waits for
            let weights: Arc<dyn Bound> =
                Arc::new(unsafe { ImportedMemory::new(&context, &pages.bytes()[..len])? });
            let mut xs = HostBuffer::new(&context, COLS * 4)?;
            xs.as_f32_mut().copy_from_slice(&x);
            let xs = Arc::new(xs);
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
                    })
                })?;
            }
        }
        context.synchronize()?;
        for (i, &got) in ys.as_f32().iter().enumerate() {
            let want = want[i % ROWS];
            assert!(
                (f64::from(got) - want).abs() <= TOLERANCE * want.abs().max(1.0),
                "{i}: {got} vs {want}"
            );
        }
        Ok(())
    }

    // an lm_head-sized weight (131072 x 2688, 374 MB) read twenty times a submit
    #[test]
    #[ignore = "benchmark"]
    fn q8_0_matvec_bandwidth() -> Result<()> {
        let Some(context) = context() else {
            return Ok(());
        };
        const ROWS: usize = 131_072;
        const COLS: usize = 2688;
        const REPEATS: usize = 20;
        const SUBMITS: usize = 5;
        let ty = BlockType::Q8_0;
        let len = len(ty, ROWS, COLS);
        let mut pages = Pages::new(len);
        weights(ty, pages.bytes(), ROWS, COLS);
        // SAFETY: as in the test above
        let weights: Arc<dyn Bound> =
            Arc::new(unsafe { ImportedMemory::new(&context, &pages.bytes()[..len])? });
        let mut xs = HostBuffer::new(&context, COLS * 4)?;
        xs.as_f32_mut().iter_mut().for_each(|x| *x = 0.5);
        let (xs, ys) = (Arc::new(xs), Arc::new(HostBuffer::new(&context, ROWS * 4)?));
        let op = MatVec {
            ty,
            weights: &weights,
            rows: ROWS,
            cols: COLS,
            x: &xs,
            x_offset: 0,
            y: &ys,
            y_offset: 0,
        };
        let run = || {
            context.submit(|r| {
                for _ in 0..REPEATS {
                    r.mat_vec(&op)?;
                }
                Ok(())
            })
        };
        run()?;
        let start = std::time::Instant::now();
        for _ in 0..SUBMITS {
            run()?;
        }
        let secs = start.elapsed().as_secs_f64();
        println!(
            "{}: Q8_0 matvec {:.1} GB/s, {:.0} us each",
            context.device_name(),
            (SUBMITS * REPEATS * len) as f64 / secs / 1e9,
            secs / (SUBMITS * REPEATS) as f64 * 1e6
        );
        Ok(())
    }
}
