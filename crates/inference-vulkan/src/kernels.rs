use std::sync::Arc;

use crate::map::{MAP_BINDINGS, MAP_PUSH_BYTES};
use crate::memory::{HostBuffer, ImportedMemory};
use crate::pipeline::Pipeline;
use crate::{Error, Recorder, Result, shaders};

// a GGUF Q8_0 block: an f16 scale and 32 int8 values
const Q8_0_VALUES: usize = 32;
const Q8_0_BYTES: usize = 34;
// rows one matvec workgroup takes (the shader's `ROWS`), and the most workgroups one grid dimension holds
const MATVEC_ROWS: usize = 2;
const MAX_GROUPS_X: usize = 65_535;
const MATVEC_PUSH_BYTES: u32 = 6 * 4;
const MATVEC_BINDINGS: usize = 3;

/// Every kernel's pipeline on one device.
pub(crate) struct Kernels {
    mat_vec_q8_0: Pipeline,
    pub(crate) map_f32: Pipeline,
    pub(crate) copy_b32: Pipeline,
}

impl Kernels {
    pub(crate) fn new(device: &ash::Device) -> Result<Self> {
        Ok(Self {
            mat_vec_q8_0: Pipeline::new(
                device,
                shaders::MUL_MAT_VEC_Q8_0,
                MATVEC_BINDINGS,
                MATVEC_PUSH_BYTES,
            )?,
            map_f32: Pipeline::new(device, shaders::MAP_F32, MAP_BINDINGS, MAP_PUSH_BYTES)?,
            copy_b32: Pipeline::new(device, shaders::COPY_B32, MAP_BINDINGS, MAP_PUSH_BYTES)?,
        })
    }

    /// # Safety
    /// No submit may still use the pipelines.
    pub(crate) unsafe fn destroy(&self, device: &ash::Device) {
        // SAFETY: the caller's contract
        unsafe {
            self.mat_vec_q8_0.destroy(device);
            self.map_f32.destroy(device);
            self.copy_b32.destroy(device);
        }
    }
}

/// `y[y_offset..][..rows] = W x[x_offset..][..cols]` for a `rows x cols` Q8_0 weight held in `weights`.
pub struct MatVec<'a> {
    pub weights: &'a Arc<ImportedMemory>,
    pub rows: usize,
    pub cols: usize,
    pub x: &'a Arc<HostBuffer>,
    pub x_offset: usize,
    pub y: &'a Arc<HostBuffer>,
    pub y_offset: usize,
}

impl Recorder<'_> {
    pub fn mat_vec_q8_0(&self, op: &MatVec<'_>) -> Result<()> {
        let blocks = op.cols / Q8_0_VALUES;
        let fits = op.cols.is_multiple_of(Q8_0_VALUES)
            && op.weights.offset.is_multiple_of(2)
            && op.weights.len() >= op.rows * blocks * Q8_0_BYTES
            && op.x.len() / 4 >= op.x_offset + op.cols
            && op.y.len() / 4 >= op.y_offset + op.rows;
        if !fits {
            return Err(Error::Invalid(format!(
                "a {}x{} Q8_0 matvec over {} weight bytes at offset {}",
                op.rows,
                op.cols,
                op.weights.len(),
                op.weights.offset
            )));
        }
        let groups = op.rows.div_ceil(MATVEC_ROWS);
        let groups_x = groups.min(MAX_GROUPS_X);
        let push = [
            op.rows,
            op.cols,
            op.weights.offset as usize / 2,
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
        self.context.kernels.mat_vec_q8_0.dispatch(
            self,
            &[op.weights.buffer, op.x.buffer, op.y.buffer],
            &bytes,
            [groups_x as u32, groups.div_ceil(groups_x) as u32, 1],
        );
        self.retain(op.weights.clone());
        self.retain(op.x.clone());
        self.retain(op.y.clone());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Context;
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

    // random Q8_0 blocks with small finite scales
    fn q8_0_weights(bytes: &mut [u8], rows: usize, cols: usize) {
        let mut state = 0x2545_f491u32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        let len = rows * cols / Q8_0_VALUES * Q8_0_BYTES;
        for block in bytes[..len].as_chunks_mut::<Q8_0_BYTES>().0 {
            let d = half::f16::from_f32(0.001 + (next() % 100) as f32 * 1e-5);
            block[..2].copy_from_slice(&d.to_le_bytes());
            block[2..].iter_mut().for_each(|q| *q = next() as u8);
        }
    }

    fn reference(bytes: &[u8], rows: usize, cols: usize, x: &[f32]) -> Vec<f64> {
        let blocks = cols / Q8_0_VALUES;
        (0..rows)
            .map(|r| {
                (0..blocks)
                    .map(|b| {
                        let block = &bytes[(r * blocks + b) * Q8_0_BYTES..][..Q8_0_BYTES];
                        let d = f64::from(half::f16::from_le_bytes([block[0], block[1]]).to_f32());
                        let dot: f64 = block[2..]
                            .iter()
                            .zip(&x[b * Q8_0_VALUES..])
                            .map(|(&q, &x)| f64::from(q as i8) * f64::from(x))
                            .sum();
                        d * dot
                    })
                    .sum()
            })
            .collect()
    }

    // rows past one workgroup's, an odd count, and a weight that starts mid-page as a GGUF tensor's does
    #[test]
    fn a_q8_0_matvec_matches_the_cpu() -> Result<()> {
        let Some(context) = context() else {
            return Ok(());
        };
        const ROWS: usize = 333;
        const COLS: usize = 2688;
        const WEIGHT_AT: usize = 96;
        let mut pages = Pages::new(WEIGHT_AT + ROWS * COLS / Q8_0_VALUES * Q8_0_BYTES);
        let weight_bytes = &mut pages.bytes()[WEIGHT_AT..];
        q8_0_weights(weight_bytes, ROWS, COLS);
        let x: Vec<f32> = (0..COLS)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 25.0)
            .collect();
        let want = reference(weight_bytes, ROWS, COLS, &x);
        // SAFETY: `pages` outlives the import and is not written while it lives
        let weights =
            Arc::new(unsafe { ImportedMemory::new(&context, &pages.bytes()[WEIGHT_AT..])? });
        let mut xs = HostBuffer::new(&context, (COLS + 8) * 4)?;
        xs.as_f32_mut()[8..8 + COLS].copy_from_slice(&x);
        let (xs, ys) = (
            Arc::new(xs),
            Arc::new(HostBuffer::new(&context, (ROWS + 4) * 4)?),
        );
        context.submit(|r| {
            r.mat_vec_q8_0(&MatVec {
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
                "row {row}: {got} vs {want}"
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
        let len = ROWS * COLS / Q8_0_VALUES * Q8_0_BYTES;
        let mut pages = Pages::new(len);
        q8_0_weights(pages.bytes(), ROWS, COLS);
        let x: Vec<f32> = (0..COLS).map(|i| (i % 7) as f32 - 3.0).collect();
        let want = reference(pages.bytes(), ROWS, COLS, &x);
        let ys = Arc::new(HostBuffer::new(&context, DISPATCHES * ROWS * 4)?);
        {
            // SAFETY: `pages` outlives every submit, which `synchronize` below waits for
            let weights =
                Arc::new(unsafe { ImportedMemory::new(&context, &pages.bytes()[..len])? });
            let mut xs = HostBuffer::new(&context, COLS * 4)?;
            xs.as_f32_mut().copy_from_slice(&x);
            let xs = Arc::new(xs);
            for d in 0..DISPATCHES {
                context.record(|r| {
                    r.mat_vec_q8_0(&MatVec {
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
        let len = ROWS * COLS / Q8_0_VALUES * Q8_0_BYTES;
        let mut pages = Pages::new(len);
        q8_0_weights(pages.bytes(), ROWS, COLS);
        // SAFETY: as in the test above
        let weights = Arc::new(unsafe { ImportedMemory::new(&context, &pages.bytes()[..len])? });
        let mut xs = HostBuffer::new(&context, COLS * 4)?;
        xs.as_f32_mut().iter_mut().for_each(|x| *x = 0.5);
        let (xs, ys) = (Arc::new(xs), Arc::new(HostBuffer::new(&context, ROWS * 4)?));
        let op = MatVec {
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
                    r.mat_vec_q8_0(&op)?;
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
