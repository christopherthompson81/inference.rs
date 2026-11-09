//! Row ops (RMS norm, softmax, reductions), row selection (index_select, gather) and a strided batched f32 matmul.

use std::sync::Arc;

use crate::memory::Buffer;
use crate::{Error, Half, Recorder, Result};

pub(crate) const BINDINGS: usize = 3;
pub(crate) const ROWS_BINDINGS: usize = 4;
pub(crate) const ROWS_PUSH_BYTES: u32 = 8 * 4;
pub(crate) const INDEX_PUSH_BYTES: u32 = 8 * 4;
pub(crate) const MATMUL_PUSH_BYTES: u32 = 12 * 4;
pub(crate) const CONV2D_PUSH_BYTES: u32 = 20 * 4;
pub(crate) const IM2COL_BINDINGS: usize = 2;
// a dense convolution's window (c_in * k_h * k_w) from which it runs as a matmul over its unfolded input
const UNFOLD_MIN_K: usize = 64;
// conv2d_f32's outputs a workgroup
const CONV2D_THREADS: usize = 256;
const MAX_GROUPS_X: usize = 65_535;
const INDEX_THREADS: usize = 256;
const MATMUL_THREADS: usize = 64;
// the inner length from which an output element gets a workgroup of its own, its threads splitting the dot product
const DOT_MIN_K: usize = 64;
// the tiled kernel's output tile, its smallest worthwhile rows and columns, and the batches its grid's z takes
const TILE_M: usize = 64;
const TILE_N: usize = 64;
const TILED_MIN: usize = 16;
// matmul_f32_tiled32's tile, taken below this many 64 x 64 tiles
const SMALL_TILE: usize = 32;
// matmul_f32_cm's tile
const CM_TILE: usize = 128;
const CM_SMALL_TILE: usize = 64;
// a discrete GPU's matrix-core product with fewer 128 x 128 tiles than this takes 64 x 64 ones (the 3090's 82 SMs ran
// Sortformer's 684 x 512 x 2048 in 24 workgroups, 10 TFLOPS)
const CM_SMALL_BELOW: usize = 160;
const SMALL_TILE_BELOW: usize = 256;
const MAX_GROUPS_Z: usize = 65_535;
const MAX_GROUPS: usize = 4096;

/// A per-row op, numbered as `rows_f32.comp`'s constants.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(u32)]
pub enum RowOp {
    /// `x / sqrt(mean(x^2) + eps) * alpha`
    RmsNorm = 0,
    Softmax = 1,
    Sum = 2,
    Max = 3,
    Min = 4,
    /// the index of the row's largest element (the first of equals), its bits in the f32 output
    ArgMax = 5,
    ArgMin = 6,
    /// `(x - mean(x)) / sqrt(var(x) + eps) * alpha + beta`
    LayerNorm = 7,
}

impl RowOp {
    /// Whether the op writes one value a row rather than a row.
    pub fn reduces(self) -> bool {
        !matches!(self, RowOp::RmsNorm | RowOp::Softmax | RowOp::LayerNorm)
    }
}

/// A direct 2D convolution: `x` `(batch, groups * c_in, i_h, i_w)` at `x_strides`, `w` contiguous `(groups * c_out,
/// c_in, k_h, k_w)`, into a contiguous `(batch, groups * c_out, o_h, o_w)` `out`; zero padding.
pub struct Conv2d<'a> {
    pub batch: usize,
    pub c_in: usize,
    pub c_out: usize,
    pub groups: usize,
    pub i_h: usize,
    pub i_w: usize,
    pub k_h: usize,
    pub k_w: usize,
    pub o_h: usize,
    pub o_w: usize,
    /// along h and w
    pub padding: (usize, usize),
    pub stride: usize,
    pub dilation: usize,
    pub x: At<'a>,
    pub x_strides: [usize; 4],
    pub w: At<'a>,
    pub out: &'a Arc<Buffer>,
}

/// `rows` contiguous rows of `cols` from `x_offset`; `alpha` and `beta` (`cols` from their offsets) are read by the norms
/// alone, `beta` by `LayerNorm`.
pub struct Rows<'a> {
    pub op: RowOp,
    pub rows: usize,
    pub cols: usize,
    pub eps: f32,
    pub x: &'a Arc<Buffer>,
    pub x_offset: usize,
    pub alpha: Option<(&'a Arc<Buffer>, usize)>,
    pub beta: Option<(&'a Arc<Buffer>, usize)>,
    pub out: &'a Arc<Buffer>,
    pub out_offset: usize,
}

/// A residual add and the RMS norm of its sum, for a decode's few rows: `sum = x + residual`, `out` its norm times
/// `alpha`, both `rows x cols` from offset 0; see [`AddRmsNorm::fits`].
pub struct AddRmsNorm<'a> {
    pub rows: usize,
    pub cols: usize,
    pub eps: f32,
    pub x: At<'a>,
    pub residual: At<'a>,
    pub alpha: At<'a>,
    pub sum: &'a Arc<Buffer>,
    pub out: &'a Arc<Buffer>,
}

impl AddRmsNorm<'_> {
    /// Whether the fused kernel takes `rows` rows of `cols`.
    pub fn fits(rows: usize, cols: usize) -> bool {
        rows > 0
            && rows <= crate::kernels::MATVEC_MAX_ROWS
            && cols.is_multiple_of(crate::kernels::BLOCK_VALUES)
            && cols <= RMS_Q8_MAX_COLS
    }
}

// one rms_norm_q8 dispatch (of AddRmsNorm::fits), with the residual and where the sum goes for an add
struct RmsQ8<'a> {
    rows: usize,
    cols: usize,
    eps: f32,
    x: At<'a>,
    alpha: At<'a>,
    out: At<'a>,
    add: Option<(At<'a>, At<'a>)>,
}

/// Row selection over 32-bit words of `src`, contiguous as `(left, src_dim, right)`, into a contiguous `out`.
pub struct Select<'a> {
    pub left: usize,
    pub src_dim: usize,
    pub right: usize,
    /// ids a (left, right) position selects among
    pub n: usize,
    /// `ids` laid out as the output (gather) rather than `n` ids shared by every position (index_select)
    pub gather: bool,
    pub src: (&'a Arc<Buffer>, usize),
    pub ids: (&'a Arc<Buffer>, usize),
    pub out: &'a Arc<Buffer>,
}

/// One operand of [`MatMul`]: its buffer, first element, and strides between batches, rows and columns.
#[derive(Clone, Copy)]
pub struct MatOperand<'a> {
    pub buffer: &'a Arc<Buffer>,
    pub offset: usize,
    pub batch_stride: usize,
    pub row_stride: usize,
    pub col_stride: usize,
}

/// `out[b] = a[b] (m x k) times b[b] (k x n)`, `out` contiguous.
pub struct MatMul<'a> {
    pub batch: usize,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub a: MatOperand<'a>,
    pub b: MatOperand<'a>,
    pub out: &'a Arc<Buffer>,
}

// a convolution's push constants, as conv2d_push.glsl reads them
fn conv2d_words(op: &Conv2d<'_>) -> Result<Vec<u8>> {
    let [sb, sc, sh, sw] = op.x_strides;
    words(&[
        op.batch,
        op.c_in,
        op.c_out,
        op.groups,
        op.i_h,
        op.i_w,
        op.k_h,
        op.k_w,
        op.o_h,
        op.o_w,
        op.padding.0,
        op.padding.1,
        op.stride,
        op.dilation,
        op.x.1,
        sb,
        sc,
        sh,
        sw,
        op.w.1,
    ])
}

fn words(values: &[usize]) -> Result<Vec<u8>> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for &v in values {
        let w = u32::try_from(v)
            .map_err(|_| Error::Invalid(format!("kernel constant {v} past u32")))?;
        bytes.extend_from_slice(&w.to_le_bytes());
    }
    Ok(bytes)
}

impl Recorder<'_> {
    pub fn rows_f32(&self, op: &Rows<'_>) -> Result<()> {
        self.rows(op, None)
    }

    /// [`Self::rows_f32`] over 16-bit float rows, `alpha` and `beta`, computed in f32 (an arg reduction's out still
    /// u32s).
    pub fn rows_half(&self, op: &Rows<'_>, half: Half) -> Result<()> {
        self.rows(op, Some(half))
    }

    fn rows(&self, op: &Rows<'_>, half: Option<Half>) -> Result<()> {
        if op.rows == 0 {
            return Ok(());
        }
        let (alpha, alpha_offset) = op.alpha.unwrap_or((op.x, 0));
        if op.op == RowOp::RmsNorm && op.alpha.is_none() {
            return Err(Error::Invalid("an RMS norm takes its alpha".into()));
        }
        if op.op == RowOp::LayerNorm && (op.alpha.is_none() || op.beta.is_none()) {
            return Err(Error::Invalid(
                "a layer norm takes its alpha and beta".into(),
            ));
        }
        let (beta, beta_offset) = op.beta.unwrap_or((op.x, 0));
        // a decode row's norm writes its int8 blocks too, cached for the matvecs that read it
        if half.is_none()
            && op.op == RowOp::RmsNorm
            && AddRmsNorm::fits(op.rows, op.cols)
            && !Arc::ptr_eq(op.x, op.out)
        {
            return self.rms_norm_q8(&RmsQ8 {
                rows: op.rows,
                cols: op.cols,
                eps: op.eps,
                x: (op.x, op.x_offset),
                alpha: (alpha, alpha_offset),
                out: (op.out, op.out_offset),
                add: None,
            });
        }
        let mut bytes = words(&[op.rows, op.cols, op.op as usize])?;
        bytes.extend_from_slice(&op.eps.to_bits().to_le_bytes());
        bytes.extend(words(&[
            op.x_offset,
            alpha_offset,
            op.out_offset,
            beta_offset,
        ])?);
        let groups_x = op.rows.min(MAX_GROUPS_X);
        let kernels = &self.context.kernels;
        let pipeline = match half {
            None => &kernels.rows_f32,
            Some(Half::Bf16) => &kernels.rows_bf16,
            Some(Half::F16) => &kernels.rows_f16,
        };
        pipeline.dispatch(
            self,
            &[op.x.buffer, alpha.buffer, op.out.buffer, beta.buffer],
            &bytes,
            [groups_x as u32, op.rows.div_ceil(groups_x) as u32, 1],
        );
        for buffer in [op.x, alpha, op.out, beta] {
            self.retain(buffer.clone());
        }
        Ok(())
    }

    pub fn add_rms_norm(&self, op: &AddRmsNorm<'_>) -> Result<()> {
        let len = op.rows * op.cols;
        let fits = AddRmsNorm::fits(op.rows, op.cols)
            && op.x.0.len() / 4 >= op.x.1 + len
            && op.residual.0.len() / 4 >= op.residual.1 + len
            && op.alpha.0.len() / 4 >= op.alpha.1 + op.cols
            && op.sum.len() / 4 >= len
            && op.out.len() / 4 >= len
            && !Arc::ptr_eq(op.x.0, op.out)
            && !Arc::ptr_eq(op.residual.0, op.out);
        if !fits {
            return Err(Error::Invalid(format!(
                "an add and RMS norm of {} rows of {}",
                op.rows, op.cols
            )));
        }
        self.rms_norm_q8(&RmsQ8 {
            rows: op.rows,
            cols: op.cols,
            eps: op.eps,
            x: op.x,
            alpha: op.alpha,
            out: (op.out, 0),
            add: Some((op.residual, (op.sum, 0))),
        })
    }

    // its int8 blocks cached for the matvecs reading `out`
    fn rms_norm_q8(&self, op: &RmsQ8<'_>) -> Result<()> {
        let RmsQ8 {
            rows,
            cols,
            eps,
            x,
            alpha,
            out,
            add,
        } = *op;
        let blocks = Arc::new(Buffer::new(
            out.0.context(),
            crate::MatVec::scratch_bytes(rows, cols),
        )?);
        let s_off = rows * cols / crate::kernels::BLOCK_VALUES * crate::kernels::Q8_WORD_BYTES
            / crate::kernels::Q8_SCALE_BYTES;
        // without an add, x stands in for the residual and the sum, neither read nor written
        let (r, sum) = add.unwrap_or((x, x));
        let mut bytes = words(&[rows, cols])?;
        bytes.extend_from_slice(&eps.to_bits().to_le_bytes());
        bytes.extend(words(&[
            x.1,
            alpha.1,
            out.1,
            s_off,
            usize::from(add.is_some()),
            r.1,
            sum.1,
        ])?);
        self.context.kernels.rms_norm_q8.dispatch(
            self,
            &[
                x.0.buffer,
                alpha.0.buffer,
                out.0.buffer,
                blocks.buffer,
                blocks.buffer,
                r.0.buffer,
                sum.0.buffer,
            ],
            &bytes,
            [rows as u32, 1, 1],
        );
        self.context
            .remember_quantized(out.0, out.1, rows * cols, &blocks, s_off);
        for buffer in [x.0, alpha.0, out.0, &blocks, r.0, sum.0] {
            self.retain(buffer.clone());
        }
        Ok(())
    }

    pub fn select_b32(&self, op: &Select<'_>) -> Result<()> {
        let count = op.left * op.n * op.right;
        if count == 0 {
            return Ok(());
        }
        let bytes = words(&[
            op.left,
            op.src_dim,
            op.right,
            op.n,
            usize::from(op.gather),
            op.src.1,
            op.ids.1,
            0,
        ])?;
        let groups = count.div_ceil(INDEX_THREADS).min(MAX_GROUPS) as u32;
        self.context.kernels.index_b32.dispatch(
            self,
            &[op.src.0.buffer, op.ids.0.buffer, op.out.buffer],
            &bytes,
            [groups, 1, 1],
        );
        for buffer in [op.src.0, op.ids.0, op.out] {
            self.retain(buffer.clone());
        }
        Ok(())
    }

    pub fn conv2d_f32(&self, op: &Conv2d<'_>) -> Result<()> {
        self.conv2d(op, None)
    }

    /// [`Self::conv2d_f32`] over 16-bit float operands, summed in f32.
    pub fn conv2d_half(&self, op: &Conv2d<'_>, half: Half) -> Result<()> {
        self.conv2d(op, Some(half))
    }

    fn conv2d(&self, op: &Conv2d<'_>, half: Option<Half>) -> Result<()> {
        let count = op.batch * op.groups * op.c_out * op.o_h * op.o_w;
        if count == 0 {
            return Ok(());
        }
        if let Some(half) = half
            && self.conv2d_unfolded(op, half)?
        {
            return Ok(());
        }
        let bytes = conv2d_words(op)?;
        let groups = count.div_ceil(CONV2D_THREADS);
        let groups_x = groups.min(MAX_GROUPS_X);
        let buffers = [op.x.0, op.w.0, op.out];
        let work = count * op.c_in * op.k_h * op.k_w;
        self.profile_shape(
            || {
                format!(
                    "b={} c={}x{}/{} {}x{} k={}x{} s={}",
                    op.batch,
                    op.c_in,
                    op.c_out,
                    op.groups,
                    op.i_h,
                    op.i_w,
                    op.k_h,
                    op.k_w,
                    op.stride
                )
            },
            work,
        );
        let kernels = &self.context.kernels;
        let pipeline = match half {
            None => &kernels.conv2d_f32,
            Some(Half::Bf16) => &kernels.conv2d_bf16,
            Some(Half::F16) => &kernels.conv2d_f16,
        };
        pipeline.dispatch(
            self,
            &buffers.map(|b| b.buffer),
            &bytes,
            [groups_x as u32, groups.div_ceil(groups_x) as u32, 1],
        );
        self.work.set(self.work.get().saturating_add(work as u64));
        for buffer in buffers {
            self.retain(buffer.clone());
        }
        Ok(())
    }

    // a dense 16-bit convolution as a matmul over its unfolded input, on matrix cores; whether it took it (a thread an
    // output, the direct kernel ran Sortformer's 192 -> 1536 conv1d at 0.2 TFLOPS)
    fn conv2d_unfolded(&self, op: &Conv2d<'_>, half: Half) -> Result<bool> {
        let k = op.c_in * op.k_h * op.k_w;
        let n = op.o_h * op.o_w;
        if op.groups != 1
            || k < UNFOLD_MIN_K
            || op.c_out < TILED_MIN
            || n < TILED_MIN
            || op.batch > MAX_GROUPS_Z
            || self.context.matmul_coop().is_none()
        {
            return Ok(false);
        }
        let count = op.batch * k * n;
        let cols = Arc::new(Buffer::new(op.out.context(), count * 2)?);
        let groups = count.div_ceil(CONV2D_THREADS);
        let groups_x = groups.min(MAX_GROUPS_X);
        self.context.kernels.im2col_bf16.dispatch(
            self,
            &[op.x.0.buffer, cols.buffer],
            &conv2d_words(op)?,
            [groups_x as u32, groups.div_ceil(groups_x) as u32, 1],
        );
        self.retain(op.x.0.clone());
        self.retain(cols.clone());
        let weight = MatOperand {
            buffer: op.w.0,
            offset: op.w.1,
            batch_stride: 0,
            row_stride: k,
            col_stride: 1,
        };
        let unfolded = MatOperand {
            buffer: &cols,
            offset: 0,
            batch_stride: k * n,
            row_stride: n,
            col_stride: 1,
        };
        self.matmul_half(
            &MatMul {
                batch: op.batch,
                m: op.c_out,
                n,
                k,
                a: weight,
                b: unfolded,
                out: op.out,
            },
            half,
        )
    }

    /// [`Self::matmul_f32`] with `op.b` stored as f16 (a weight kept in f16), on matrix cores; `false` (nothing recorded)
    /// where the device has none or the product is too small for their tiles.
    pub fn matmul_f32_half_b(&self, op: &MatMul<'_>) -> Result<bool> {
        let Some(coop) = self.context.coop.as_ref() else {
            return Ok(false);
        };
        self.matmul_cm(op, [&coop.matmul_f32_bh, &coop.matmul_cm64[1]], "half b")
    }

    /// [`Self::matmul_f32`] over 16-bit float operands into an `out` of the same: on matrix cores (the inputs rounded
    /// to f16 as the f32 product's are), and a product too small for their tiles in f32 sums; `false` (nothing
    /// recorded) for a tiled product on a device without them.
    pub fn matmul_half(&self, op: &MatMul<'_>, half: Half) -> Result<bool> {
        let tiled = op.m >= TILED_MIN && op.n >= TILED_MIN && op.batch <= MAX_GROUPS_Z;
        if tiled {
            return match self.context.coop.as_ref() {
                Some(coop) => match half {
                    Half::Bf16 => {
                        self.matmul_cm(op, [&coop.matmul_bf16_cm, &coop.matmul_cm64[2]], "bf16")
                    }
                    Half::F16 => {
                        self.matmul_cm(op, [&coop.matmul_f16_cm, &coop.matmul_cm64[3]], "f16")
                    }
                },
                None => Ok(false),
            };
        }
        let count = op.batch * op.m * op.n;
        if count == 0 {
            return Ok(true);
        }
        let (a, b) = (op.a, op.b);
        let bytes = words(&[
            op.batch,
            op.m,
            op.n,
            op.k,
            a.offset,
            a.batch_stride,
            a.row_stride,
            a.col_stride,
            b.offset,
            b.batch_stride,
            b.row_stride,
            b.col_stride,
        ])?;
        let buffers = [a.buffer.buffer, b.buffer.buffer, op.out.buffer];
        let work = (count as u64).saturating_mul(op.k as u64);
        self.work.set(self.work.get().saturating_add(work));
        let kernels = &self.context.kernels;
        let (dot, simple) = match half {
            Half::Bf16 => (&kernels.matmul_bf16_dot, &kernels.matmul_bf16),
            Half::F16 => (&kernels.matmul_f16_dot, &kernels.matmul_f16),
        };
        if op.k >= DOT_MIN_K {
            let groups_x = count.min(MAX_GROUPS_X);
            dot.dispatch(
                self,
                &buffers,
                &bytes,
                [groups_x as u32, count.div_ceil(groups_x) as u32, 1],
            );
        } else {
            let groups = count.div_ceil(MATMUL_THREADS).min(MAX_GROUPS) as u32;
            simple.dispatch(self, &buffers, &bytes, [groups, 1, 1]);
        }
        for buffer in [a.buffer, b.buffer, op.out] {
            self.retain(buffer.clone());
        }
        Ok(true)
    }

    // `builds`: the 128 x 128 tile's, then the 64 x 64 one's
    fn matmul_cm(
        &self,
        op: &MatMul<'_>,
        builds: [&crate::pipeline::Pipeline; 2],
        kind: &str,
    ) -> Result<bool> {
        if op.m < TILED_MIN || op.n < TILED_MIN || op.batch > MAX_GROUPS_Z {
            return Ok(false);
        }
        let (a, b) = (op.a, op.b);
        let bytes = words(&[
            op.batch,
            op.m,
            op.n,
            op.k,
            a.offset,
            a.batch_stride,
            a.row_stride,
            a.col_stride,
            b.offset,
            b.batch_stride,
            b.row_stride,
            b.col_stride,
        ])?;
        let count = op.batch * op.m * op.n;
        self.work.set(
            self.work
                .get()
                .saturating_add((count as u64).saturating_mul(op.k as u64)),
        );
        self.profile_shape(
            || format!("b={} m={} n={} k={} {kind}", op.batch, op.m, op.n, op.k),
            count * op.k,
        );
        let (pipeline, tile) = self.cm_build(op, builds);
        pipeline.dispatch(
            self,
            &[a.buffer.buffer, b.buffer.buffer, op.out.buffer],
            &bytes,
            [
                op.n.div_ceil(tile) as u32,
                op.m.div_ceil(tile) as u32,
                op.batch as u32,
            ],
        );
        for buffer in [a.buffer, b.buffer, op.out] {
            self.retain(buffer.clone());
        }
        Ok(true)
    }

    // the matrix-core build a product takes, and its tile; 64 x 64 on a discrete GPU alone, the iGPU's unmeasured
    fn cm_build<'p>(
        &self,
        op: &MatMul<'_>,
        [large, small]: [&'p crate::pipeline::Pipeline; 2],
    ) -> (&'p crate::pipeline::Pipeline, usize) {
        let tiles = op.n.div_ceil(CM_TILE) * op.m.div_ceil(CM_TILE) * op.batch;
        if !self.context.is_unified() && tiles < CM_SMALL_BELOW {
            (small, CM_SMALL_TILE)
        } else {
            (large, CM_TILE)
        }
    }

    pub fn matmul_f32(&self, op: &MatMul<'_>) -> Result<()> {
        let count = op.batch * op.m * op.n;
        if count == 0 {
            return Ok(());
        }
        let (a, b) = (op.a, op.b);
        let bytes = words(&[
            op.batch,
            op.m,
            op.n,
            op.k,
            a.offset,
            a.batch_stride,
            a.row_stride,
            a.col_stride,
            b.offset,
            b.batch_stride,
            b.row_stride,
            b.col_stride,
        ])?;
        let buffers = [a.buffer.buffer, b.buffer.buffer, op.out.buffer];
        let work = (count as u64).saturating_mul(op.k as u64);
        self.work.set(self.work.get().saturating_add(work));
        self.profile_shape(
            || format!("b={} m={} n={} k={}", op.batch, op.m, op.n, op.k),
            count * op.k,
        );
        let tiled = op.m >= TILED_MIN && op.n >= TILED_MIN && op.batch <= MAX_GROUPS_Z;
        // on an iGPU too, unlike the quantized matmuls: f32 has no int8 dot product to lose to (a 780M's Parakeet
        // encoder products ran ~1 TFLOPS tiled, 1.5-2.4 here)
        if tiled && let Some(coop) = self.context.coop.as_ref() {
            let (pipeline, tile) = self.cm_build(op, [&coop.matmul_f32, &coop.matmul_cm64[0]]);
            pipeline.dispatch(
                self,
                &buffers,
                &bytes,
                [
                    op.n.div_ceil(tile) as u32,
                    op.m.div_ceil(tile) as u32,
                    op.batch as u32,
                ],
            );
        } else if tiled {
            // a product whose 64 x 64 tiles leave most of the device idle (a router's 128 experts) takes 32 x 32
            let tiles = op.n.div_ceil(TILE_N) * op.m.div_ceil(TILE_M) * op.batch;
            let (pipeline, tile_m, tile_n) = if tiles < SMALL_TILE_BELOW {
                (
                    &self.context.kernels.matmul_f32_tiled32,
                    SMALL_TILE,
                    SMALL_TILE,
                )
            } else {
                (&self.context.kernels.matmul_f32_tiled, TILE_M, TILE_N)
            };
            pipeline.dispatch(
                self,
                &buffers,
                &bytes,
                [
                    op.n.div_ceil(tile_n) as u32,
                    op.m.div_ceil(tile_m) as u32,
                    op.batch as u32,
                ],
            );
        } else if op.k >= DOT_MIN_K {
            let groups_x = count.min(MAX_GROUPS_X);
            self.context.kernels.matmul_f32_dot.dispatch(
                self,
                &buffers,
                &bytes,
                [groups_x as u32, count.div_ceil(groups_x) as u32, 1],
            );
        } else {
            let groups = count.div_ceil(MATMUL_THREADS).min(MAX_GROUPS) as u32;
            self.context
                .kernels
                .matmul_f32
                .dispatch(self, &buffers, &bytes, [groups, 1, 1]);
        }
        for buffer in [a.buffer, b.buffer, op.out] {
            self.retain(buffer.clone());
        }
        Ok(())
    }
}

/// A buffer and the element an operand starts at.
pub type At<'a> = (&'a Arc<Buffer>, usize);

pub(crate) const CONV_STEP_BINDINGS: usize = 6;
pub(crate) const CONV_STEP_PUSH_BYTES: u32 = 10 * 4;
pub(crate) const SCAN_STEP_BINDINGS: usize = 10;
pub(crate) const SCAN_STEP_PUSH_BYTES: u32 = 14 * 4;
pub(crate) const ROPE_BINDINGS: usize = 5;
pub(crate) const ROPE_PUSH_BYTES: u32 = 11 * 4;
const ROPE_THREADS: usize = 256;
pub(crate) const GDN_BINDINGS: usize = 8;
pub(crate) const GDN_PUSH_BYTES: u32 = 14 * 4;
// gdn_recurrence.comp's threads, a value column each, and the key rows a thread's registers hold
const GDN_MAX_HEAD: usize = 128;
// the scan step's lanes a state row and states a lane, and the rows a 256-thread workgroup takes
const SCAN_LANES: usize = 32;
const SCAN_VEC: usize = 4;
const SCAN_ROWS_PER_GROUP: usize = 256 / SCAN_LANES;
pub(crate) const SSD_BINDINGS: usize = 11;
// the chunked scan's positions a chunk, threads a state-pass workgroup and widest head (ssd_head.glsl's)
const SSD_CHUNK: usize = 32;
const SSD_THREADS: usize = 256;
const SSD_MAX_HEAD_DIM: usize = 128;
/// The shortest sequence a scan with scratch runs chunked; shorter ones step in order.
pub const SSD_MIN_SEQ: usize = 4 * SSD_CHUNK;

/// `seq` steps of Mamba's causal conv on a conv-state pool (`(rows, channels, kernel)` f32, in place), each batch
/// row's state the pool row `slots` names; `x` and `out` are `(batch, seq, channels)`.
pub struct ConvStep<'a> {
    pub batch: usize,
    pub seq: usize,
    pub channels: usize,
    pub kernel: usize,
    pub x: At<'a>,
    pub weight: At<'a>,
    pub bias: Option<At<'a>>,
    pub pool: At<'a>,
    pub slots: At<'a>,
    pub out: &'a Arc<Buffer>,
}

/// Mamba-2's gated RMSNorm: `rows` rows of `x` and `gate` (`groups * group_size` f32 each), each value gated by
/// silu(gate), each group normalized by its RMS, and all scaled by `weight` (`groups * group_size`).
pub struct GatedRmsNorm<'a> {
    pub rows: usize,
    pub groups: usize,
    pub group_size: usize,
    pub eps: f32,
    /// Gated DeltaNet's order, the norm of x then the gate, rather than Mamba-2's norm of the gated x
    pub norm_first: bool,
    /// a sigmoid gate rather than SiLU
    pub sigmoid: bool,
    pub x: At<'a>,
    pub gate: At<'a>,
    pub weight: At<'a>,
    pub out: &'a Arc<Buffer>,
}

/// `seq` steps of Mamba-2's selective scan on a state pool (`(rows, heads, head_dim, state)` f32, in place); `a`, `d`
/// and `dt_bias` are `(heads,)` from their buffers' first element; `out` is `(batch, seq, heads, head_dim)`.
pub struct ScanStep<'a> {
    pub batch: usize,
    pub seq: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub groups: usize,
    pub state: usize,
    pub dt_range: [f32; 2],
    pub x: At<'a>,
    pub dt: At<'a>,
    pub a: &'a Arc<Buffer>,
    pub b: At<'a>,
    pub c: At<'a>,
    pub d: &'a Arc<Buffer>,
    pub dt_bias: &'a Arc<Buffer>,
    pub pool: At<'a>,
    pub slots: At<'a>,
    pub out: &'a Arc<Buffer>,
    /// Scratch of [`Self::chunked_scratch_bytes`], with which a sequence of [`SSD_MIN_SEQ`] or more runs as chunked
    /// SSD (each chunk's scan as dot products, only the state carried between chunks in order).
    pub scratch: Option<&'a Arc<Buffer>>,
}

/// Gated DeltaNet's recurrence over `seq` tokens of `batch` sequences from the convolved rows `mixed` (a token's
/// `conv_dim` f32s: q and k `key_dim` each, then v), the gate inputs `b` and `a` `(tokens, v_heads)`, `a_log` and
/// `dt_bias` `(v_heads,)` from their buffers' start; reads `state_in` `(batch, v_heads, head_k, head_v)` key-major, writes
/// the advanced state to `state_out` and `(tokens, v_heads, head_v)` to `out`.
pub struct GdnRecurrence<'a> {
    pub batch: usize,
    pub seq: usize,
    pub k_heads: usize,
    pub v_heads: usize,
    pub head_k: usize,
    pub head_v: usize,
    pub conv_dim: usize,
    pub key_dim: usize,
    pub tiled: bool,
    pub eps: f32,
    pub mixed: At<'a>,
    pub b: At<'a>,
    pub a: At<'a>,
    pub a_log: &'a Arc<Buffer>,
    pub dt_bias: &'a Arc<Buffer>,
    pub state_in: At<'a>,
    /// the advanced state: `state_in`'s own rows (in place) or a fresh buffer's
    pub state_out: At<'a>,
    pub out: &'a Arc<Buffer>,
}

/// Where a RoPE row's cos/sin come from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RopeRows {
    /// The u32 position a token: `positions[b * seq + t]`.
    Positions,
    /// The token's own row (`batch * seq` rows).
    Tokens,
    /// The token's sequence index (`seq` rows).
    Sequence,
}

/// RoPE over `x` `(batch, heads, seq, head_dim)` f32 with `rot`-wide cos/sin rows, into `out` (x's shape).
pub struct Rope<'a> {
    pub batch: usize,
    pub heads: usize,
    pub seq: usize,
    pub head_dim: usize,
    pub rot: usize,
    pub rows: RopeRows,
    pub neox: bool,
    pub x: At<'a>,
    pub cos: At<'a>,
    pub sin: At<'a>,
    pub positions: At<'a>,
    pub out: &'a Arc<Buffer>,
}

impl GdnRecurrence<'_> {
    /// Whether the kernel takes heads of these widths.
    pub fn supports(head_k: usize, head_v: usize) -> bool {
        (1..=GDN_MAX_HEAD).contains(&head_k) && (1..=GDN_MAX_HEAD).contains(&head_v)
    }
}

impl ScanStep<'_> {
    /// The scratch a chunked scan of this shape takes, or `None` where it steps in order.
    pub fn chunked_scratch_bytes(
        batch: usize,
        seq: usize,
        heads: usize,
        head_dim: usize,
    ) -> Option<usize> {
        if seq < SSD_MIN_SEQ || head_dim > SSD_MAX_HEAD_DIM {
            return None;
        }
        let chunks = seq.div_ceil(SSD_CHUNK);
        let bytes = batch * chunks * heads * (SSD_CHUNK + head_dim * SCAN_LANES * SCAN_VEC) * 4;
        (bytes as u64 <= crate::memory::MAX_BINDING_BYTES).then_some(bytes)
    }
}

impl MoeMatVec<'_> {
    // the tiles moe_sort cuts `slots` routes over `experts` into, at most
    fn max_tiles(slots: usize, experts: usize) -> usize {
        slots.div_ceil(MOE_TILE) + experts
    }

    /// The activation scratch `slots` routes over rows of `cols` take on `context`, `x_div` routes a row: its rows as
    /// int8 blocks, and where the routes may run on matrix cores, their rows as f16 in expert order, padded a tile.
    pub fn scratch_bytes(
        context: &crate::Context,
        slots: usize,
        x_div: usize,
        cols: usize,
    ) -> usize {
        let int8 = crate::MatVec::scratch_bytes(slots.div_ceil(x_div), cols);
        if context.matmul_coop().is_none() {
            return int8;
        }
        // the f16 rows in expert order with a tile of zeros past them (the matrix-core paths quantize nothing)
        int8.max((slots + MOE_GATHER_PAD) * cols * 2)
    }

    /// The routing scratch `slots` routes over `experts` take: their order, the tile count and the tiles.
    pub fn routing_bytes(slots: usize, experts: usize) -> usize {
        (slots + 1 + 3 * Self::max_tiles(slots, experts)) * 4
    }
}

impl Recorder<'_> {
    pub fn ssm_conv_step(&self, op: &ConvStep<'_>) -> Result<()> {
        let (bias, bias_off) = op.bias.unwrap_or(op.x);
        let bytes = words(&[
            op.batch,
            op.channels,
            op.kernel,
            usize::from(op.bias.is_some()),
            op.x.1,
            op.weight.1,
            bias_off,
            op.pool.1,
            op.slots.1,
            op.seq,
        ])?;
        let buffers = [op.x.0, op.weight.0, bias, op.pool.0, op.slots.0, op.out];
        let groups = (op.batch * op.channels)
            .div_ceil(INDEX_THREADS)
            .min(MAX_GROUPS) as u32;
        self.context.kernels.ssm_conv_step.dispatch(
            self,
            &buffers.map(|b| b.buffer),
            &bytes,
            [groups.max(1), 1, 1],
        );
        for buffer in buffers {
            self.retain(buffer.clone());
        }
        Ok(())
    }

    pub fn gated_rmsnorm(&self, op: &GatedRmsNorm<'_>) -> Result<()> {
        // groups across x, rows down y and on through z
        let rows_y = op.rows.min(MAX_DISPATCH_GROUPS);
        let grid = [
            op.groups as u32,
            rows_y as u32,
            op.rows.div_ceil(rows_y.max(1)) as u32,
        ];
        if op.rows == 0
            || op.groups == 0
            || op.group_size == 0
            || op.groups > MAX_DISPATCH_GROUPS
            || op.rows.div_ceil(rows_y) > MAX_DISPATCH_GROUPS
        {
            return Err(Error::Invalid(format!(
                "gated RMSNorm of {} rows of {} groups of {}",
                op.rows, op.groups, op.group_size
            )));
        }
        let mut bytes = words(&[op.group_size, op.groups])?;
        bytes.extend_from_slice(&op.eps.to_bits().to_le_bytes());
        bytes.extend(words(&[op.x.1, op.gate.1, op.weight.1])?);
        // gated_rmsnorm*.comp's NORM_FIRST and SIGMOID_GATE bits
        let mode = usize::from(op.norm_first) | (usize::from(op.sigmoid) << 1);
        let cols = op.groups * op.group_size;
        // a decode row's norm writes its int8 blocks too, cached for the matvec that reads it
        if op.rows <= crate::kernels::MATVEC_MAX_ROWS
            && op.group_size.is_multiple_of(crate::kernels::BLOCK_VALUES)
            && ![op.x.0, op.gate.0, op.weight.0]
                .into_iter()
                .any(|b| Arc::ptr_eq(b, op.out))
        {
            let blocks = Arc::new(Buffer::new(
                op.out.context(),
                crate::MatVec::scratch_bytes(op.rows, cols),
            )?);
            let s_off = op.rows * cols / crate::kernels::BLOCK_VALUES
                * crate::kernels::Q8_WORD_BYTES
                / crate::kernels::Q8_SCALE_BYTES;
            bytes.extend(words(&[s_off, mode, op.rows])?);
            let buffers = [op.x.0, op.gate.0, op.weight.0, op.out, &blocks, &blocks];
            self.context.kernels.gated_rmsnorm_q8.dispatch(
                self,
                &buffers.map(|b| b.buffer),
                &bytes,
                grid,
            );
            self.context
                .remember_quantized(op.out, 0, op.rows * cols, &blocks, s_off);
            for buffer in buffers {
                self.retain(buffer.clone());
            }
            return Ok(());
        }
        bytes.extend(words(&[mode, op.rows])?);
        let buffers = [op.x.0, op.gate.0, op.weight.0, op.out];
        self.context
            .kernels
            .gated_rmsnorm
            .dispatch(self, &buffers.map(|b| b.buffer), &bytes, grid);
        for buffer in buffers {
            self.retain(buffer.clone());
        }
        Ok(())
    }

    pub fn rope(&self, op: &Rope<'_>) -> Result<()> {
        if op.rot == 0 || 2 * op.rot > op.head_dim {
            return Err(Error::Invalid(format!(
                "RoPE of {} rotated columns in a head of {}",
                2 * op.rot,
                op.head_dim
            )));
        }
        let rows = op.batch * op.heads * op.seq;
        if rows == 0 {
            return Ok(());
        }
        let mode = match op.rows {
            RopeRows::Positions => 0,
            RopeRows::Tokens => 1,
            RopeRows::Sequence => 2,
        };
        let bytes = words(&[
            op.batch,
            op.heads,
            op.seq,
            op.head_dim,
            op.rot,
            mode,
            usize::from(op.neox),
            op.x.1,
            op.cos.1,
            op.sin.1,
            op.positions.1,
        ])?;
        let buffers = [op.x.0, op.cos.0, op.sin.0, op.positions.0, op.out];
        let items = rows * (op.head_dim - op.rot);
        let groups = items.div_ceil(ROPE_THREADS).clamp(1, MAX_GROUPS_X);
        self.context.kernels.rope.dispatch(
            self,
            &buffers.map(|b| b.buffer),
            &bytes,
            [groups as u32, 1, 1],
        );
        for buffer in buffers {
            self.retain(buffer.clone());
        }
        Ok(())
    }

    pub fn gdn_recurrence(&self, op: &GdnRecurrence<'_>) -> Result<()> {
        if !GdnRecurrence::supports(op.head_k, op.head_v)
            || op.k_heads == 0
            || !op.v_heads.is_multiple_of(op.k_heads)
        {
            return Err(Error::Invalid(format!(
                "a GDN recurrence of {} value heads over {} key heads, widths {} and {}",
                op.v_heads, op.k_heads, op.head_k, op.head_v
            )));
        }
        if op.batch == 0 || op.seq == 0 {
            return Ok(());
        }
        let mut bytes = words(&[
            op.seq,
            op.k_heads,
            op.v_heads,
            op.head_k,
            op.head_v,
            op.conv_dim,
            op.key_dim,
            usize::from(op.tiled),
            op.mixed.1,
            op.b.1,
            op.a.1,
            op.state_in.1,
        ])?;
        bytes.extend_from_slice(&op.eps.to_bits().to_le_bytes());
        bytes.extend(words(&[op.state_out.1])?);
        let buffers = [
            op.mixed.0,
            op.b.0,
            op.a.0,
            op.a_log,
            op.dt_bias,
            op.state_in.0,
            op.state_out.0,
            op.out,
        ];
        self.context.kernels.gdn_recurrence.dispatch(
            self,
            &buffers.map(|b| b.buffer),
            &bytes,
            [op.v_heads as u32, op.batch as u32, 1],
        );
        self.work.set(
            self.work.get().saturating_add(
                (4 * op.batch * op.seq * op.v_heads * op.head_k * op.head_v) as u64,
            ),
        );
        for buffer in buffers {
            self.retain(buffer.clone());
        }
        Ok(())
    }

    pub fn ssm_scan_step(&self, op: &ScanStep<'_>) -> Result<()> {
        if op.groups == 0
            || !op.heads.is_multiple_of(op.groups)
            || op.state != SCAN_LANES * SCAN_VEC
        {
            return Err(Error::Invalid(format!(
                "{} heads over {} groups of {} states",
                op.heads, op.groups, op.state
            )));
        }
        let mut bytes = words(&[op.batch, op.heads, op.head_dim, op.groups, op.state])?;
        for v in op.dt_range {
            bytes.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        bytes.extend(words(&[
            op.pool.1, op.slots.1, op.x.1, op.dt.1, op.b.1, op.c.1, op.seq,
        ])?);
        let buffers = [
            op.x.0, op.dt.0, op.a, op.b.0, op.c.0, op.d, op.dt_bias, op.pool.0, op.slots.0, op.out,
        ];
        let chunked = ScanStep::chunked_scratch_bytes(op.batch, op.seq, op.heads, op.head_dim);
        if let (Some(scratch), Some(bytes_needed)) = (op.scratch, chunked) {
            if scratch.len() < bytes_needed {
                return Err(Error::Invalid(format!(
                    "a chunked scan's {bytes_needed} bytes of scratch in {}",
                    scratch.len()
                )));
            }
            let mut bound = buffers.map(|b| b.buffer).to_vec();
            bound.push(scratch.buffer);
            let tiles = [
                op.seq.div_ceil(SSD_CHUNK) as u32,
                op.heads as u32,
                op.batch as u32,
            ];
            let kernels = &self.context.kernels;
            kernels
                .ssd_chunk_state
                .dispatch(self, &bound, &bytes, tiles);
            let items = (op.batch * op.heads * op.head_dim * op.state).div_ceil(SSD_THREADS);
            let items_x = items.clamp(1, MAX_GROUPS_X);
            kernels.ssd_state_pass.dispatch(
                self,
                &bound,
                &bytes,
                [items_x as u32, items.div_ceil(items_x).max(1) as u32, 1],
            );
            kernels.ssd_chunk_out.dispatch(self, &bound, &bytes, tiles);
            self.retain(scratch.clone());
            for buffer in buffers {
                self.retain(buffer.clone());
            }
            return Ok(());
        }
        let groups = (op.batch * op.heads * op.head_dim).div_ceil(SCAN_ROWS_PER_GROUP);
        let groups_x = groups.clamp(1, MAX_GROUPS_X);
        self.context.kernels.ssm_scan_step.dispatch(
            self,
            &buffers.map(|b| b.buffer),
            &bytes,
            [groups_x as u32, groups.div_ceil(groups_x).max(1) as u32, 1],
        );
        for buffer in buffers {
            self.retain(buffer.clone());
        }
        Ok(())
    }
}

pub(crate) const MOE_MATVEC_BINDINGS: usize = 5;
pub(crate) const MOE_MATVEC_PUSH_BYTES: u32 = 10 * 4;
pub(crate) const MOE_SORT_BINDINGS: usize = 2;
pub(crate) const RMS_Q8_BINDINGS: usize = 7;
// rms_norm_q8's widest row: its 1024 threads hold 4 values a pass, 4 passes
const RMS_Q8_MAX_COLS: usize = 16384;
pub(crate) const GATED_Q8_BINDINGS: usize = 6;
pub(crate) const GATED_Q8_PUSH_BYTES: u32 = 9 * 4;
pub(crate) const RMS_Q8_PUSH_BYTES: u32 = 10 * 4;
pub(crate) const MOE_SORT_PUSH_BYTES: u32 = 5 * 4;
pub(crate) const MOE_MM_BINDINGS: usize = 5;
pub(crate) const MOE_MM_PUSH_BYTES: u32 = 8 * 4;
// past this many routes a step's experts take the tiled matmul over routes grouped by expert (moe_sort's tiles of
// MOE_TILE), its weight rows a tile, and the sort's expert limit
const MOE_TILED_MIN_SLOTS: usize = 64;
const MOE_TILE: usize = 64;
const MOE_TILE_N: usize = 256;
// moe_mm_cm2's weight rows and routes a tile, and the gathered rows' zero padding past the last route (a tile's
// reach, the larger of the matrix-core kernels')
const MOE_CM2_TILE_N: usize = 128;
const MOE_CM2_TILE_M: usize = 128;
const MOE_GATHER_PAD: usize = 128;
// the matrix-core expert kernel's weight rows a workgroup, and the routes an expert averages before it beats
// streaming the rows (its 64-token tiles mostly empty below: 3090, 128 experts, 24 a expert 2.37 against 1.97 ms,
// 96 a expert 5.16 against 5.55 ms)
const MMC_TILE_N: usize = 64;
const MMC_MIN_ROUTES_PER_EXPERT: usize = 48;
const MAX_SORT_EXPERTS: usize = 1024;
pub(crate) const ROUTER_BINDINGS: usize = 4;
pub(crate) const ROUTER_PUSH_BYTES: u32 = 13 * 4;
// the expert matvec's rows a workgroup, and the router shader's limits
const MATVEC_ROWS: usize = 4;
const MAX_ROUTER_EXPERTS: usize = 1024;
const MAX_ROUTER_TOP_K: usize = 32;

/// `y[slot] = W[ids[slot]] x[slot / x_div]` over a stack of Q5_0 experts (each `rows x cols`) in `weights`, each
/// expert read once for every slot routed to it and `x`'s rows quantized to int8 blocks into `scratch` (of
/// [`crate::MatVec::scratch_bytes`] for `slots / x_div` rows); `y` is `(slots, rows)`.
pub struct MoeMatVec<'a> {
    pub ty: crate::BlockType,
    pub weights: &'a Arc<dyn crate::Bound>,
    pub experts: usize,
    pub rows: usize,
    pub cols: usize,
    pub slots: usize,
    /// slots a row of `x` serves: the routes of one token, or 1 for a row a route
    pub x_div: usize,
    pub x: At<'a>,
    pub ids: At<'a>,
    pub y: &'a Arc<Buffer>,
    pub scratch: &'a Arc<Buffer>,
    /// past MOE_TILED_MIN_SLOTS routes, the routes grouped by expert: [`MoeMatVec::routing_bytes`]
    pub routing: &'a Arc<Buffer>,
}

/// How a router turns logits into scores or weights.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(u32)]
pub enum RouterFn {
    Raw = 0,
    Softmax = 1,
    Sigmoid = 2,
}

/// Top-k routing of `tokens` rows of `experts` logits into `values` (f32) and `indices` (u32), each `(tokens, top_k)`.
pub struct Router<'a> {
    pub tokens: usize,
    pub experts: usize,
    pub top_k: usize,
    pub score_fn: RouterFn,
    pub weight_fn: RouterFn,
    pub renormalize: bool,
    pub norm_min: f32,
    pub output_scale: f32,
    pub clip: Option<(f32, f32)>,
    pub logits: At<'a>,
    pub bias: Option<&'a Arc<Buffer>>,
    pub values: &'a Arc<Buffer>,
    pub indices: &'a Arc<Buffer>,
}

impl Recorder<'_> {
    pub fn moe_mat_vec(&self, op: &MoeMatVec<'_>) -> Result<()> {
        let expert_bytes = op.rows * op.cols / op.ty.block_values() * op.ty.block_bytes();
        if !op.cols.is_multiple_of(op.ty.block_values())
            || op.x_div == 0
            || op.weights.bytes() < op.experts * expert_bytes
            || !op.weights.offset().is_multiple_of(2)
            || op.experts > MAX_SORT_EXPERTS
            || op.routing.len() < MoeMatVec::routing_bytes(op.slots, op.experts)
        {
            return Err(Error::Invalid(format!(
                "a {} x {}x{} {:?} expert matvec over {} slots",
                op.experts, op.rows, op.cols, op.ty, op.slots
            )));
        }
        if op.slots == 0 {
            return Ok(());
        }
        if op.slots > MOE_TILED_MIN_SLOTS
            && let Some(coop) = self.context.matmul_coop()
        {
            // NVIDIA's cooperative matrix 2 takes every prompt's experts, the KHR matrix cores only busy ones
            if let Some(cm2) = coop.moe_mm_cm2(op.ty)
                && op.cols.is_multiple_of(crate::kernels::CM2_K_STEP)
            {
                return self.moe_mmc(op, coop, Some(cm2), expert_bytes);
            }
            if op.slots >= op.experts * MMC_MIN_ROUTES_PER_EXPERT {
                return self.moe_mmc(op, coop, None, expert_bytes);
            }
        }
        if op.slots > MOE_TILED_MIN_SLOTS {
            let s_off =
                self.quantize_rows(op.x, op.slots.div_ceil(op.x_div), op.cols, op.scratch)?;
            return self.moe_mm(op, s_off, expert_bytes);
        }
        let (blocks, s_off) =
            self.quantized(op.x, op.slots.div_ceil(op.x_div), op.cols, op.scratch)?;
        if op.slots > MAX_GROUPS_X {
            return Err(Error::Invalid(format!("{} expert routes", op.slots)));
        }
        let groups = op.rows.div_ceil(MATVEC_ROWS);
        let groups_x = groups.min(MAX_GROUPS_X);
        let bytes = words(&[
            op.rows,
            op.cols,
            op.weights.offset() as usize / 2,
            0,
            groups_x,
            op.x_div,
            op.ids.1,
            expert_bytes / 2,
            op.slots,
            s_off,
        ])?;
        let buffers = [
            op.weights.vk_buffer(),
            blocks.buffer,
            op.y.buffer,
            op.ids.0.buffer,
            blocks.buffer,
        ];
        self.profile_shape(
            || {
                format!(
                    "m={} n={} k={} e={}",
                    op.rows, op.slots, op.cols, op.experts
                )
            },
            op.rows * op.cols * op.slots,
        );
        self.context.kernels.moe_mat_vecs(op.ty).dispatch(
            self,
            &buffers,
            &bytes,
            [
                groups_x as u32,
                groups.div_ceil(groups_x) as u32,
                op.slots as u32,
            ],
        );
        self.retain(op.weights.clone() as crate::Retained);
        for buffer in [op.ids.0, op.y] {
            self.retain(buffer.clone());
        }
        Ok(())
    }

    // a prompt's routes on matrix cores: sorted by expert, their activations gathered in that order as f16, then the
    // tiles; on NVIDIA's cooperative matrix 2 a build a tile width (16 to 128 routes) side by side, else the KHR tile
    fn moe_mmc(
        &self,
        op: &MoeMatVec<'_>,
        coop: &crate::kernels::CoopKernels,
        cm2: Option<&[crate::pipeline::Pipeline; 4]>,
        expert_bytes: usize,
    ) -> Result<()> {
        if op.scratch.len() < MoeMatVec::scratch_bytes(self.context, op.slots, op.x_div, op.cols) {
            return Err(Error::Invalid(format!(
                "{} routes' scratch of {} bytes",
                op.slots,
                op.scratch.len()
            )));
        }
        let tiles_off = op.slots;
        let tile = if cm2.is_some() {
            MOE_CM2_TILE_M
        } else {
            crate::kernels::MMC_TILE_M
        };
        self.context.kernels.moe_sort.dispatch(
            self,
            &[op.ids.0.buffer, op.routing.buffer],
            &words(&[op.slots, op.experts, op.ids.1, tiles_off, tile])?,
            [1, 1, 1],
        );
        // a tile's last matrices read rows past its routes: the next tile's, or past the last route these zeros
        let pad_at = (op.slots * op.cols * 2) as u64;
        let pad = (MOE_GATHER_PAD * op.cols * 2) as u64;
        self.fill_range(op.scratch.buffer, pad_at, pad, 0);
        let pairs = (op.slots * op.cols / 2).div_ceil(crate::kernels::CONVERT_THREADS);
        let pairs_x = pairs.min(MAX_GROUPS_X);
        coop.moe_gather_f16.dispatch(
            self,
            &[op.x.0.buffer, op.routing.buffer, op.scratch.buffer],
            &words(&[op.slots, op.cols, op.x_div, op.x.1])?,
            [pairs_x as u32, pairs.div_ceil(pairs_x) as u32, 1],
        );
        self.profile_shape(
            || {
                format!(
                    "m={} n={} k={} e={}",
                    op.rows, op.slots, op.cols, op.experts
                )
            },
            op.rows * op.cols * op.slots,
        );
        let buffers = [
            op.weights.vk_buffer(),
            op.scratch.buffer,
            op.y.buffer,
            op.routing.buffer,
        ];
        if let Some(builds) = cm2 {
            let push = words(&[
                op.rows,
                op.cols,
                op.weights.offset() as usize / 2,
                0,
                0,
                expert_bytes / 2,
                tiles_off,
                op.slots + MOE_GATHER_PAD,
            ])?;
            let groups = [
                op.rows.div_ceil(MOE_CM2_TILE_N) as u32,
                MoeMatVec::max_tiles(op.slots, op.experts) as u32,
                1,
            ];
            // the widths' tiles are disjoint, so the builds may run side by side
            let (last, rest) = builds.split_last().expect("four builds");
            for pipeline in rest {
                pipeline.dispatch_beside_next(self, &buffers, &push, groups);
            }
            last.dispatch(self, &buffers, &push, groups);
        } else {
            coop.moe_mmc(op.ty).dispatch(
                self,
                &buffers,
                &words(&[
                    op.rows,
                    op.cols,
                    op.weights.offset() as usize / 2,
                    0,
                    0,
                    expert_bytes / 2,
                    tiles_off,
                ])?,
                [
                    op.rows.div_ceil(MMC_TILE_N) as u32,
                    MoeMatVec::max_tiles(op.slots, op.experts) as u32,
                    1,
                ],
            );
        }
        self.work.set(
            self.work
                .get()
                .saturating_add((op.slots * op.rows * op.cols) as u64),
        );
        self.retain(op.weights.clone() as crate::Retained);
        for buffer in [op.x.0, op.ids.0, op.y, op.routing, op.scratch] {
            self.retain(buffer.clone());
        }
        Ok(())
    }

    fn moe_mm(&self, op: &MoeMatVec<'_>, s_off: usize, expert_bytes: usize) -> Result<()> {
        let tiles_off = op.slots;
        self.context.kernels.moe_sort.dispatch(
            self,
            &[op.ids.0.buffer, op.routing.buffer],
            &words(&[op.slots, op.experts, op.ids.1, tiles_off, MOE_TILE])?,
            [1, 1, 1],
        );
        let bytes = words(&[
            op.rows,
            op.cols,
            op.weights.offset() as usize / 2,
            0,
            op.x_div,
            expert_bytes / 2,
            s_off,
            tiles_off,
        ])?;
        self.profile_shape(
            || {
                format!(
                    "m={} n={} k={} e={}",
                    op.rows, op.slots, op.cols, op.experts
                )
            },
            op.rows * op.cols * op.slots,
        );
        self.context.kernels.moe_mm(op.ty).dispatch(
            self,
            &[
                op.weights.vk_buffer(),
                op.scratch.buffer,
                op.y.buffer,
                op.scratch.buffer,
                op.routing.buffer,
            ],
            &bytes,
            [
                op.rows.div_ceil(MOE_TILE_N) as u32,
                MoeMatVec::max_tiles(op.slots, op.experts) as u32,
                1,
            ],
        );
        self.work.set(
            self.work
                .get()
                .saturating_add((op.slots * op.rows * op.cols) as u64),
        );
        self.retain(op.weights.clone() as crate::Retained);
        for buffer in [op.ids.0, op.y, op.routing, op.scratch] {
            self.retain(buffer.clone());
        }
        Ok(())
    }

    pub fn moe_router(&self, op: &Router<'_>) -> Result<()> {
        if op.experts > MAX_ROUTER_EXPERTS || op.top_k > MAX_ROUTER_TOP_K || op.top_k > op.experts {
            return Err(Error::Invalid(format!(
                "top {} of {} experts",
                op.top_k, op.experts
            )));
        }
        if op.tokens == 0 {
            return Ok(());
        }
        let (clip_min, clip_max) = op.clip.unwrap_or((0.0, 0.0));
        let mut bytes = words(&[
            op.tokens,
            op.experts,
            op.top_k,
            op.score_fn as usize,
            op.weight_fn as usize,
            usize::from(op.renormalize),
            usize::from(op.bias.is_some()),
            usize::from(op.clip.is_some()),
        ])?;
        for v in [clip_min, clip_max, op.norm_min, op.output_scale] {
            bytes.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        bytes.extend(words(&[op.logits.1])?);
        let bias = op.bias.unwrap_or(op.logits.0);
        let buffers = [op.logits.0, bias, op.values, op.indices];
        self.context.kernels.moe_router.dispatch(
            self,
            &buffers.map(|b| b.buffer),
            &bytes,
            [op.tokens as u32, 1, 1],
        );
        for buffer in buffers {
            self.retain(buffer.clone());
        }
        Ok(())
    }
}

pub(crate) const PAGED_WRITE_BINDINGS: usize = 5;
pub(crate) const PAGED_WRITE_PUSH_BYTES: u32 = 8 * 4;
pub(crate) const PAGED_WRITE_QUANT_PUSH_BYTES: u32 = 9 * 4;
pub(crate) const PAGED_DECODE_BINDINGS: usize = 6;
pub(crate) const PAGED_DECODE_PUSH_BYTES: u32 = 12 * 4;
pub(crate) const GATED_RMSNORM_BINDINGS: usize = 4;
pub(crate) const GATED_RMSNORM_PUSH_BYTES: u32 = 8 * 4;
pub(crate) const PAGED_PREFILL_BINDINGS: usize = 5;
pub(crate) const PAGED_PREFILL_PUSH_BYTES: u32 = 13 * 4;
pub(crate) const HADAMARD_BINDINGS: usize = 2;
pub(crate) const HADAMARD_PUSH_BYTES: u32 = 3 * 4;
// the quantizing write's and the Hadamard op's threads a workgroup; the values a rotation (and a scale) covers
const QUANT_THREADS: usize = 64;
const QUANT_GROUP: usize = 32;
// a quantized row's head size is a multiple of this, so a row is whole words with whole pairs of f16 scales
const QUANT_HEAD_MULTIPLE: usize = 64;
const QUANT_BITS: std::ops::RangeInclusive<u32> = 2..=8;
const MAX_DISPATCH_GROUPS: usize = 65535;
// the prefill shader's rows a workgroup and its tile; a split covers at least PREFILL_MIN_SPAN positions, at most
// MAX_PREFILL_SPLITS of them, their partials within PREFILL_PARTIALS_BYTES (a prompt chunk's rows already fill the GPU,
// and every layer's partials stay live until its batch completes)
const PREFILL_ROWS: usize = 8;
// threads a (row, head) in paged_prefill_main.glsl's narrow and wide builds
const PREFILL_LANES: usize = 2;
const PREFILL_WIDE_LANES: usize = 4;
// paged_prefill_cm's query rows (positions x heads) a workgroup, and the heads it tiles
const PREFILL_CM_ROWS: usize = 64;
const PREFILL_CM_HEAD_STEP: usize = 16;
const PREFILL_CM_MAX_HEAD: usize = 128;
const PREFILL_MIN_SPAN: usize = 1024;
const MAX_PREFILL_SPLITS: usize = 32;
const PREFILL_PARTIALS_BYTES: usize = 32 << 20;
pub(crate) const PAGED_REDUCE_BINDINGS: usize = 2;
pub(crate) const PAGED_REDUCE_PUSH_BYTES: u32 = 3 * 4;
// the decode shader's geometry: the positions a workgroup takes, a tile of whole blocks, its threads, the query heads
// a kv head serves and those a value thread sums, and the cache's vec4 rows (x)
const PAGED_CHUNK: usize = 64;
// the shared tiles' head size, sized for occupancy; decode and prefill have wide builds to 256 (Qwen3.5's)
pub(crate) const MAX_PAGED_NARROW_HEAD: usize = 128;
const MAX_PAGED_WIDE_HEAD: usize = 256;
// the partials one piece of a call holds; many queries (an MTP head reading a whole prompt) run in pieces
const PAGED_PARTIALS_BYTES: usize = 256 << 20;
const PAGED_TILE: usize = 32;
const PAGED_THREADS: usize = 256;
const MAX_PAGED_GROUP: usize = 16;
const PAGED_VALUE_HEADS: usize = 8;
const PAGED_X: usize = 4;
// an f16 cache's key rows hold 8 dimensions a position (16 bytes, as vLLM's layout has it)
const PAGED_X_F16: usize = 8;

/// The geometry of a pair of caches: vLLM's layout, keys `[blocks, kv_heads, head_size / x, block_size, x]` and
/// values `[blocks, kv_heads, head_size, block_size]`, f32 or (`f16`) f16; or with `quant` (K and V bits) rows of
/// bytes `[blocks, kv_heads, block_size, row]` in inference-paged-attn's kv_quant format, `x` unused.
#[derive(Clone, Copy, Debug)]
pub struct CacheShape {
    pub kv_heads: usize,
    pub head_size: usize,
    pub block_size: usize,
    pub x: usize,
    pub f16: bool,
    pub quant: Option<(u32, u32)>,
}

impl CacheShape {
    /// The bits a quantized row of `row_bytes` holds a value of `head_size`, if it is one.
    pub fn quant_bits(row_bytes: usize, head_size: usize) -> Option<u32> {
        if head_size == 0 || !head_size.is_multiple_of(QUANT_HEAD_MULTIPLE) {
            return None;
        }
        let data = row_bytes.checked_sub(head_size / 16)?;
        let bits = u32::try_from(data * 8 / head_size).ok()?;
        (QUANT_BITS.contains(&bits) && data * 8 == head_size * bits as usize).then_some(bits)
    }

    // the vec4 row layout the float kernels read, or a quantized cache's head size
    fn readable(&self) -> bool {
        match self.quant {
            Some((k, v)) => {
                self.head_size.is_multiple_of(QUANT_HEAD_MULTIPLE)
                    && QUANT_BITS.contains(&k)
                    && QUANT_BITS.contains(&v)
            }
            None => self.x == if self.f16 { PAGED_X_F16 } else { PAGED_X },
        }
    }

    fn bits(&self) -> [usize; 2] {
        self.quant.map_or([0, 0], |(k, v)| [k as usize, v as usize])
    }
}

/// Writes `tokens` key and value rows `(tokens, kv_heads, head_size)` into the caches at the i64 `slots`.
pub struct PagedWrite<'a> {
    pub shape: CacheShape,
    pub tokens: usize,
    pub key: At<'a>,
    pub value: At<'a>,
    pub key_cache: &'a Arc<Buffer>,
    pub value_cache: &'a Arc<Buffer>,
    pub slots: At<'a>,
}

/// One sequence's `rows` prompt queries `(rows, heads, head_size)` over its `context` cached positions (the rows'
/// own included, last), row r causal over the first `context - rows + r + 1`, through its block table row (u32);
/// `out` is laid out as the queries, and `partials` holds [`PagedPrefill::partials_bytes`].
pub struct PagedPrefill<'a> {
    pub shape: CacheShape,
    pub rows: usize,
    pub heads: usize,
    pub context: usize,
    pub scale: f32,
    pub q: At<'a>,
    pub key_cache: &'a Arc<Buffer>,
    pub value_cache: &'a Arc<Buffer>,
    pub block_table: At<'a>,
    pub partials: &'a Arc<Buffer>,
    pub out: At<'a>,
}

impl PagedPrefill<'_> {
    // the context's splits and the positions each covers (whole tiles)
    fn splits(rows: usize, heads: usize, head_size: usize, context: usize) -> (usize, usize) {
        let per_split = (rows * heads * (head_size + 2) * 4).max(1);
        let splits = context
            .div_ceil(PREFILL_MIN_SPAN)
            .clamp(1, MAX_PREFILL_SPLITS)
            .min((PREFILL_PARTIALS_BYTES / per_split).max(1));
        let span = context.div_ceil(splits).next_multiple_of(PAGED_TILE);
        (context.div_ceil(span).max(1), span)
    }

    pub fn partials_bytes(rows: usize, heads: usize, head_size: usize, context: usize) -> usize {
        let (splits, _) = Self::splits(rows, heads, head_size, context);
        rows * heads * splits * (head_size + 2) * 4
    }
}

/// One query `(seqs, heads, head_size)` a sequence over its `context_lens` cached positions through `block_tables`
/// (`(seqs, max_blocks)` u32); `out` is laid out as the queries, and `partials` holds
/// [`PagedDecode::partials_bytes`] of each context chunk's sums.
pub struct PagedDecode<'a> {
    pub shape: CacheShape,
    pub seqs: usize,
    pub heads: usize,
    pub max_blocks: usize,
    pub max_context: usize,
    pub scale: f32,
    pub q: At<'a>,
    pub key_cache: &'a Arc<Buffer>,
    pub value_cache: &'a Arc<Buffer>,
    pub block_tables: At<'a>,
    pub context_lens: At<'a>,
    pub partials: &'a Arc<Buffer>,
    pub out: &'a Arc<Buffer>,
}

impl PagedDecode<'_> {
    pub fn partials_bytes(
        seqs: usize,
        heads: usize,
        head_size: usize,
        max_context: usize,
    ) -> usize {
        seqs.min(Self::piece_seqs(heads, head_size, max_context))
            * Self::row_bytes(heads, head_size, max_context)
    }

    // a sequence's partials, and the sequences one piece of the call takes within PAGED_PARTIALS_BYTES
    fn row_bytes(heads: usize, head_size: usize, max_context: usize) -> usize {
        heads * max_context.div_ceil(PAGED_CHUNK).max(1) * (head_size + 2) * 4
    }

    fn piece_seqs(heads: usize, head_size: usize, max_context: usize) -> usize {
        (PAGED_PARTIALS_BYTES / Self::row_bytes(heads, head_size, max_context)).max(1)
    }
}

impl Recorder<'_> {
    pub fn paged_cache_write(&self, op: &PagedWrite<'_>) -> Result<()> {
        let s = op.shape;
        if op.tokens == 0 {
            return Ok(());
        }
        if let Some((k_bits, v_bits)) = s.quant {
            return self.paged_cache_write_quant(op, k_bits, v_bits);
        }
        let bytes = words(&[
            op.tokens,
            s.kv_heads,
            s.head_size,
            s.block_size,
            s.x,
            op.key.1,
            op.value.1,
            op.slots.1 * 2,
        ])?;
        let buffers = [
            op.key.0,
            op.value.0,
            op.key_cache,
            op.value_cache,
            op.slots.0,
        ];
        let groups = (op.tokens * s.kv_heads * s.head_size)
            .div_ceil(INDEX_THREADS)
            .min(MAX_GROUPS) as u32;
        self.context.kernels.paged_cache_write(op.shape).dispatch(
            self,
            &buffers.map(|b| b.buffer),
            &bytes,
            [groups, 1, 1],
        );
        for buffer in buffers {
            self.retain(buffer.clone());
        }
        Ok(())
    }

    fn paged_cache_write_quant(&self, op: &PagedWrite<'_>, k_bits: u32, v_bits: u32) -> Result<()> {
        let s = op.shape;
        let threads = op.tokens * s.kv_heads * 2;
        if !s.readable() || threads.div_ceil(QUANT_THREADS) > MAX_DISPATCH_GROUPS {
            return Err(Error::Invalid(format!(
                "quantized paged write of {} tokens, {} kv heads, head size {}",
                op.tokens, s.kv_heads, s.head_size
            )));
        }
        let bytes = words(&[
            op.tokens,
            s.kv_heads,
            s.head_size,
            s.block_size,
            k_bits as usize,
            op.key.1,
            op.value.1,
            op.slots.1 * 2,
            v_bits as usize,
        ])?;
        let buffers = [
            op.key.0,
            op.value.0,
            op.key_cache,
            op.value_cache,
            op.slots.0,
        ];
        self.context.kernels.paged_cache_write_quant.dispatch(
            self,
            &buffers.map(|b| b.buffer),
            &bytes,
            [threads.div_ceil(QUANT_THREADS) as u32, 1, 1],
        );
        for buffer in buffers {
            self.retain(buffer.clone());
        }
        Ok(())
    }

    /// `out` = `x` with each of `values / 32` consecutive groups through the normalized 32-wide Hadamard transform
    /// (its own inverse: a query into a quantized cache's space, attention's output back out); in place when both
    /// name the same place.
    pub fn hadamard_32(&self, x: At<'_>, out: At<'_>, values: usize) -> Result<()> {
        if !values.is_multiple_of(QUANT_GROUP) {
            return Err(Error::Invalid(format!("hadamard of {values} values")));
        }
        let groups = values / QUANT_GROUP;
        if groups == 0 {
            return Ok(());
        }
        let wide = groups.div_ceil(QUANT_THREADS);
        let x_groups = wide.min(MAX_DISPATCH_GROUPS);
        self.context.kernels.hadamard_32.dispatch(
            self,
            &[x.0.buffer, out.0.buffer],
            &words(&[groups, x.1, out.1])?,
            [x_groups as u32, wide.div_ceil(x_groups) as u32, 1],
        );
        self.retain(x.0.clone());
        self.retain(out.0.clone());
        Ok(())
    }

    pub fn paged_prefill(&self, op: &PagedPrefill<'_>) -> Result<()> {
        let s = op.shape;
        let group = op.heads / s.kv_heads.max(1);
        // the wide build holds a (row, head) in four threads, half the narrow one's rows a workgroup
        let (lanes, rows_a_group) = if s.head_size > MAX_PAGED_NARROW_HEAD {
            (PREFILL_WIDE_LANES, PREFILL_ROWS / 2)
        } else {
            (PREFILL_LANES, PREFILL_ROWS)
        };
        if s.head_size == 0
            || s.head_size > MAX_PAGED_WIDE_HEAD
            || !s.head_size.is_multiple_of(lanes * PAGED_X)
            || !s.readable()
            || s.kv_heads == 0
            || !op.heads.is_multiple_of(s.kv_heads)
            || group > MAX_PAGED_GROUP
            || !PAGED_TILE.is_multiple_of(s.block_size)
            || !s.block_size.is_multiple_of(PAGED_X)
            || !op.q.1.is_multiple_of(PAGED_X)
            || op.rows > op.context
            || op.partials.len()
                < PagedPrefill::partials_bytes(op.rows, op.heads, s.head_size, op.context)
        {
            return Err(Error::Invalid(format!(
                "paged prefill of {} rows, {} heads over {} kv heads, head size {}, context {}",
                op.rows, op.heads, s.kv_heads, s.head_size, op.context
            )));
        }
        if op.rows == 0 {
            return Ok(());
        }
        let (splits, span) = PagedPrefill::splits(op.rows, op.heads, s.head_size, op.context);
        let mut bytes = words(&[op.heads, s.kv_heads, s.head_size, s.block_size])?;
        bytes.extend_from_slice(&op.scale.to_bits().to_le_bytes());
        bytes.extend(words(&[
            op.q.1 / PAGED_X,
            op.block_table.1,
            op.context,
            op.rows,
            splits,
            span,
        ])?);
        bytes.extend(words(&s.bits())?);
        let buffers = [
            op.q.0,
            op.key_cache,
            op.value_cache,
            op.block_table.0,
            op.partials,
        ];
        let bound = buffers.map(|b| b.buffer);
        match self.prefill_cm(s, group) {
            Some(pipeline) => pipeline.dispatch(
                self,
                &bound,
                &bytes,
                [
                    splits as u32,
                    s.kv_heads as u32,
                    op.rows.div_ceil(PREFILL_CM_ROWS / group) as u32,
                ],
            ),
            None => self.context.kernels.paged_prefill(s).dispatch(
                self,
                &bound,
                &bytes,
                [
                    splits as u32,
                    s.kv_heads as u32,
                    op.rows.div_ceil(rows_a_group) as u32,
                ],
            ),
        }
        self.context.kernels.paged_decode_reduce.dispatch(
            self,
            &[op.partials.buffer, op.out.0.buffer],
            &words(&[s.head_size, splits, op.out.1])?,
            [(op.rows * op.heads) as u32, 1, 1],
        );
        self.work.set(
            self.work
                .get()
                .saturating_add((op.rows * op.context * op.heads * s.head_size) as u64),
        );
        for buffer in buffers.iter().copied().chain([op.out.0]) {
            self.retain(buffer.clone());
        }
        Ok(())
    }

    // the matrix-core prompt attention, for heads it tiles (multiples of 16 to 128) and groups within its rows
    fn prefill_cm(&self, s: CacheShape, group: usize) -> Option<&crate::pipeline::Pipeline> {
        if !s.head_size.is_multiple_of(PREFILL_CM_HEAD_STEP)
            || s.head_size > PREFILL_CM_MAX_HEAD
            || group > PREFILL_CM_ROWS
        {
            return None;
        }
        Some(self.context.coop.as_ref()?.paged_prefill(s))
    }

    pub fn paged_decode(&self, op: &PagedDecode<'_>) -> Result<()> {
        let s = op.shape;
        let group = op.heads / s.kv_heads.max(1);
        let chunks = op.max_context.div_ceil(PAGED_CHUNK).max(1);
        if s.head_size == 0
            || s.head_size > MAX_PAGED_WIDE_HEAD
            || !PAGED_THREADS.is_multiple_of(s.head_size)
            || !s.readable()
            || s.kv_heads == 0
            || !op.heads.is_multiple_of(s.kv_heads)
            || group > MAX_PAGED_GROUP
            || group > PAGED_VALUE_HEADS * (PAGED_THREADS / s.head_size)
            || !PAGED_TILE.is_multiple_of(s.block_size)
            || !s.block_size.is_multiple_of(PAGED_X)
            || !op.q.1.is_multiple_of(PAGED_X)
            || op.partials.len()
                < PagedDecode::partials_bytes(op.seqs, op.heads, s.head_size, op.max_context)
        {
            return Err(Error::Invalid(format!(
                "paged decode of {} heads over {} kv heads, head size {}, context {}",
                op.heads, s.kv_heads, s.head_size, op.max_context
            )));
        }
        if op.seqs == 0 {
            return Ok(());
        }
        let piece = PagedDecode::piece_seqs(op.heads, s.head_size, op.max_context);
        let buffers = [
            op.q.0,
            op.key_cache,
            op.value_cache,
            op.block_tables.0,
            op.context_lens.0,
            op.partials,
        ];
        for first in (0..op.seqs).step_by(piece) {
            let seqs = piece.min(op.seqs - first);
            let mut bytes = words(&[
                op.heads,
                s.kv_heads,
                s.head_size,
                s.block_size,
                op.max_blocks,
            ])?;
            bytes.extend_from_slice(&op.scale.to_bits().to_le_bytes());
            bytes.extend(words(&[
                (op.q.1 + first * op.heads * s.head_size) / PAGED_X,
                op.block_tables.1 + first * op.max_blocks,
                op.context_lens.1 + first,
                chunks,
            ])?);
            bytes.extend(words(&s.bits())?);
            self.context.kernels.paged_decode(s).dispatch(
                self,
                &buffers.map(|b| b.buffer),
                &bytes,
                [chunks as u32, s.kv_heads as u32, seqs as u32],
            );
            self.context.kernels.paged_decode_reduce.dispatch(
                self,
                &[op.partials.buffer, op.out.buffer],
                &words(&[s.head_size, chunks, first * op.heads * s.head_size])?,
                [(seqs * op.heads) as u32, 1, 1],
            );
        }
        for buffer in buffers.iter().copied().chain([op.out]) {
            self.retain(buffer.clone());
        }
        Ok(())
    }
}

pub(crate) const FLASH_BINDINGS: usize = 5;
pub(crate) const FLASH_PUSH_BYTES: u32 = 16 * 4;
// flash_attn.comp's rows a workgroup, widest head and vec4 width
const FLASH_ROWS: usize = 128;
const FLASH_MAX_HEAD: usize = 128;
const FLASH_VEC: usize = 4;

/// Fused prompt attention over contiguous f32 `q` `(batch, heads, positions, head_size)` and `k`, `v`
/// `(batch, kv_heads, keys, head_size)` into `out` laid out as `q`, for positions `first..first + count` (a call's
/// piece: [`FlashAttn::piece_positions`] bounds one submit's work). `mask` is an additive `(positions, keys)` matrix
/// every batch and head shares; `causal` skips keys past each position's diagonal, which the mask must hide too.
pub struct FlashAttn<'a> {
    pub batch: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub positions: usize,
    pub keys: usize,
    pub head_size: usize,
    pub scale: f32,
    pub causal: bool,
    pub first: usize,
    pub count: usize,
    pub q: At<'a>,
    pub k: At<'a>,
    pub v: At<'a>,
    pub mask: Option<At<'a>>,
    pub out: &'a Arc<Buffer>,
}

impl FlashAttn<'_> {
    /// Whether the kernel takes this shape.
    pub fn supports(heads: usize, kv_heads: usize, head_size: usize) -> bool {
        kv_heads > 0
            && heads.is_multiple_of(kv_heads)
            && head_size > 0
            && head_size <= FLASH_MAX_HEAD
            && head_size.is_multiple_of(FLASH_VEC)
    }

    /// Positions a piece of a call takes so its multiply-adds stay within one submit's work.
    pub fn piece_positions(heads: usize, keys: usize, head_size: usize) -> usize {
        let per_position = (2 * heads * keys.max(1) * head_size) as u64;
        let positions = (crate::stream::WORK_PER_SUBMIT / per_position) as usize;
        positions.max(FLASH_ROWS).next_multiple_of(FLASH_ROWS)
    }

    // the query heads a workgroup's rows take (a divisor of the group), and its positions
    fn rows(&self) -> (usize, usize) {
        let group = self.heads / self.kv_heads;
        let row_heads = (1..=group.min(FLASH_ROWS))
            .rev()
            .find(|h| group.is_multiple_of(*h))
            .unwrap_or(1);
        (row_heads, FLASH_ROWS / row_heads)
    }
}

impl Recorder<'_> {
    pub fn flash_attn(&self, op: &FlashAttn<'_>) -> Result<()> {
        let vec4 = |at: usize| at.is_multiple_of(FLASH_VEC);
        if !FlashAttn::supports(op.heads, op.kv_heads, op.head_size)
            || (op.causal && op.keys < op.positions)
            || op.first + op.count > op.positions
            || ![op.q.1, op.k.1, op.v.1].into_iter().all(vec4)
        {
            return Err(Error::Invalid(format!(
                "fused attention of {} heads over {} kv heads, head size {}, {} positions over {} keys",
                op.heads, op.kv_heads, op.head_size, op.positions, op.keys
            )));
        }
        if op.count == 0 || op.batch == 0 {
            return Ok(());
        }
        let (row_heads, row_positions) = op.rows();
        let chunks = op.heads / op.kv_heads / row_heads;
        let mut bytes = words(&[
            op.positions,
            op.keys,
            op.heads,
            op.kv_heads,
            op.head_size / FLASH_VEC,
        ])?;
        bytes.extend_from_slice(&op.scale.to_bits().to_le_bytes());
        bytes.extend(words(&[
            op.q.1 / FLASH_VEC,
            op.k.1 / FLASH_VEC,
            op.v.1 / FLASH_VEC,
            op.mask.map_or(0, |m| m.1),
            0,
            usize::from(op.mask.is_some()),
            usize::from(op.causal),
            row_heads,
            row_positions,
            op.first,
        ])?);
        // without a mask the queries stand in for its binding, never read
        let mask = op.mask.map_or(op.q.0, |m| m.0);
        let buffers = [op.q.0, op.k.0, op.v.0, mask, op.out];
        if op.positions == 1 {
            // one query a head: a workgroup a head, its subgroups splitting the keys
            self.context.kernels.flash_decode.dispatch(
                self,
                &buffers.map(|b| b.buffer),
                &bytes,
                [op.heads as u32, 1, op.batch as u32],
            );
        } else if let Some(pipeline) = self.flash_attn_cm(op) {
            let group = op.heads / op.kv_heads;
            pipeline.dispatch(
                self,
                &buffers.map(|b| b.buffer),
                &bytes,
                [
                    op.count.div_ceil(PREFILL_CM_ROWS / group) as u32,
                    op.kv_heads as u32,
                    op.batch as u32,
                ],
            );
        } else {
            self.context.kernels.flash_attn.dispatch(
                self,
                &buffers.map(|b| b.buffer),
                &bytes,
                [
                    op.count.div_ceil(row_positions) as u32,
                    (op.kv_heads * chunks) as u32,
                    op.batch as u32,
                ],
            );
        }
        let keys = if op.causal { op.keys / 2 + 1 } else { op.keys };
        self.work.set(
            self.work
                .get()
                .saturating_add((2 * op.batch * op.heads * op.count * keys * op.head_size) as u64),
        );
        for buffer in buffers {
            self.retain(buffer.clone());
        }
        Ok(())
    }
}

impl Recorder<'_> {
    // the matrix-core prompt attention, for the heads and groups paged_prefill_cm tiles
    fn flash_attn_cm(&self, op: &FlashAttn<'_>) -> Option<&crate::pipeline::Pipeline> {
        let group = op.heads / op.kv_heads;
        if !op.head_size.is_multiple_of(PREFILL_CM_HEAD_STEP)
            || op.head_size > PREFILL_CM_MAX_HEAD
            || group > PREFILL_CM_ROWS
        {
            return None;
        }
        Some(&self.context.coop.as_ref()?.flash_attn)
    }
}

pub(crate) const RANKED_DRAW_BINDINGS: usize = 5;
pub(crate) const RANKED_DRAW_PUSH_BYTES: u32 = 3 * 4;

/// A token drawn from each of `rows` packed [`RankedTopK`] rows under its sampler's filters, with the support it was
/// drawn from: `params` holds [`RankedDraw::PARAM_WORDS`] words a row (inverse temperature, top-k or 0, top-p and
/// min-p as f32 bits, a uniform in `[0, 1)`); `ids` and `probs` take [`RankedTopK::K`] entries a row (u32, f32),
/// `tokens` one u32 a row.
pub struct RankedDraw<'a> {
    pub rows: usize,
    pub packed: At<'a>,
    pub params: At<'a>,
    pub ids: &'a Arc<Buffer>,
    pub probs: &'a Arc<Buffer>,
    pub tokens: &'a Arc<Buffer>,
}

impl RankedDraw<'_> {
    pub const PARAM_WORDS: usize = 5;
}

impl Recorder<'_> {
    pub fn ranked_draw(&self, op: &RankedDraw<'_>) -> Result<()> {
        let k = RankedTopK::K;
        if op.ids.len() < op.rows * k * 4
            || op.probs.len() < op.rows * k * 4
            || op.tokens.len() < op.rows * 4
            || op.rows > MAX_DISPATCH_GROUPS
        {
            return Err(Error::Invalid(format!("ranked draw of {} rows", op.rows)));
        }
        if op.rows == 0 {
            return Ok(());
        }
        let buffers = [op.packed.0, op.params.0, op.ids, op.probs, op.tokens];
        self.context.kernels.ranked_draw.dispatch(
            self,
            &buffers.map(|b| b.buffer),
            &words(&[op.rows, op.packed.1, op.params.1])?,
            [op.rows as u32, 1, 1],
        );
        for buffer in buffers {
            self.retain(buffer.clone());
        }
        Ok(())
    }
}

pub(crate) const RELPOS_BINDINGS: usize = 6;
pub(crate) const RELPOS_PUSH_BYTES: u32 = 10 * 4;
// flash_attn_relpos_cm's padding rows before its tables' first head (FRONT) and after their last (a span runs up to
// 33 past, a key tile 31)
const RELPOS_FRONT: usize = 64;
const RELPOS_BACK: usize = 64;
// its query rows a workgroup
const RELPOS_ROWS: usize = 64;
// f32s a convert_f16 thread takes
const CONVERT_VALUES: usize = 2;

/// Attention with a relative-position term (Transformer-XL's, as the FastConformer's), a query head a key head:
/// score(i, j) = qc_i . k_j + qp_i . p[pos_base - i + j], both query sets scaled beforehand, softmax over the keys.
/// `qc`, `qp`, `out` are contiguous f32 `(batch, heads, positions, head_size)`; `k`, `v` and `p` the f16 tables
/// [`Recorder::relpos_table`] wrote from `(batch, heads, keys, head_size)` keys and values and `(batch, heads, p_rows,
/// head_size)` positions; the call takes positions `first..first + count`.
pub struct RelPosAttn<'a> {
    pub batch: usize,
    pub heads: usize,
    pub positions: usize,
    pub keys: usize,
    pub head_size: usize,
    pub p_rows: usize,
    pub pos_base: usize,
    pub first: usize,
    pub count: usize,
    pub qc: At<'a>,
    pub qp: At<'a>,
    pub k: &'a Arc<Buffer>,
    pub v: &'a Arc<Buffer>,
    pub p: &'a Arc<Buffer>,
    pub out: &'a Arc<Buffer>,
}

impl RelPosAttn<'_> {
    /// Whether the kernel takes this head size.
    pub fn supports(head_size: usize) -> bool {
        head_size > 0
            && head_size.is_multiple_of(PREFILL_CM_HEAD_STEP)
            && head_size <= PREFILL_CM_MAX_HEAD
    }

    /// An f16 table's bytes over `rows` a head, padded before its first head and after its last.
    pub fn table_bytes(batch: usize, heads: usize, rows: usize, head_size: usize) -> usize {
        (RELPOS_FRONT + batch * heads * rows + RELPOS_BACK) * head_size * 2
    }

    /// Positions a piece of a call takes so its multiply-adds stay within one submit's work: the content, position
    /// (a tile's span, ~1.5 its keys) and value products.
    pub fn piece_positions(heads: usize, keys: usize, head_size: usize) -> usize {
        let per_position = (7 * heads * keys.max(1) * head_size) as u64;
        let positions = (crate::stream::WORK_PER_SUBMIT / per_position) as usize;
        positions.max(RELPOS_ROWS).next_multiple_of(RELPOS_ROWS)
    }
}

impl Recorder<'_> {
    /// `x` (`count` f32s of `(batch, heads, rows, head_size)`) as one of [`RelPosAttn`]'s f16 tables into `table`
    /// ([`RelPosAttn::table_bytes`]), its padding zeros.
    pub fn relpos_table(
        &self,
        x: At<'_>,
        count: usize,
        head_size: usize,
        table: &Arc<Buffer>,
    ) -> Result<()> {
        let coop = self.context.coop.as_ref().ok_or_else(|| {
            Error::Invalid("relative-position attention runs on matrix cores".into())
        })?;
        let bytes = (count + (RELPOS_FRONT + RELPOS_BACK) * head_size) * 2;
        self.fill(table.buffer, bytes.next_multiple_of(4) as u64, 0);
        let pairs = count.div_ceil(CONVERT_VALUES);
        let groups = pairs.div_ceil(crate::kernels::CONVERT_THREADS);
        let groups_x = groups.min(MAX_GROUPS_X);
        coop.convert_f16.dispatch(
            self,
            &[x.0.buffer, table.buffer],
            &words(&[count, x.1, RELPOS_FRONT * head_size / CONVERT_VALUES])?,
            [groups_x as u32, groups.div_ceil(groups_x) as u32, 1],
        );
        self.retain(x.0.clone());
        self.retain(table.clone());
        Ok(())
    }

    pub fn relpos_attn(&self, op: &RelPosAttn<'_>) -> Result<()> {
        let vec4 = |at: usize| at.is_multiple_of(FLASH_VEC);
        let coop = self.context.coop.as_ref();
        if !RelPosAttn::supports(op.head_size)
            || op.first + op.count > op.positions
            || op.pos_base + 1 < op.positions
            || op.pos_base + op.keys > op.p_rows
            || ![op.qc.1, op.qp.1].into_iter().all(vec4)
            || coop.is_none()
        {
            return Err(Error::Invalid(format!(
                "relative-position attention of {} heads, head size {}, {} positions over {} keys and {} position rows",
                op.heads, op.head_size, op.positions, op.keys, op.p_rows
            )));
        }
        if op.count == 0 || op.batch == 0 {
            return Ok(());
        }
        let bytes = words(&[
            op.positions,
            op.keys,
            op.heads,
            op.head_size / FLASH_VEC,
            op.p_rows,
            op.pos_base,
            op.qc.1 / FLASH_VEC,
            op.qp.1 / FLASH_VEC,
            0,
            op.first,
        ])?;
        let buffers = [op.qc.0, op.qp.0, op.k, op.v, op.p, op.out];
        coop.expect("checked above").flash_attn_relpos.dispatch(
            self,
            &buffers.map(|b| b.buffer),
            &bytes,
            [
                op.count.div_ceil(RELPOS_ROWS) as u32,
                op.heads as u32,
                op.batch as u32,
            ],
        );
        self.work.set(
            self.work.get().saturating_add(
                (7 * op.batch * op.heads * op.count * op.keys * op.head_size) as u64,
            ),
        );
        for buffer in buffers {
            self.retain(buffer.clone());
        }
        Ok(())
    }
}

pub(crate) const RANKED_BINDINGS: usize = 2;
pub(crate) const RANKED_PUSH_BYTES: u32 = 2 * 4;
pub(crate) const RANKED_PART_PUSH_BYTES: u32 = 4 * 4;
// logits a ranked_topk_part workgroup takes (the last of a row takes the remainder)
const RANKED_CHUNK: usize = 4096;

/// Each of `rows` f32 logits rows of `vocab` (from `logits`): its [`RankedTopK::K`] largest values sorted
/// descending (ties to the lower index), their indices as floats, the softmax normalizer at `inv_temperature` and
/// the scaled max, [`RankedTopK::row_floats`] a row into `out`.
pub struct RankedTopK<'a> {
    pub rows: usize,
    pub vocab: usize,
    pub inv_temperature: f32,
    pub logits: At<'a>,
    /// each row's chunks' candidates: [`RankedTopK::scratch_bytes`]
    pub scratch: &'a Arc<Buffer>,
    pub out: &'a Arc<Buffer>,
}

impl RankedTopK<'_> {
    pub const K: usize = 128;

    pub fn row_floats() -> usize {
        2 * Self::K + 2
    }

    fn chunks(vocab: usize) -> usize {
        (vocab / RANKED_CHUNK).max(1)
    }

    pub fn scratch_bytes(rows: usize, vocab: usize) -> usize {
        rows * Self::chunks(vocab) * Self::row_floats() * 4
    }
}

impl Recorder<'_> {
    pub fn ranked_top_k(&self, op: &RankedTopK<'_>) -> Result<()> {
        let temperature_ok = op.inv_temperature.is_finite() && op.inv_temperature > 0.0;
        if op.vocab < RankedTopK::K
            || !temperature_ok
            || op.out.len() < op.rows * RankedTopK::row_floats() * 4
            || op.scratch.len() < RankedTopK::scratch_bytes(op.rows, op.vocab)
            || op.rows > MAX_DISPATCH_GROUPS
        {
            return Err(Error::Invalid(format!(
                "ranked top-{} of {} rows of {} at inverse temperature {}",
                RankedTopK::K,
                op.rows,
                op.vocab,
                op.inv_temperature
            )));
        }
        if op.rows == 0 {
            return Ok(());
        }
        // each chunk's candidates over many workgroups, then a workgroup a row picks from them
        let chunks = RankedTopK::chunks(op.vocab);
        let mut bytes = words(&[op.vocab, op.logits.1])?;
        bytes.extend_from_slice(&op.inv_temperature.to_bits().to_le_bytes());
        bytes.extend(words(&[chunks])?);
        self.context.kernels.ranked_topk_part.dispatch(
            self,
            &[op.logits.0.buffer, op.scratch.buffer],
            &bytes,
            [chunks as u32, op.rows as u32, 1],
        );
        let mut bytes = words(&[chunks])?;
        bytes.extend_from_slice(&op.inv_temperature.to_bits().to_le_bytes());
        self.context.kernels.ranked_topk.dispatch(
            self,
            &[op.scratch.buffer, op.out.buffer],
            &bytes,
            [op.rows as u32, 1, 1],
        );
        for buffer in [op.logits.0, op.scratch, op.out] {
            self.retain(buffer.clone());
        }
        Ok(())
    }
}

pub(crate) const PENALIZE_BINDINGS: usize = 2;
pub(crate) const PENALIZE_PUSH_BYTES: u32 = 6 * 4;
const PENALIZE_THREADS: usize = 64;

/// The sampler's frequency, presence and repetition penalties and logit bias on rows of `vocab` f32 `logits`, in
/// place, through `count` u32 4-word entries (`row << 1 | seen`, token, generated count, bias bits), one a distinct
/// token of a row.
pub struct Penalize<'a> {
    pub vocab: usize,
    pub count: usize,
    pub logits: At<'a>,
    pub entries: At<'a>,
    pub frequency: f32,
    pub presence: f32,
    pub repetition: f32,
}

impl Recorder<'_> {
    pub fn penalize(&self, op: &Penalize<'_>) -> Result<()> {
        if op.entries.0.len() < (op.entries.1 + 4 * op.count) * 4 || !op.entries.1.is_multiple_of(4)
        {
            return Err(Error::Invalid(format!("{} penalty entries", op.count)));
        }
        if op.count == 0 {
            return Ok(());
        }
        let mut bytes = words(&[op.vocab, op.count, op.logits.1])?;
        for value in [op.frequency, op.presence, op.repetition] {
            bytes.extend_from_slice(&value.to_bits().to_le_bytes());
        }
        let buffers = [op.logits.0, op.entries.0];
        self.context.kernels.penalize.dispatch(
            self,
            &buffers.map(|b| b.buffer),
            &bytes,
            [op.count.div_ceil(PENALIZE_THREADS) as u32, 1, 1],
        );
        for buffer in buffers {
            self.retain(buffer.clone());
        }
        Ok(())
    }
}
