//! Row ops (RMS norm, softmax, reductions), row selection (index_select, gather) and a strided batched f32 matmul.

use std::sync::Arc;

use crate::memory::HostBuffer;
use crate::{Error, Recorder, Result};

pub(crate) const BINDINGS: usize = 3;
pub(crate) const ROWS_PUSH_BYTES: u32 = 7 * 4;
pub(crate) const INDEX_PUSH_BYTES: u32 = 8 * 4;
pub(crate) const MATMUL_PUSH_BYTES: u32 = 12 * 4;
const MAX_GROUPS_X: usize = 65_535;
const INDEX_THREADS: usize = 256;
const MATMUL_THREADS: usize = 64;
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
}

impl RowOp {
    /// Whether the op writes one value a row rather than a row.
    pub fn reduces(self) -> bool {
        !matches!(self, RowOp::RmsNorm | RowOp::Softmax)
    }
}

/// `rows` contiguous rows of `cols` from `x_offset`; `alpha` (`cols` from `alpha_offset`) is read by `RmsNorm` alone.
pub struct Rows<'a> {
    pub op: RowOp,
    pub rows: usize,
    pub cols: usize,
    pub eps: f32,
    pub x: &'a Arc<HostBuffer>,
    pub x_offset: usize,
    pub alpha: Option<(&'a Arc<HostBuffer>, usize)>,
    pub out: &'a Arc<HostBuffer>,
    pub out_offset: usize,
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
    pub src: (&'a Arc<HostBuffer>, usize),
    pub ids: (&'a Arc<HostBuffer>, usize),
    pub out: &'a Arc<HostBuffer>,
}

/// One operand of [`MatMul`]: its buffer, first element, and strides between batches, rows and columns.
#[derive(Clone, Copy)]
pub struct MatOperand<'a> {
    pub buffer: &'a Arc<HostBuffer>,
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
    pub out: &'a Arc<HostBuffer>,
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
        if op.rows == 0 {
            return Ok(());
        }
        let (alpha, alpha_offset) = op.alpha.unwrap_or((op.x, 0));
        if op.op == RowOp::RmsNorm && op.alpha.is_none() {
            return Err(Error::Invalid("an RMS norm takes its alpha".into()));
        }
        let mut bytes = words(&[op.rows, op.cols, op.op as usize])?;
        bytes.extend_from_slice(&op.eps.to_bits().to_le_bytes());
        bytes.extend(words(&[op.x_offset, alpha_offset, op.out_offset])?);
        let groups_x = op.rows.min(MAX_GROUPS_X);
        self.context.kernels.rows_f32.dispatch(
            self,
            &[op.x.buffer, alpha.buffer, op.out.buffer],
            &bytes,
            [groups_x as u32, op.rows.div_ceil(groups_x) as u32, 1],
        );
        for buffer in [op.x, alpha, op.out] {
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
        let groups = count.div_ceil(MATMUL_THREADS).min(MAX_GROUPS) as u32;
        self.context.kernels.matmul_f32.dispatch(
            self,
            &[a.buffer.buffer, b.buffer.buffer, op.out.buffer],
            &bytes,
            [groups, 1, 1],
        );
        for buffer in [a.buffer, b.buffer, op.out] {
            self.retain(buffer.clone());
        }
        Ok(())
    }
}

/// A buffer and the element an operand starts at.
pub type At<'a> = (&'a Arc<HostBuffer>, usize);

pub(crate) const CONV_STEP_BINDINGS: usize = 6;
pub(crate) const CONV_STEP_PUSH_BYTES: u32 = 9 * 4;
pub(crate) const SCAN_STEP_BINDINGS: usize = 10;
pub(crate) const SCAN_STEP_PUSH_BYTES: u32 = 13 * 4;
const STEP_THREADS: usize = 64;

/// One decode step of Mamba's causal conv on a conv-state pool (`(rows, channels, kernel)` f32, in place), each batch
/// row's state the pool row `slots` names; `out` is `(batch, channels)`.
pub struct ConvStep<'a> {
    pub batch: usize,
    pub channels: usize,
    pub kernel: usize,
    pub x: At<'a>,
    pub weight: At<'a>,
    pub bias: Option<At<'a>>,
    pub pool: At<'a>,
    pub slots: At<'a>,
    pub out: &'a Arc<HostBuffer>,
}

/// One decode step of Mamba-2's selective scan on a state pool (`(rows, heads, head_dim, state)` f32, in place); `a`,
/// `d` and `dt_bias` are `(heads,)` from their buffers' first element; `out` is `(batch, heads, head_dim)`.
pub struct ScanStep<'a> {
    pub batch: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub groups: usize,
    pub state: usize,
    pub dt_range: [f32; 2],
    pub x: At<'a>,
    pub dt: At<'a>,
    pub a: &'a Arc<HostBuffer>,
    pub b: At<'a>,
    pub c: At<'a>,
    pub d: &'a Arc<HostBuffer>,
    pub dt_bias: &'a Arc<HostBuffer>,
    pub pool: At<'a>,
    pub slots: At<'a>,
    pub out: &'a Arc<HostBuffer>,
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

    pub fn ssm_scan_step(&self, op: &ScanStep<'_>) -> Result<()> {
        if op.groups == 0 || !op.heads.is_multiple_of(op.groups) {
            return Err(Error::Invalid(format!(
                "{} heads over {} groups",
                op.heads, op.groups
            )));
        }
        let mut bytes = words(&[op.batch, op.heads, op.head_dim, op.groups, op.state])?;
        for v in op.dt_range {
            bytes.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        bytes.extend(words(&[
            op.pool.1, op.slots.1, op.x.1, op.dt.1, op.b.1, op.c.1,
        ])?);
        let buffers = [
            op.x.0, op.dt.0, op.a, op.b.0, op.c.0, op.d, op.dt_bias, op.pool.0, op.slots.0, op.out,
        ];
        let groups = (op.batch * op.heads * op.head_dim)
            .div_ceil(STEP_THREADS)
            .min(MAX_GROUPS) as u32;
        self.context.kernels.ssm_scan_step.dispatch(
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
}

pub(crate) const MOE_MATVEC_BINDINGS: usize = 4;
pub(crate) const MOE_MATVEC_PUSH_BYTES: u32 = 9 * 4;
pub(crate) const ROUTER_BINDINGS: usize = 4;
pub(crate) const ROUTER_PUSH_BYTES: u32 = 13 * 4;
// the matvec shaders' rows a workgroup, and the router shader's limits
const MATVEC_ROWS: usize = 2;
const MAX_ROUTER_EXPERTS: usize = 1024;
const MAX_ROUTER_TOP_K: usize = 32;

/// `y[slot] = W[ids[slot]] x[slot / x_div]` over a stack of Q5_0 experts (each `rows x cols`) in `weights`; `y` is
/// `(slots, rows)`.
pub struct MoeMatVec<'a> {
    pub weights: &'a Arc<dyn crate::Bound>,
    pub experts: usize,
    pub rows: usize,
    pub cols: usize,
    pub slots: usize,
    /// slots a row of `x` serves: the routes of one token, or 1 for a row a route
    pub x_div: usize,
    pub x: At<'a>,
    pub ids: At<'a>,
    pub y: &'a Arc<HostBuffer>,
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
    pub bias: Option<&'a Arc<HostBuffer>>,
    pub values: &'a Arc<HostBuffer>,
    pub indices: &'a Arc<HostBuffer>,
}

impl Recorder<'_> {
    pub fn moe_mat_vec_q5_0(&self, op: &MoeMatVec<'_>) -> Result<()> {
        const BLOCK_VALUES: usize = 32;
        const BLOCK_BYTES: usize = 22;
        let blocks = op.cols / BLOCK_VALUES;
        let expert_bytes = op.rows * blocks * BLOCK_BYTES;
        if !op.cols.is_multiple_of(BLOCK_VALUES)
            || op.x_div == 0
            || op.weights.bytes() < op.experts * expert_bytes
            || !op.weights.offset().is_multiple_of(2)
            || op.slots > MAX_GROUPS_X
        {
            return Err(Error::Invalid(format!(
                "a {} x {}x{} Q5_0 expert matvec over {} slots",
                op.experts, op.rows, op.cols, op.slots
            )));
        }
        if op.slots == 0 {
            return Ok(());
        }
        let groups = op.rows.div_ceil(MATVEC_ROWS);
        let groups_x = groups.min(MAX_GROUPS_X);
        let bytes = words(&[
            op.rows,
            op.cols,
            op.weights.offset() as usize / 2,
            op.x.1,
            0,
            groups_x,
            op.x_div,
            op.ids.1,
            expert_bytes / 2,
        ])?;
        let buffers = [
            op.weights.vk_buffer(),
            op.x.0.buffer,
            op.y.buffer,
            op.ids.0.buffer,
        ];
        self.context.kernels.moe_mat_vec_q5_0.dispatch(
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
        for buffer in [op.x.0, op.ids.0, op.y] {
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
