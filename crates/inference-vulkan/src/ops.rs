//! Row ops (RMS norm, softmax, reductions), row selection (index_select, gather) and a strided batched f32 matmul.

use std::sync::Arc;

use crate::memory::Buffer;
use crate::{Error, Recorder, Result};

pub(crate) const BINDINGS: usize = 3;
pub(crate) const ROWS_PUSH_BYTES: u32 = 7 * 4;
pub(crate) const INDEX_PUSH_BYTES: u32 = 8 * 4;
pub(crate) const MATMUL_PUSH_BYTES: u32 = 12 * 4;
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
    pub x: &'a Arc<Buffer>,
    pub x_offset: usize,
    pub alpha: Option<(&'a Arc<Buffer>, usize)>,
    pub out: &'a Arc<Buffer>,
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
        let buffers = [a.buffer.buffer, b.buffer.buffer, op.out.buffer];
        let work = (count as u64).saturating_mul(op.k as u64);
        self.work.set(self.work.get().saturating_add(work));
        self.profile_shape(
            || format!("b={} m={} n={} k={}", op.batch, op.m, op.n, op.k),
            count * op.k,
        );
        if op.m >= TILED_MIN && op.n >= TILED_MIN && op.batch <= MAX_GROUPS_Z {
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
        if context.coop.is_none() {
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
        let workgroups = op.rows * op.groups;
        if workgroups == 0 || op.group_size == 0 || workgroups > MAX_DISPATCH_GROUPS {
            return Err(Error::Invalid(format!(
                "gated RMSNorm of {} rows of {} groups of {}",
                op.rows, op.groups, op.group_size
            )));
        }
        let mut bytes = words(&[op.group_size, op.groups])?;
        bytes.extend_from_slice(&op.eps.to_bits().to_le_bytes());
        bytes.extend(words(&[op.x.1, op.gate.1, op.weight.1])?);
        let buffers = [op.x.0, op.gate.0, op.weight.0, op.out];
        self.context.kernels.gated_rmsnorm.dispatch(
            self,
            &buffers.map(|b| b.buffer),
            &bytes,
            [workgroups as u32, 1, 1],
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
            && let Some(coop) = self.context.coop.as_ref()
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
        let s_off = self.quantize_rows(op.x, op.slots.div_ceil(op.x_div), op.cols, op.scratch)?;
        if op.slots > MOE_TILED_MIN_SLOTS {
            return self.moe_mm(op, s_off, expert_bytes);
        }
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
            op.scratch.buffer,
            op.y.buffer,
            op.ids.0.buffer,
            op.scratch.buffer,
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
pub(crate) const GATED_RMSNORM_PUSH_BYTES: u32 = 6 * 4;
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
// the shared tiles' head size, sized for occupancy
const MAX_PAGED_HEAD: usize = 128;
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
        if s.head_size == 0
            || s.head_size > MAX_PAGED_HEAD
            || !s.head_size.is_multiple_of(2 * PAGED_X)
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
                    op.rows.div_ceil(PREFILL_ROWS) as u32,
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
            || s.head_size > MAX_PAGED_HEAD
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
