use candle_core::{CpuStorage, CustomOp3, Layout, Result, Shape, Tensor};
use rayon::prelude::*;

/// Output channels per register tile.
pub const OC_T: usize = 4;
const LANES: usize = 8;
/// Pixel vectors per register tile: `OC_T * MAX_XV` = 12 of the 16 ymm registers hold accumulators.
const MAX_XV: usize = 3;
/// Tail vectors may read up to a tile width past the last phase plane.
const SLACK: usize = MAX_XV * LANES + 16;
/// Output columns per cache segment (two full register tiles).
const SEG_W: usize = 2 * MAX_XV * LANES;
/// Input floats one channel block may touch per segment (~24 KB of the 32 KB L1).
const L1_INPUT_FLOATS: usize = 6144;
const MIN_IC_BLOCK: usize = 4;
/// Pixels per parallel task of the pointwise kernel.
const PW_CHUNK: usize = 4 * SEG_W;
/// Weight floats one task's group of output-channel tiles may use (~128 KB of L2).
const L2_WEIGHT_FLOATS: usize = 32768;

/// Activation fused into the conv epilogue.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Act {
    None,
    Relu,
    Silu,
}

impl Act {
    pub fn from_candle(act: Option<candle_nn::Activation>) -> Option<Self> {
        match act {
            None => Some(Self::None),
            Some(candle_nn::Activation::Relu) => Some(Self::Relu),
            Some(candle_nn::Activation::Silu) => Some(Self::Silu),
            Some(_) => None,
        }
    }
}

pub fn available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Zero-padded input split into `s*s` stride phases: `[(phase, channel, row, col)]` with planes of `hs x ws`.
pub struct PhasePlanes {
    pub data: Vec<f32>,
    pub hs: usize,
    pub ws: usize,
}

impl PhasePlanes {
    pub fn new(xb: &[f32], c: usize, h: usize, w: usize, s: usize, p: usize) -> Self {
        let hs = (h + 2 * p).div_ceil(s);
        let ws = (w + 2 * p).div_ceil(s).next_multiple_of(LANES);
        let plane = hs * ws;
        let mut data = vec![0f32; s * s * c * plane + SLACK];
        data[..s * s * c * plane]
            .par_chunks_mut(plane)
            .enumerate()
            .for_each(|(pc, dst)| {
                let (ph, ch) = (pc / c, pc % c);
                let (py, px) = (ph / s, ph % s);
                let src = &xb[ch * h * w..(ch + 1) * h * w];
                // plane column `col` reads input column `col * s + px - p`; only a contiguous range is in bounds
                let lo = p.saturating_sub(px).div_ceil(s);
                let hi = (w + p).saturating_sub(px).div_ceil(s).min(ws);
                for (r, row) in dst.chunks_mut(ws).enumerate() {
                    let iy = (r * s + py) as isize - p as isize;
                    if iy < 0 || iy >= h as isize || lo >= hi {
                        continue;
                    }
                    let srow = &src[iy as usize * w..(iy as usize + 1) * w];
                    let first = lo * s + px - p;
                    if s == 1 {
                        row[lo..hi].copy_from_slice(&srow[first..first + (hi - lo)]);
                    } else {
                        for (v, &x) in row[lo..hi].iter_mut().zip(srow[first..].iter().step_by(s)) {
                            *v = x;
                        }
                    }
                }
            });
        Self { data, hs, ws }
    }

    /// Offset of tap `(ky, kx)` relative to `(channel 0, output row 0, output col 0)`.
    fn tap_offsets(&self, c: usize, k: usize, s: usize) -> Vec<usize> {
        let plane = self.hs * self.ws;
        (0..k * k)
            .map(|t| {
                let (ky, kx) = (t / k, t % k);
                ((ky % s) * s + kx % s) * c * plane + (ky / s) * self.ws + kx / s
            })
            .collect()
    }
}

#[derive(Clone, Copy)]
struct SendPtr(*mut f32);
// SAFETY: every parallel task writes a disjoint set of output rows through this pointer.
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

impl SendPtr {
    fn get(self) -> *mut f32 {
        self.0
    }
}

fn silu_in_place(xs: &mut [f32]) {
    for v in xs {
        *v /= 1. + (-*v).exp();
    }
}

/// `(o, c, k, k)` -> `(o / OC_T, c, k*k, OC_T)` so each tap's output-channel weights are one broadcast source.
pub fn pack_weights(w: &Tensor) -> Result<Tensor> {
    let (o, c, k, _) = w.dims4()?;
    if o % OC_T != 0 {
        candle_core::bail!("direct conv packs {OC_T} output channels at a time, got {o}");
    }
    let src = w
        .to_dtype(candle_core::DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let kk = k * k;
    let mut packed = vec![0f32; o * c * kk];
    for oc in 0..o {
        let (t, j) = (oc / OC_T, oc % OC_T);
        for ic in 0..c {
            for tap in 0..kk {
                packed[((t * c + ic) * kk + tap) * OC_T + j] = src[(oc * c + ic) * kk + tap];
            }
        }
    }
    Tensor::from_vec(packed, (o / OC_T, c, kk, OC_T), &candle_core::Device::Cpu)
}

/// Dense conv with fused bias + activation on `pack_weights` output. Requires `available()`.
pub fn conv2d(
    xs: &Tensor,
    w: &Tensor,
    b: &Tensor,
    stride: usize,
    padding: usize,
    act: Act,
) -> Result<Tensor> {
    xs.contiguous()?.apply_op3_no_bwd(
        &w.contiguous()?,
        &b.contiguous()?,
        &DirectConv {
            stride,
            padding,
            act,
        },
    )
}

/// 1x1 conv with fused bias + activation on `pack_weights` output; reads the input in place. Requires `available()`.
pub fn pointwise(xs: &Tensor, w: &Tensor, b: &Tensor, act: Act) -> Result<Tensor> {
    xs.contiguous()?
        .apply_op3_no_bwd(&w.contiguous()?, &b.contiguous()?, &DirectPointwise { act })
}

/// Depthwise conv with fused bias + activation. Requires `available()`.
pub fn depthwise(
    xs: &Tensor,
    w: &Tensor,
    b: &Tensor,
    stride: usize,
    padding: usize,
    act: Act,
) -> Result<Tensor> {
    xs.contiguous()?.apply_op3_no_bwd(
        &w.contiguous()?,
        &b.contiguous()?,
        &DirectDepthwise {
            stride,
            padding,
            act,
        },
    )
}

struct DirectConv {
    stride: usize,
    padding: usize,
    act: Act,
}

pub(crate) fn f32_slices<'a>(items: [(&'a CpuStorage, &Layout); 3]) -> Result<[&'a [f32]; 3]> {
    let mut out: [&[f32]; 3] = [&[]; 3];
    for (o, (s, l)) in out.iter_mut().zip(items) {
        let CpuStorage::F32(v) = s else {
            candle_core::bail!("layout CPU kernels are f32 only");
        };
        *o = &v[l.start_offset()..];
    }
    Ok(out)
}

impl CustomOp3 for DirectConv {
    fn name(&self) -> &'static str {
        "direct-conv2d"
    }

    fn cpu_fwd(
        &self,
        sx: &CpuStorage,
        lx: &Layout,
        sw: &CpuStorage,
        lw: &Layout,
        sb: &CpuStorage,
        lb: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let [x, packed, bias] = f32_slices([(sx, lx), (sw, lw), (sb, lb)])?;
        let (bn, c, h, w) = lx.shape().dims4()?;
        let (tiles, ci, kk, oct) = lw.shape().dims4()?;
        let k = kk.isqrt();
        if ci != c || k * k != kk || oct != OC_T || !available() {
            candle_core::bail!(
                "direct conv: packed weight {:?} does not fit input {:?}",
                lw.shape(),
                lx.shape()
            );
        }
        let o = tiles * OC_T;
        let (s, p) = (self.stride, self.padding);
        let ho = (h + 2 * p - k) / s + 1;
        let wo = (w + 2 * p - k) / s + 1;
        // distinct (phase, row) pairs a tap set touches, times the segment's input columns
        let phase_rows = s * s * k.div_ceil(s);
        let ic_block =
            (L1_INPUT_FLOATS / (phase_rows * (SEG_W + k))).clamp(MIN_IC_BLOCK, c.max(MIN_IC_BLOCK));
        let tile_floats = c * kk * OC_T;
        // tiles sharing one task reuse each L1-resident input block; their weights should fit in L2
        let group = (L2_WEIGHT_FLOATS / tile_floats).clamp(1, tiles);
        let groups = tiles.div_ceil(group);
        let mut out = vec![0f32; bn * o * ho * wo];
        for bi in 0..bn {
            let planes = PhasePlanes::new(&x[bi * c * h * w..(bi + 1) * c * h * w], c, h, w, s, p);
            let taps = planes.tap_offsets(c, k, s);
            let ob = SendPtr(out[bi * o * ho * wo..].as_mut_ptr());
            (0..groups * ho).into_par_iter().for_each(|task| {
                let (g, oy) = (task / ho, task % ho);
                let tile_range = g * group..((g + 1) * group).min(tiles);
                for seg in (0..wo).step_by(SEG_W) {
                    let seg_end = (seg + SEG_W).min(wo);
                    for start in (0..c).step_by(ic_block) {
                        let blk = IcBlock {
                            start,
                            end: (start + ic_block).min(c),
                        };
                        let act = if blk.end == c { self.act } else { Act::None };
                        for t in tile_range.clone() {
                            let geo = RowGeom {
                                x: planes.data.as_ptr(),
                                plane: planes.hs * planes.ws,
                                row: oy * planes.ws,
                                taps: &taps,
                                w: packed[t * tile_floats..].as_ptr(),
                                bias: std::array::from_fn(|j| bias[t * OC_T + j]),
                                act,
                                blk,
                            };
                            let mut x0 = seg;
                            while x0 < seg_end {
                                let n = (seg_end - x0).min(MAX_XV * LANES);
                                // SAFETY: rows (t*OC_T + j, oy) belong to this task only; `n` bounds every access.
                                let dst: [*mut f32; OC_T] = std::array::from_fn(|j| unsafe {
                                    ob.get().add(((t * OC_T + j) * ho + oy) * wo + x0)
                                });
                                unsafe {
                                    match n.div_ceil(LANES) {
                                        1 => conv_tile::<1>(&geo, x0, n, dst),
                                        2 => conv_tile::<2>(&geo, x0, n, dst),
                                        _ => conv_tile::<3>(&geo, x0, n, dst),
                                    }
                                }
                                x0 += n;
                            }
                        }
                    }
                }
            });
        }
        Ok((CpuStorage::F32(out), Shape::from((bn, o, ho, wo))))
    }
}

/// Half-open input-channel range handled by one pass over an output row.
#[derive(Clone, Copy)]
struct IcBlock {
    start: usize,
    end: usize,
}

struct DirectPointwise {
    act: Act,
}

impl CustomOp3 for DirectPointwise {
    fn name(&self) -> &'static str {
        "direct-pointwise"
    }

    fn cpu_fwd(
        &self,
        sx: &CpuStorage,
        lx: &Layout,
        sw: &CpuStorage,
        lw: &Layout,
        sb: &CpuStorage,
        lb: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let [x, packed, bias] = f32_slices([(sx, lx), (sw, lw), (sb, lb)])?;
        let (bn, c, h, w) = lx.shape().dims4()?;
        let (tiles, ci, kk, oct) = lw.shape().dims4()?;
        if ci != c || kk != 1 || oct != OC_T || !available() {
            candle_core::bail!(
                "direct pointwise: packed weight {:?} does not fit input {:?}",
                lw.shape(),
                lx.shape()
            );
        }
        let o = tiles * OC_T;
        let hw = h * w;
        let ic_block = (L1_INPUT_FLOATS / SEG_W).clamp(MIN_IC_BLOCK, c.max(MIN_IC_BLOCK));
        let tile_floats = c * OC_T;
        let group = (L2_WEIGHT_FLOATS / tile_floats).clamp(1, tiles);
        let groups = tiles.div_ceil(group);
        // whole vectors run in place; the final partial vector (which would read past the tensor) uses a scratch
        let body = hw / LANES * LANES;
        let chunks = body.div_ceil(PW_CHUNK);
        let taps = [0usize];
        let mut out = vec![0f32; bn * o * hw];
        for bi in 0..bn {
            let xb = &x[bi * c * hw..(bi + 1) * c * hw];
            let ob = SendPtr(out[bi * o * hw..].as_mut_ptr());
            (0..groups * chunks).into_par_iter().for_each(|task| {
                let (g, chunk) = (task / chunks, task % chunks);
                let tile_range = g * group..((g + 1) * group).min(tiles);
                let chunk_end = ((chunk + 1) * PW_CHUNK).min(body);
                for seg in (chunk * PW_CHUNK..chunk_end).step_by(SEG_W) {
                    let seg_end = (seg + SEG_W).min(chunk_end);
                    for start in (0..c).step_by(ic_block) {
                        let blk = IcBlock {
                            start,
                            end: (start + ic_block).min(c),
                        };
                        let act = if blk.end == c { self.act } else { Act::None };
                        for t in tile_range.clone() {
                            let geo = RowGeom {
                                x: xb.as_ptr(),
                                plane: hw,
                                row: 0,
                                taps: &taps,
                                w: packed[t * tile_floats..].as_ptr(),
                                bias: std::array::from_fn(|j| bias[t * OC_T + j]),
                                act,
                                blk,
                            };
                            let mut x0 = seg;
                            while x0 < seg_end {
                                let n = (seg_end - x0).min(MAX_XV * LANES);
                                // SAFETY: pixels [x0, x0+n) of planes t*OC_T+j belong to this task only.
                                let dst: [*mut f32; OC_T] = std::array::from_fn(|j| unsafe {
                                    ob.get().add((t * OC_T + j) * hw + x0)
                                });
                                unsafe {
                                    match n.div_ceil(LANES) {
                                        1 => conv_tile::<1>(&geo, x0, n, dst),
                                        2 => conv_tile::<2>(&geo, x0, n, dst),
                                        _ => conv_tile::<3>(&geo, x0, n, dst),
                                    }
                                }
                                x0 += n;
                            }
                        }
                    }
                }
            });
            if body < hw {
                let tail = hw - body;
                let mut scratch = vec![0f32; c * LANES];
                for (dst, src) in scratch.chunks_mut(LANES).zip(xb.chunks(hw)) {
                    dst[..tail].copy_from_slice(&src[body..]);
                }
                let blk = IcBlock { start: 0, end: c };
                (0..tiles).into_par_iter().for_each(|t| {
                    let geo = RowGeom {
                        x: scratch.as_ptr(),
                        plane: LANES,
                        row: 0,
                        taps: &taps,
                        w: packed[t * tile_floats..].as_ptr(),
                        bias: std::array::from_fn(|j| bias[t * OC_T + j]),
                        act: self.act,
                        blk,
                    };
                    // SAFETY: the tail pixels of planes t*OC_T+j belong to this task only.
                    let dst: [*mut f32; OC_T] = std::array::from_fn(|j| unsafe {
                        ob.get().add((t * OC_T + j) * hw + body)
                    });
                    unsafe { conv_tile::<1>(&geo, 0, tail, dst) };
                });
            }
        }
        Ok((CpuStorage::F32(out), Shape::from((bn, o, h, w))))
    }
}

struct RowGeom<'a> {
    /// Phase planes of the current image.
    x: *const f32,
    plane: usize,
    /// Offset of the output row inside a plane.
    row: usize,
    taps: &'a [usize],
    /// Packed weights of this output-channel tile.
    w: *const f32,
    bias: [f32; OC_T],
    /// Applied only after the final channel block.
    act: Act,
    blk: IcBlock,
}

// SAFETY: raw pointers are only read; the owning buffers outlive the parallel loop.
unsafe impl Sync for RowGeom<'_> {}

/// One `OC_T x n` output tile (`n <= XV * LANES`) of one output row.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn conv_tile<const XV: usize>(g: &RowGeom, x0: usize, n: usize, dst: [*mut f32; OC_T]) {
    use std::arch::x86_64::*;
    let (act, blk) = (g.act, g.blk);
    let mut acc = [[_mm256_setzero_ps(); XV]; OC_T];
    for (j, row) in acc.iter_mut().enumerate() {
        if blk.start == 0 {
            *row = [_mm256_set1_ps(g.bias[j]); XV];
            continue;
        }
        // later channel blocks resume from the partial sums already stored in the output row
        for (v, a) in row.iter_mut().enumerate() {
            let lanes = n.saturating_sub(v * LANES).min(LANES);
            let mut tmp = [0f32; LANES];
            std::ptr::copy_nonoverlapping(dst[j].add(v * LANES), tmp.as_mut_ptr(), lanes);
            *a = _mm256_loadu_ps(tmp.as_ptr());
        }
    }
    let base = g.x.add(g.row + x0);
    let mut wp = g.w.add(blk.start * g.taps.len() * OC_T);
    for ic in blk.start..blk.end {
        let xc = base.add(ic * g.plane);
        for &off in g.taps {
            let src = xc.add(off);
            let mut xin = [_mm256_setzero_ps(); XV];
            for (v, xv) in xin.iter_mut().enumerate() {
                *xv = _mm256_loadu_ps(src.add(v * LANES));
            }
            for row in acc.iter_mut() {
                let wj = _mm256_broadcast_ss(&*wp);
                wp = wp.add(1);
                for (a, &xv) in row.iter_mut().zip(&xin) {
                    *a = _mm256_fmadd_ps(xv, wj, *a);
                }
            }
        }
    }
    // epilogue inline: passing `&acc` to a helper forces the accumulators onto the stack
    let zero = _mm256_setzero_ps();
    for (j, row) in acc.into_iter().enumerate() {
        for (v, a) in row.into_iter().enumerate() {
            let r = if act == Act::Relu {
                _mm256_max_ps(a, zero)
            } else {
                a
            };
            let lanes = n.saturating_sub(v * LANES).min(LANES);
            if lanes == LANES {
                _mm256_storeu_ps(dst[j].add(v * LANES), r);
            } else if lanes > 0 {
                let mut tmp = [0f32; LANES];
                _mm256_storeu_ps(tmp.as_mut_ptr(), r);
                std::ptr::copy_nonoverlapping(tmp.as_ptr(), dst[j].add(v * LANES), lanes);
            }
        }
        if act == Act::Silu {
            silu_in_place(std::slice::from_raw_parts_mut(dst[j], n));
        }
    }
}

#[cfg(not(target_arch = "x86_64"))]
unsafe fn conv_tile<const XV: usize>(_: &RowGeom, _: usize, _: usize, _: [*mut f32; OC_T]) {
    unreachable!("direct conv requires x86_64 AVX2")
}

/// Per-channel invariants of one depthwise output plane.
struct DwChannel<'a> {
    taps: &'a [usize],
    w: &'a [f32],
    bias: f32,
    act: Act,
}

struct DirectDepthwise {
    stride: usize,
    padding: usize,
    act: Act,
}

impl CustomOp3 for DirectDepthwise {
    fn name(&self) -> &'static str {
        "direct-depthwise"
    }

    fn cpu_fwd(
        &self,
        sx: &CpuStorage,
        lx: &Layout,
        sw: &CpuStorage,
        lw: &Layout,
        sb: &CpuStorage,
        lb: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let [x, wt, bias] = f32_slices([(sx, lx), (sw, lw), (sb, lb)])?;
        let (bn, c, h, w) = lx.shape().dims4()?;
        let (cw, one, k, k2) = lw.shape().dims4()?;
        if cw != c || one != 1 || k != k2 || !available() {
            candle_core::bail!(
                "direct depthwise: weight {:?} vs input {:?}",
                lw.shape(),
                lx.shape()
            );
        }
        let (s, p) = (self.stride, self.padding);
        let ho = (h + 2 * p - k) / s + 1;
        let wo = (w + 2 * p - k) / s + 1;
        let mut out = vec![0f32; bn * c * ho * wo];
        for bi in 0..bn {
            let planes = PhasePlanes::new(&x[bi * c * h * w..(bi + 1) * c * h * w], c, h, w, s, p);
            let taps = planes.tap_offsets(c, k, s);
            let plane = planes.hs * planes.ws;
            out[bi * c * ho * wo..(bi + 1) * c * ho * wo]
                .par_chunks_mut(ho * wo)
                .enumerate()
                .for_each(|(ch, dst)| {
                    let xc = planes.data[ch * plane..].as_ptr();
                    let dw = DwChannel {
                        taps: &taps,
                        w: &wt[ch * k * k..(ch + 1) * k * k],
                        bias: bias[ch],
                        act: self.act,
                    };
                    for (oy, row) in dst.chunks_mut(wo).enumerate() {
                        // SAFETY: loads stay inside the phase planes plus SLACK; stores are bounded by `row`.
                        unsafe { depthwise_row(xc.add(oy * planes.ws), &dw, row) };
                    }
                });
        }
        Ok((CpuStorage::F32(out), Shape::from((bn, c, ho, wo))))
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn depthwise_row(x: *const f32, dw: &DwChannel, row: &mut [f32]) {
    use std::arch::x86_64::*;
    let (taps, wk, bias, act) = (dw.taps, dw.w, dw.bias, dw.act);
    let zero = _mm256_setzero_ps();
    let mut x0 = 0;
    while x0 < row.len() {
        let mut acc = _mm256_set1_ps(bias);
        for (&off, &wv) in taps.iter().zip(wk) {
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(x.add(off + x0)), _mm256_set1_ps(wv), acc);
        }
        if act == Act::Relu {
            acc = _mm256_max_ps(acc, zero);
        }
        let lanes = (row.len() - x0).min(LANES);
        let mut tmp = [0f32; LANES];
        _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
        row[x0..x0 + lanes].copy_from_slice(&tmp[..lanes]);
        x0 += LANES;
    }
    if act == Act::Silu {
        silu_in_place(row);
    }
}

#[cfg(not(target_arch = "x86_64"))]
unsafe fn depthwise_row(_: *const f32, _: &DwChannel, _: &mut [f32]) {
    unreachable!("direct depthwise requires x86_64 AVX2")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::rel_err;
    use candle_core::Device;

    fn reference(y: Tensor, act: Act) -> Result<Tensor> {
        match act {
            Act::None => Ok(y),
            Act::Relu => y.relu(),
            Act::Silu => y.silu(),
        }
    }

    #[test]
    fn direct_conv_matches_candle() -> Result<()> {
        if !available() {
            return Ok(());
        }
        let dev = Device::Cpu;
        // 45 input channels span several channel blocks; widths up to 101 span several segments
        let cases = [
            (3, 1, 1, 9, 7),
            (3, 2, 1, 10, 11),
            (2, 1, 0, 6, 5),
            (5, 2, 2, 9, 9),
            (3, 1, 1, 30, 101),
            (3, 2, 1, 21, 99),
        ];
        for (k, s, p, h, w) in cases {
            for act in [Act::None, Act::Relu, Act::Silu] {
                let x = Tensor::randn(0f32, 1., (2, 45, h, w), &dev)?;
                let wt = Tensor::randn(0f32, 1., (8, 45, k, k), &dev)?;
                let b = Tensor::randn(0f32, 1., 8, &dev)?;
                let want = reference(
                    x.conv2d(&wt, p, s, 1, 1)?
                        .broadcast_add(&b.reshape((1, 8, 1, 1))?)?,
                    act,
                )?;
                let got = conv2d(&x, &pack_weights(&wt)?, &b, s, p, act)?;
                assert_eq!(got.dims(), want.dims());
                // many-term fp32 sums: compare relative to the output scale
                let err = rel_err(&got, &want)?;
                assert!(err < 1e-5, "k={k} s={s} {h}x{w} {act:?} rel err={err}");
            }
        }
        Ok(())
    }

    #[test]
    fn direct_pointwise_matches_candle() -> Result<()> {
        if !available() {
            return Ok(());
        }
        let dev = Device::Cpu;
        // planes not a multiple of 8 floats take the tail scratch; 2x3 has no whole vector at all; 300 channels span blocks
        for (c, h, w) in [
            (7, 8, 8),
            (300, 5, 5),
            (45, 50, 50),
            (20, 17, 31),
            (9, 2, 3),
        ] {
            for act in [Act::None, Act::Relu, Act::Silu] {
                let x = Tensor::randn(0f32, 1., (2, c, h, w), &dev)?;
                let wt = Tensor::randn(0f32, 1., (12, c, 1, 1), &dev)?;
                let b = Tensor::randn(0f32, 1., 12, &dev)?;
                let want = reference(
                    x.conv2d(&wt, 0, 1, 1, 1)?
                        .broadcast_add(&b.reshape((1, 12, 1, 1))?)?,
                    act,
                )?;
                let got = pointwise(&x, &pack_weights(&wt)?, &b, act)?;
                let err = rel_err(&got, &want)?;
                assert!(err < 1e-5, "c={c} {h}x{w} {act:?} rel err={err}");
            }
        }
        Ok(())
    }

    #[test]
    fn direct_depthwise_matches_candle() -> Result<()> {
        if !available() {
            return Ok(());
        }
        let dev = Device::Cpu;
        for (k, s, h, w) in [
            (3, 1, 9, 7),
            (3, 2, 10, 10),
            (3, 2, 9, 11),
            (5, 1, 8, 13),
            (5, 2, 7, 7),
            (5, 1, 25, 25),
        ] {
            for act in [Act::None, Act::Relu, Act::Silu] {
                let c = 6;
                let p = (k - 1) / 2;
                let x = Tensor::randn(0f32, 1., (2, c, h, w), &dev)?;
                let wt = Tensor::randn(0f32, 1., (c, 1, k, k), &dev)?;
                let b = Tensor::randn(0f32, 1., c, &dev)?;
                let want = reference(
                    x.conv2d(&wt, p, s, 1, c)?
                        .broadcast_add(&b.reshape((1, c, 1, 1))?)?,
                    act,
                )?;
                let got = depthwise(&x, &wt, &b, s, p, act)?;
                // many-term fp32 sums: compare relative to the output scale
                let err = rel_err(&got, &want)?;
                assert!(err < 1e-5, "k={k} s={s} {h}x{w} {act:?} rel err={err}");
            }
        }
        Ok(())
    }
}
