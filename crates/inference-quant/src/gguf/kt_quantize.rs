//! ik_llama.cpp's trellis quantizers (`QuantizerIQKT` and `quantize_row_iq*_kt_impl` in iqk_quantize.cpp) as its
//! AVX2 build runs them: each group of 8 values snaps to one of the codebook's codes, searched among the codes near
//! the cluster its coordinates fall in. Sums keep the build's eight lanes and g++'s fused multiply-adds.

// the loops index as ik's C++ does, so each line can be checked against it
#![allow(clippy::needless_range_loop)]

use std::sync::OnceLock;

use super::iq_quantize::{IQ4K_VALUES, IQ4NL_INDEX};
use super::kernel::GgufType;
use super::trellis::{IK, IK_INDEX_OFFSET};

const QK_K: usize = 256;
const GROUP: usize = 8;
const BLOCK: usize = 32;
const NG: usize = BLOCK / GROUP;
const NBLOCK: usize = QK_K / BLOCK;
// 256 clusters: one per side of each coordinate's split, which ik bins directly
const CLUSTERS: usize = 256;
pub(super) const F32_ROW_SCALE: usize = 4;
pub(super) const BLOCK_IQ2_KT: usize = 68;
pub(super) const BLOCK_IQ1_KT: usize = 56;
pub(super) const BLOCK_IQ3_KT: usize = 100;
const IQ2_KT_NEIGHBOURS: usize = 8;
const IQ1_KT_NEIGHBOURS: usize = 32;
const IQ3_KT_NEIGHBOURS: usize = 8;
// `ggml_get_quantize_fudge_factor`'s defaults
const IQ1_KT_FUDGE: f32 = 1.07;
// IQ1_KT tries its start scale again this much lower
const IQ1_KT_RETRY_STEP: f32 = 8.0;
// IQ3_KT's start scales, seven of them this far apart
const IQ3_KT_STEP: f32 = 8.0;
const IQ3_KT_TRIES: i32 = 3;
const IQ2_KT_SIGMA_SCALE: f32 = 2.0;
const KT_EPS2: f32 = 1e-14;
// a weight for values the weighting would leave at nothing
const KT_WEIGHT: f32 = 1e-4;
const KT_EPS: f32 = 1e-16;
// the scale search's steps either side of the max-based row scale
const KT_SCALE_TRIES: i32 = 9;

type V = [f32; 8];

#[inline(always)]
fn fma(a: f32, b: f32, c: f32) -> f32 {
    a.mul_add(b, c)
}

fn add(a: V, b: V) -> V {
    std::array::from_fn(|i| a[i] + b[i])
}

// `_mm256_permute2f128_ps` with 0x20 (both low halves) or 0x31 (both high halves)
fn halves(a: V, b: V, high: bool) -> V {
    let o = if high { 4 } else { 0 };
    std::array::from_fn(|i| if i < 4 { a[o + i] } else { b[o + i - 4] })
}

fn unpack(a: V, b: V, high: bool) -> V {
    let o = if high { 2 } else { 0 };
    [
        a[o],
        b[o],
        a[o + 1],
        b[o + 1],
        a[o + 4],
        b[o + 4],
        a[o + 5],
        b[o + 5],
    ]
}

// `hsum_float_8x8`: lane i the sum of `acc[i]`, added in the intrinsics' order
fn hsum8x8(mut acc: [V; 8]) -> V {
    for i in 0..4 {
        acc[i] = add(
            halves(acc[i], acc[i + 4], false),
            halves(acc[i], acc[i + 4], true),
        );
    }
    for i in 0..2 {
        acc[i] = add(
            unpack(acc[i], acc[i + 2], false),
            unpack(acc[i], acc[i + 2], true),
        );
    }
    add(unpack(acc[0], acc[1], false), unpack(acc[0], acc[1], true))
}

fn hsum8(v: &V) -> f32 {
    let r: [f32; 4] = std::array::from_fn(|i| v[i] + v[i + 4]);
    (r[0] + r[2]) + (r[1] + r[3])
}

/// `QuantizerIQKT` with 256 clusters: every code's 8 values, each coordinate's split point, and per cluster the codes
/// a search there tries (with their values in that order).
pub(super) struct KtQuantizer {
    values: Vec<f32>,
    mid: V,
    in_cluster: Vec<Vec<u32>>,
    c_values: Vec<Vec<f32>>,
}

impl KtQuantizer {
    fn new(num_bits: u32, is_abs: bool, num_neighbours: usize) -> Self {
        let ncodes = 1usize << num_bits;
        let mut values = Vec::with_capacity(ncodes * GROUP);
        for i in 0..ncodes as u32 {
            for v in IK.values::<GROUP>(i + IK_INDEX_OFFSET) {
                values.push(if is_abs { v.abs() } else { v } as f32);
            }
        }
        let point = |ip: usize| &values[ip * GROUP..][..GROUP];
        let mut mid = [0f32; 8];
        for k in 0..GROUP {
            if is_abs {
                let mut p = (0..ncodes)
                    .map(|ip| point(ip)[k] as i32)
                    .collect::<Vec<_>>();
                p.sort_unstable();
                mid[k] = 0.5 * (p[ncodes / 2] + p[ncodes / 2 - 1]) as f32;
            } else {
                let (lo, hi) = (0..ncodes)
                    .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), ip| {
                        (lo.min(point(ip)[k]), hi.max(point(ip)[k]))
                    });
                mid[k] = 0.5 * (lo + hi);
            }
        }
        let cluster_of = |v: &[f32]| {
            (0..GROUP)
                .filter(|&k| v[k] > mid[k])
                .map(|k| 1 << k)
                .sum::<usize>()
        };
        let (mut sump, mut counts) = (vec![0f32; CLUSTERS * GROUP], vec![0u32; CLUSTERS]);
        for ip in 0..ncodes {
            let u = cluster_of(point(ip));
            counts[u] += 1;
            for k in 0..GROUP {
                sump[GROUP * u + k] += point(ip)[k];
            }
        }
        let clusters = (0..CLUSTERS * GROUP)
            .map(|i| sump[i] / counts[i / GROUP] as f32)
            .collect::<Vec<_>>();
        let in_cluster = finalize_clusters(num_neighbours, &values, &clusters);
        let c_values = in_cluster
            .iter()
            .map(|points| {
                points
                    .iter()
                    .flat_map(|&p| point(p as usize).to_vec())
                    .collect()
            })
            .collect();
        Self {
            values,
            mid,
            in_cluster,
            c_values,
        }
    }

    pub(super) fn code(&self, index: u32) -> &[f32] {
        &self.values[index as usize * GROUP..][..GROUP]
    }

    /// The best code for each group of 8 of `xb` at scale `d`.
    pub(super) fn find_best_match(&self, d: f32, xb: &[f32], w: &[f32]) -> [u32; NG] {
        if d == 0.0 {
            return [0; NG];
        }
        let id = 1.0 / d;
        std::array::from_fn(|l| {
            let vx: V = std::array::from_fn(|k| id * xb[GROUP * l + k]);
            let vw: V = std::array::from_fn(|k| w[GROUP * l + k]);
            let u = (0..GROUP)
                .filter(|&k| vx[k] > self.mid[k])
                .map(|k| 1 << k)
                .sum::<usize>();
            let points = &self.in_cluster[u];
            let values = if points.is_empty() {
                &self.values
            } else {
                &self.c_values[u]
            };
            let (mut vbest, mut index) = ([f32::INFINITY; 8], [usize::MAX; 8]);
            for j in (0..values.len() / GROUP).step_by(8) {
                let sqx: [V; 8] = std::array::from_fn(|i| {
                    let q = &values[GROUP * (j + i)..][..GROUP];
                    std::array::from_fn(|k| {
                        let diff = q[k] - vx[k];
                        vw[k] * (diff * diff)
                    })
                });
                let score = hsum8x8(sqx);
                for lane in 0..8 {
                    if score[lane] < vbest[lane] {
                        index[lane] = j + lane;
                    }
                    vbest[lane] = if vbest[lane] < score[lane] {
                        vbest[lane]
                    } else {
                        score[lane]
                    };
                }
            }
            let (mut best, mut jbest) = (f32::INFINITY, usize::MAX);
            for lane in 0..8 {
                if vbest[lane] < best {
                    (best, jbest) = (vbest[lane], index[lane]);
                }
            }
            if points.is_empty() {
                jbest as u32
            } else {
                points[jbest]
            }
        })
    }

    /// The scale fitting `xb` best over the codes, and its score.
    pub(super) fn find_best_scale(&self, xb: &[f32], w: &[f32], idx: &[u32]) -> (f32, f32) {
        let (mut vqx, mut vq2) = ([0f32; 8], [0f32; 8]);
        for (l, &index) in idx.iter().enumerate() {
            let q = self.code(index);
            for k in 0..GROUP {
                let vqw = q[k] * w[GROUP * l + k];
                vqx[k] = fma(vqw, xb[GROUP * l + k], vqx[k]);
                vq2[k] = fma(vqw, q[k], vq2[k]);
            }
        }
        let (sumqx, sumq2) = (hsum8(&vqx), hsum8(&vq2));
        if sumq2 > 0.0 {
            (sumqx / sumq2, sumqx * sumqx / sumq2)
        } else {
            (0.0, 0.0)
        }
    }
}

// per code its `num_neighbours` nearest clusters, and per cluster the codes that name it, padded to a multiple of 8
// with the nearest codes that do not
fn finalize_clusters(num_neighbours: usize, values: &[f32], clusters: &[f32]) -> Vec<Vec<u32>> {
    let ncodes = values.len() / GROUP;
    let dist2 = |ip: usize, ic: usize| {
        let (vp, vc) = (
            &values[ip * GROUP..][..GROUP],
            &clusters[ic * GROUP..][..GROUP],
        );
        (0..GROUP).fold(0f32, |s, k| {
            let d = vp[k] - vc[k];
            fma(d, d, s)
        })
    };
    let nearest = |ip: usize| {
        let (mut best, mut ibest) = (
            vec![f32::INFINITY; num_neighbours],
            vec![usize::MAX; num_neighbours],
        );
        for ic in 0..CLUSTERS {
            let d2 = dist2(ip, ic);
            if let Some(j) = (0..num_neighbours).find(|&j| d2 < best[j]) {
                best.insert(j, d2);
                ibest.insert(j, ic);
                best.pop();
                ibest.pop();
            }
        }
        ibest
    };
    let which = split_map(ncodes, nearest);
    let mut in_cluster = vec![Vec::new(); CLUSTERS];
    for (ip, near) in which.iter().enumerate() {
        for &ic in near {
            in_cluster[ic].push(ip as u32);
        }
    }
    for (ic, points) in in_cluster.iter_mut().enumerate() {
        if !points.is_empty() && points.len().is_multiple_of(8) {
            continue;
        }
        let mut extra = (0..ncodes)
            .filter(|&ip| !which[ip].contains(&ic))
            .map(|ip| (dist2(ip, ic), ip))
            .collect::<Vec<_>>();
        extra.sort_by(|a, b| a.partial_cmp(b).expect("finite distances"));
        let nadd = 8 * points.len().div_ceil(8) - points.len();
        points.extend(extra[..nadd].iter().map(|&(_, ip)| ip as u32));
    }
    in_cluster
}

// `f` over `0..n` on threads of their own (the tables build inside a OnceLock that rayon workers may wait on)
fn split_map<T: Send>(n: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
    let threads = std::thread::available_parallelism().map_or(1, |t| t.get());
    let chunk = n.div_ceil(threads);
    std::thread::scope(|scope| {
        let f = &f;
        let handles = (0..n)
            .step_by(chunk)
            .map(|start| {
                scope.spawn(move || (start..n.min(start + chunk)).map(f).collect::<Vec<_>>())
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("a cluster thread"))
            .collect()
    })
}

fn iq2_kt() -> &'static KtQuantizer {
    static Q: OnceLock<KtQuantizer> = OnceLock::new();
    Q.get_or_init(|| KtQuantizer::new(16, false, IQ2_KT_NEIGHBOURS))
}

fn iq1_kt() -> &'static KtQuantizer {
    static Q: OnceLock<KtQuantizer> = OnceLock::new();
    Q.get_or_init(|| KtQuantizer::new(13, false, IQ1_KT_NEIGHBOURS))
}

fn iq3_kt() -> &'static KtQuantizer {
    static Q: OnceLock<KtQuantizer> = OnceLock::new();
    Q.get_or_init(|| KtQuantizer::new(16, true, IQ3_KT_NEIGHBOURS))
}

// IQ1_KT and IQ2_KT share their row quantizer; they differ in the code width, a second start scale and the packing
#[derive(Clone, Copy, PartialEq)]
enum Signed {
    Iq1,
    Iq2,
}

impl Signed {
    fn quantizer(self) -> &'static KtQuantizer {
        match self {
            Self::Iq1 => iq1_kt(),
            Self::Iq2 => iq2_kt(),
        }
    }

    fn block_bytes(self) -> usize {
        match self {
            Self::Iq1 => BLOCK_IQ1_KT,
            Self::Iq2 => BLOCK_IQ2_KT,
        }
    }

    fn fudge(self) -> f32 {
        match self {
            Self::Iq1 => IQ1_KT_FUDGE,
            Self::Iq2 => 1.0,
        }
    }

    // block `ib`'s 4-bit scale code
    fn scale_code(self, block: &[u8], ib: usize) -> u8 {
        match self {
            Self::Iq1 => block[ib] & 0xf,
            Self::Iq2 => (block[ib % (NBLOCK / 2)] >> (4 * (ib / (NBLOCK / 2)))) & 0xf,
        }
    }

    fn set_scale_code(self, block: &mut [u8], ib: usize, code: u8) {
        match self {
            Self::Iq1 => block[ib] = code,
            Self::Iq2 => block[ib % (NBLOCK / 2)] |= code << (4 * (ib / (NBLOCK / 2))),
        }
    }

    // group `j` of block `ib`'s code: IQ2_KT's a u16 after the scales; IQ1_KT's 13 bits split over ql, qh and the
    // top of its scale byte
    fn put_code(self, block: &mut [u8], ib: usize, j: usize, index: u32) {
        match self {
            Self::Iq2 => {
                block[4 + 2 * (ib * NG + j)..][..2].copy_from_slice(&(index as u16).to_le_bytes())
            }
            Self::Iq1 => {
                block[8 + ib * NG + j] = (index & 0xff) as u8;
                block[40 + (ib % (NBLOCK / 2)) * NG + j] |=
                    (((index >> 8) & 0xf) as u8) << (4 * (ib / (NBLOCK / 2)));
                block[ib] |= ((index >> 12) as u8) << (4 + j);
            }
        }
    }
}

// `best_index_iq4nl` over the IQ4_NL codebook
fn best_index_iq4nl(x: f32) -> usize {
    let v = |i: usize| f32::from(IQ4K_VALUES[i]);
    let ix = x as i32 - i32::from(IQ4K_VALUES[0]);
    if !(0..241).contains(&ix) {
        return if ix < 0 { 0 } else { 15 };
    }
    let ix = IQ4NL_INDEX[ix as usize] as usize;
    if ix < 16 {
        return ix;
    }
    if x - v(ix - 16) < v(ix - 15) - x {
        ix - 16
    } else {
        ix - 15
    }
}

/// `QuantizerIQKT::set_weights`: the importance of each value, from the imatrix where there is one. Inlined into
/// IQ1_KT's and IQ3_KT's row quantizers, g++ vectorizes the super-block's sum of squares (the products in a vector,
/// then added in order, unfused); in IQ2_KT's it stays scalar and fused, as `fused` picks.
fn set_weights(sigma2_scale: f32, x: &[f32], imatrix: Option<&[f32]>, fused: bool) -> Vec<f32> {
    let mut out = vec![0f32; x.len()];
    let blocks = out.as_chunks_mut::<QK_K>().0;
    for (ibl, (xbl, wbl)) in x.as_chunks::<QK_K>().0.iter().zip(blocks).enumerate() {
        let sumx2 = if fused {
            xbl.iter().fold(0f32, |s, v| fma(*v, *v, s))
        } else {
            xbl.iter().fold(0f32, |s, v| s + v * v)
        };
        if sumx2 < KT_EPS2 * QK_K as f32 {
            wbl.fill(KT_WEIGHT);
            continue;
        }
        let sigma2 = sigma2_scale * sumx2 / QK_K as f32;
        match imatrix {
            Some(imatrix) => {
                for ib in 0..NBLOCK {
                    let qw = &imatrix[ibl * QK_K + ib * BLOCK..][..BLOCK];
                    let xb = &xbl[ib * BLOCK..][..BLOCK];
                    let wb = &mut wbl[ib * BLOCK..][..BLOCK];
                    let (mut sumwx, mut sumw2, mut sumx2) = (0f32, 0f32, 0f32);
                    for j in 0..BLOCK {
                        wb[j] = qw[j] * fma(xb[j], xb[j], sigma2).sqrt();
                        sumwx = fma(wb[j], xb[j].abs(), sumwx);
                        sumw2 = fma(wb[j], wb[j], sumw2);
                        sumx2 = fma(xb[j], xb[j], sumx2);
                    }
                    if sumx2 < KT_EPS2 || sumw2 < KT_EPS2 || sumwx < KT_EPS2 {
                        wb.fill(KT_WEIGHT);
                    }
                }
            }
            None => {
                for (w, v) in wbl.iter_mut().zip(xbl) {
                    *w = fma(*v, *v, 0.25 * sigma2);
                }
            }
        }
    }
    out
}

/// One IQ1_KT or IQ2_KT row (`quantize_row_iq1_kt_impl` / `quantize_row_iq2_kt_impl`): per 32 values four codes and
/// a scale at either sign; the scales as IQ4_NL codes of a row scale searched over 19 steps; the codes refound at the
/// coded scales (kept where they fit better), and the row scale refit.
fn quantize_signed_kt_row(kind: Signed, x: &[f32], imatrix: Option<&[f32]>, out: &mut [u8]) {
    let q = kind.quantizer();
    let block_bytes = kind.block_bytes();
    let (header, blocks) = out.split_at_mut(F32_ROW_SCALE);
    header.fill(0);
    blocks.fill(0);
    let nblock = x.len() / QK_K;
    let weights = set_weights(IQ2_KT_SIGMA_SCALE, x, imatrix, kind == Signed::Iq2);
    let amax_row = x.iter().fold(0f32, |m, v| m.max(v.abs()));
    let mut scales = vec![0f32; nblock * NBLOCK];
    let mut all_idx = vec![0u32; x.len() / GROUP];
    let (mut amax_scale, mut max_scale) = (0f32, 0f32);
    for ib in 0..nblock * NBLOCK {
        let xb = &x[ib * BLOCK..][..BLOCK];
        let w = &weights[ib * BLOCK..][..BLOCK];
        let amax = xb.iter().fold(0f32, |m, v| m.max(v.abs()));
        if amax < KT_EPS {
            continue;
        }
        let scale_0 = 90f32.max(124.0 * amax / amax_row);
        let idx_p = q.find_best_match(amax / scale_0, xb, w);
        let (dp, score_p) = q.find_best_scale(xb, w, &idx_p);
        let idx_m = q.find_best_match(-amax / scale_0, xb, w);
        let (dm, score_m) = q.find_best_scale(xb, w, &idx_m);
        let (mut scale, mut idx, score) = if score_p > score_m {
            (dp, idx_p, score_p)
        } else {
            (dm, idx_m, score_m)
        };
        if kind == Signed::Iq1 {
            let scale_1 = scale_0 - IQ1_KT_RETRY_STEP;
            let idx_p1 = q.find_best_match(amax / scale_1, xb, w);
            let (dp1, score_p1) = q.find_best_scale(xb, w, &idx_p1);
            let idx_m1 = q.find_best_match(-amax / scale_1, xb, w);
            let (dm1, score_m1) = q.find_best_scale(xb, w, &idx_m1);
            if score_p1 > score || score_m1 > score {
                (scale, idx) = if score_p1 > score_m1 {
                    (dp1, idx_p1)
                } else {
                    (dm1, idx_m1)
                };
            }
        }
        scales[ib] = scale;
        all_idx[ib * NG..][..NG].copy_from_slice(&idx);
        if scale.abs() > amax_scale {
            (amax_scale, max_scale) = (scale.abs(), scale);
        }
    }
    if max_scale == 0.0 {
        return;
    }
    let v0 = i32::from(IQ4K_VALUES[0]);
    let mut d = max_scale / v0 as f32;
    let mut best = 0f32;
    for itry in -KT_SCALE_TRIES..=KT_SCALE_TRIES {
        let id = (itry + v0) as f32 / max_scale;
        let (mut sumqx, mut sumq2) = (0f32, 0f32);
        for ib in 0..nblock * NBLOCK {
            let dl = f32::from(IQ4K_VALUES[best_index_iq4nl(id * scales[ib])]);
            for ig in 0..NG {
                let code = q.code(all_idx[ib * NG + ig]);
                for j in 0..GROUP {
                    let jj = ib * BLOCK + ig * GROUP + j;
                    let qv = dl * code[j];
                    sumqx = fma(weights[jj] * x[jj], qv, sumqx);
                    sumq2 = fma(weights[jj] * qv, qv, sumq2);
                }
            }
        }
        if sumq2 > 0.0 && sumqx * sumqx > best * sumq2 {
            d = sumqx / sumq2;
            best = d * sumqx;
        }
    }
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    for ibl in 0..nblock {
        let block = &mut blocks[ibl * block_bytes..][..block_bytes];
        for ib in 0..NBLOCK {
            let code = best_index_iq4nl(id * scales[ibl * NBLOCK + ib]) as u8;
            kind.set_scale_code(block, ib, code);
        }
    }
    header.copy_from_slice(&(kind.fudge() * d).to_le_bytes());
    if d == 0.0 {
        return;
    }
    let (mut sumqx, mut sumq2) = (0f32, 0f32);
    for ibl in 0..nblock {
        let block = &mut blocks[ibl * block_bytes..][..block_bytes];
        for ib in 0..NBLOCK {
            let at = ibl * QK_K + ib * BLOCK;
            let (xb, w) = (&x[at..][..BLOCK], &weights[at..][..BLOCK]);
            let ls = i32::from(IQ4K_VALUES[usize::from(kind.scale_code(block, ib))]);
            let dl = d * ls as f32;
            let mut best_idx = q.find_best_match(dl, xb, w);
            let prev = &mut all_idx[at / GROUP..][..NG];
            let (mut mse1, mut mse2) = (0f32, 0f32);
            for ig in 0..NG {
                let (q1, q2) = (q.code(prev[ig]), q.code(best_idx[ig]));
                for j in 0..GROUP {
                    let jj = ig * GROUP + j;
                    let diff1 = fma(-dl, q1[j], xb[jj]);
                    let diff2 = fma(-dl, q2[j], xb[jj]);
                    mse1 = fma(w[jj] * diff1, diff1, mse1);
                    mse2 = fma(w[jj] * diff2, diff2, mse2);
                }
            }
            if mse1 < mse2 {
                best_idx.copy_from_slice(prev);
            } else {
                prev.copy_from_slice(&best_idx);
            }
            for (j, &index) in best_idx.iter().enumerate() {
                kind.put_code(block, ib, j, index);
                let ql = q.code(index);
                for k in 0..GROUP {
                    let qv = ql[k] * ls as f32;
                    let (wl, xl) = (w[GROUP * j + k], xb[GROUP * j + k]);
                    sumqx = fma(wl * xl, qv, sumqx);
                    sumq2 = fma(wl * qv, qv, sumq2);
                }
            }
        }
    }
    if sumq2 > 0.0 {
        header.copy_from_slice(&(sumqx / sumq2 * kind.fudge()).to_le_bytes());
    }
}

/// One IQ3_KT row (`quantize_row_iq3_kt_impl`): magnitudes against the absolute codebook, the signs kept as bits;
/// per 32 values the best of seven start scales; the scales as 4-bit multiples of a row scale searched over 19
/// steps; the codes refound at the coded scales and the row scale refit.
fn quantize_iq3_kt_row(x: &[f32], imatrix: Option<&[f32]>, out: &mut [u8]) {
    let q = iq3_kt();
    let (header, blocks) = out.split_at_mut(F32_ROW_SCALE);
    header.fill(0);
    blocks.fill(0);
    let amax_row = x.iter().fold(0f32, |m, v| m.max(v.abs()));
    if amax_row == 0.0 {
        return;
    }
    let nblock = x.len() / QK_K;
    let weights = set_weights(IQ2_KT_SIGMA_SCALE, x, imatrix, false);
    let mut scales = vec![0f32; nblock * NBLOCK];
    let mut qtmp = vec![0f32; x.len()];
    let (mut amax_scale, mut max_scale) = (0f32, 0f32);
    for ib in 0..nblock * NBLOCK {
        let xaux: [f32; BLOCK] = std::array::from_fn(|j| x[ib * BLOCK + j].abs());
        let w = &weights[ib * BLOCK..][..BLOCK];
        let amax = xaux.iter().fold(0f32, |m, v| m.max(*v));
        if amax < KT_EPS {
            continue;
        }
        let scale_0 = 84f32.max(123.0 * amax / amax_row);
        let (mut best, mut best_idx) = (0f32, None);
        for itry in -IQ3_KT_TRIES..=IQ3_KT_TRIES {
            let idx = q.find_best_match(amax / (scale_0 + IQ3_KT_STEP * itry as f32), &xaux, w);
            let (d, score) = q.find_best_scale(&xaux, w, &idx);
            if score > best {
                best = score;
                scales[ib] = d;
                best_idx = Some(idx);
            }
        }
        let idx = best_idx.expect("a start scale fits the block");
        for (ig, &index) in idx.iter().enumerate() {
            qtmp[ib * BLOCK + ig * GROUP..][..GROUP].copy_from_slice(q.code(index));
        }
        if scales[ib].abs() > amax_scale {
            (amax_scale, max_scale) = (scales[ib].abs(), scales[ib]);
        }
    }
    let mut d = max_scale / 15.0;
    let mut best = 0f32;
    for itry in -KT_SCALE_TRIES..=KT_SCALE_TRIES {
        let id = (itry as f32 * 0.2 + 15.0) / max_scale;
        let (mut sumqx, mut sumq2) = (0f32, 0f32);
        for ib in 0..nblock * NBLOCK {
            let dl = super::iq_quantize::nearest_int(id * scales[ib]).clamp(0, 15) as f32;
            for j in 0..BLOCK {
                let jj = ib * BLOCK + j;
                let qv = dl * qtmp[jj];
                sumqx = fma(weights[jj] * x[jj].abs(), qv, sumqx);
                sumq2 = fma(weights[jj] * qv, qv, sumq2);
            }
        }
        if sumq2 > 0.0 && sumqx * sumqx > best * sumq2 {
            d = sumqx / sumq2;
            best = d * sumqx;
        }
    }
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    let code = |s: f32| super::iq_quantize::nearest_int(id * s).clamp(0, 15) as u8;
    for ibl in 0..nblock {
        for ib in 0..NBLOCK / 2 {
            let s = &scales[ibl * NBLOCK..];
            blocks[ibl * BLOCK_IQ3_KT + ib] = code(s[ib]) | (code(s[ib + NBLOCK / 2]) << 4);
        }
    }
    header.copy_from_slice(&d.to_le_bytes());
    let (mut sumqx, mut sumq2) = (0f32, 0f32);
    for ibl in 0..nblock {
        let block = &mut blocks[ibl * BLOCK_IQ3_KT..][..BLOCK_IQ3_KT];
        for ib in 0..NBLOCK {
            let at = ibl * QK_K + ib * BLOCK;
            let xb = &x[at..][..BLOCK];
            let w = &weights[at..][..BLOCK];
            let xaux: [f32; BLOCK] = std::array::from_fn(|j| xb[j].abs());
            for j in 0..BLOCK {
                if xb[j] < 0.0 {
                    block[68 + j] |= 1 << ib;
                }
            }
            let ls = (block[ib % (NBLOCK / 2)] >> (4 * (ib / (NBLOCK / 2)))) & 0xf;
            let idx = q.find_best_match(d * f32::from(ls), &xaux, w);
            for (j, &index) in idx.iter().enumerate() {
                block[4 + 2 * (ib * NG + j)..][..2].copy_from_slice(&(index as u16).to_le_bytes());
                let ql = q.code(index);
                for k in 0..GROUP {
                    let qv = ql[k] * f32::from(ls);
                    let (wl, xl) = (w[GROUP * j + k], xaux[GROUP * j + k]);
                    sumqx = fma(wl * xl, qv, sumqx);
                    sumq2 = fma(wl * qv, qv, sumq2);
                }
            }
        }
    }
    if sumq2 > 0.0 {
        header.copy_from_slice(&(sumqx / sumq2).to_le_bytes());
    }
}

/// One row of trellis type `ty`; ik passes the imatrix through as it is.
pub(super) fn quantize_row(ty: GgufType, row: &[f32], imatrix: Option<&[f32]>, out: &mut [u8]) {
    match ty {
        GgufType::Iq1Kt => quantize_signed_kt_row(Signed::Iq1, row, imatrix, out),
        GgufType::Iq2Kt => quantize_signed_kt_row(Signed::Iq2, row, imatrix, out),
        GgufType::Iq3Kt => quantize_iq3_kt_row(row, imatrix, out),
        other => unreachable!("{other:?} is not a trellis type"),
    }
}

#[cfg(test)]
mod tests {
    use super::{V, hsum8x8};

    // each lane sums its own vector, so a lane's score is its candidate's
    #[test]
    fn the_eight_way_sum_keeps_each_vector_in_its_lane() {
        let acc: [V; 8] = std::array::from_fn(|i| std::array::from_fn(|k| (i * 10 + k) as f32));
        let sums = hsum8x8(acc);
        for i in 0..8 {
            assert_eq!(sums[i], acc[i].iter().sum::<f32>(), "lane {i}");
        }
    }
}
