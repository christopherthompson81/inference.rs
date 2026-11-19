//! ggml's lattice quantizers for the grid types (`quantize_row_iq2_xxs_impl`, `quantize_row_iq2_xs_impl`,
//! `quantize_row_iq2_s_impl`, `quantize_row_iq3_xxs_impl`, `quantize_row_iq3_s_impl` in ggml-quants.c): each group of
//! 8 (IQ2) or 4 (IQ3) magnitudes snaps to a grid point, found through `iq2xs_init_impl` / `iq3xs_init_impl`'s map
//! from every point of the grid's levels to its grid index or its nearest grid points.
// the loops index as ggml's C does, so each line can be checked against it
#![allow(clippy::needless_range_loop)]

use std::sync::OnceLock;

use super::iq_tables::{
    IQ1S_GRID, IQ2S_GRID, IQ2XS_GRID, IQ2XXS_GRID, IQ3S_GRID, IQ3XXS_GRID, KSIGNS_IQ2XS,
};
use super::kernel::GgufType;
use inference_tensor::quantized::Quantizer;

// ggml's rounding constant, 1.5 * 2^23
const ROUNDING_MAGIC: f32 = 12_582_912.0;
const QK_K: usize = 256;
// every point's key, `sum(level << bits k)`: 8 coordinates of 2 bits, or 4 of 3
const KMAP_SIZE_IQ2: usize = 43692;
const KMAP_SIZE_IQ3: usize = 4096;
const GROUP_MAX_EPS_IQ3_XXS: f32 = 1e-8;
// eight levels per IQ3 coordinate
const KMAX_Q3: i32 = 8;
// ggml stores the IQ3 scales this much above the max-based ones
const IQ3_XXS_SCALE_FUDGE: f32 = 1.0125;
const IQ3_S_SCALE_FUDGE: f32 = 1.033;
pub(super) const BLOCK_IQ3_XXS: usize = 98;
pub(super) const BLOCK_IQ1_S: usize = 50;
pub(super) const BLOCK_IQ1_M: usize = 56;
const GROUP_MAX_EPS_IQ1_S: f32 = 1e-12;
const GROUP_MAX_EPS_IQ1_M: f32 = 1e-7;
// IQ1 values are -1, 0, 1 shifted by + or - this
const IQ1_DELTA: f32 = 0.125;
const IQ1_X_P: [f32; 3] = [-1.0 + IQ1_DELTA, IQ1_DELTA, 1.0 + IQ1_DELTA];
const IQ1_X_M: [f32; 3] = [-1.0 - IQ1_DELTA, -IQ1_DELTA, 1.0 - IQ1_DELTA];
// ggml's fudge factors on the IQ1 block scales
const IQ1_S_SCALE_FUDGE: f32 = 1.125;
const IQ1_M_SCALE_FUDGE: f32 = 1.1125;
// ik_llama.cpp's IQ1_M: its block scale fudge, and the imatrix weights' share of each 32 values' mean square
const IQ1_M_FUDGE_IK: f32 = 1.085;
const IQ1_M_SIGMA_IK: f32 = 1.5;
// IQ1_M's qh bits for each half's shift: none, the second half's, the first's, both
const IQ1_M_SHIFT_MASKS: [u8; 4] = [0x00, 0x80, 0x08, 0x88];
pub(super) const BLOCK_IQ3_S: usize = 110;
const GROUP_MAX_EPS: f32 = 1e-15;
const GROUP_MAX_EPS_IQ2_S: f32 = 1e-8;
// three levels per coordinate, written 2 l + 1
const KMAX_Q: i32 = 3;
// ggml stores IQ2_S's scale this much below the max-based one
const IQ2_S_SCALE_FUDGE: f32 = 0.9875;
// ik_llama.cpp's IQ2_XS block scale fudge, and the grid bytes' step (a level 2l+1 is stored as 8 times it)
const IQ2_XS_FUDGE_IK: f32 = 1.05;
const IQ2_XS_GRID_STEP: f32 = 0.125;

// `a * b + c` as `q`'s build computes it: ik_llama.cpp's (gcc, -march=native) fuses it, ggml's base library rounds the
// product first
#[inline(always)]
fn madd(q: Quantizer, a: f32, b: f32, c: f32) -> f32 {
    match q {
        Quantizer::IkLlama => a.mul_add(b, c),
        Quantizer::LlamaCpp => a * b + c,
    }
}

// `nearest_int(0.5 * (id * x - 1))`, the levels' rounding
fn half_level(q: Quantizer, id: f32, x: f32) -> i32 {
    let inner = madd(q, id, x, -1.0);
    let v = madd(q, 0.5, inner, ROUNDING_MAGIC);
    (v.to_bits() & 0x007f_ffff) as i32 - 0x0040_0000
}
pub(super) const BLOCK_IQ2_XXS: usize = 66;
pub(super) const BLOCK_IQ2_XS: usize = 74;
pub(super) const BLOCK_IQ2_S: usize = 82;

/// A grid's points (coordinates `2 l + 1`), the map from every 2-bit point to its index (`>= 0`) or to its
/// neighbour list (`-(offset + 1)`), and the lists: a count, then that many grid indices.
pub(super) struct GridTables {
    dims: usize,
    bits: usize,
    grid: Vec<[i8; 8]>,
    map: Vec<i32>,
    neighbours: Vec<u16>,
}

// each coordinate's level is its value's rank among the grid's (8 < 25 < 43 for IQ2, -1 < 0 < 1 for IQ1, 4 < ... < 62
// for IQ3_XXS); `points` gives each grid point's `dims` coordinate bytes
fn levels(points: &[[u8; 8]], dims: usize) -> Vec<[u8; 8]> {
    let mut values = points
        .iter()
        .flat_map(|p| p[..dims].iter().map(|&b| b as i8))
        .collect::<Vec<_>>();
    values.sort_unstable();
    values.dedup();
    points
        .iter()
        .map(|p| {
            std::array::from_fn(|i| {
                if i >= dims {
                    return 0;
                }
                values
                    .iter()
                    .position(|&v| v == p[i] as i8)
                    .expect("a grid value") as u8
            })
        })
        .collect()
}

fn build(
    points: &[[u8; 8]],
    dims: usize,
    bits: usize,
    kmap_size: usize,
    nwant: usize,
) -> GridTables {
    let grid = levels(points, dims)
        .into_iter()
        .map(|l| l.map(|l| 2 * l as i8 + 1))
        .collect::<Vec<_>>();
    let mut map = vec![-1i32; kmap_size];
    let mask = (1 << bits) - 1;
    for (k, point) in grid.iter().enumerate() {
        let key = (0..dims)
            .map(|i| (((point[i] - 1) / 2) as usize) << (bits * i))
            .sum::<usize>();
        map[key] = k as i32;
    }
    // each off-grid point's grid points at its `nwant` smallest distances, nearest first, ties by index
    let list = |i: usize| {
        if map[i] >= 0 {
            return Vec::new();
        }
        let pos: [i32; 8] = std::array::from_fn(|k| 2 * ((i >> (bits * k)) & mask) as i32 + 1);
        let mut dist = grid
            .iter()
            .enumerate()
            .map(|(j, pg)| {
                let d2 = (0..dims)
                    .map(|k| (i32::from(pg[k]) - pos[k]).pow(2))
                    .sum::<i32>();
                (d2, j as u16)
            })
            .collect::<Vec<_>>();
        dist.sort_unstable();
        let (mut d2, mut nhave, mut out) = (dist[0].0, 1, Vec::new());
        for &(dj, j) in &dist {
            if dj > d2 {
                if nhave == nwant {
                    break;
                }
                d2 = dj;
                nhave += 1;
            }
            out.push(j);
        }
        out
    };
    // on threads of its own: the first quantizer to need the tables builds them while the others wait on it, so
    // they would starve a build on their rayon pool
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let chunk = kmap_size.div_ceil(threads);
    let lists = std::thread::scope(|scope| {
        let handles = (0..kmap_size)
            .step_by(chunk)
            .map(|start| {
                scope.spawn(move || {
                    (start..kmap_size.min(start + chunk))
                        .map(list)
                        .collect::<Vec<_>>()
                })
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("a neighbour thread"))
            .collect::<Vec<_>>()
    });
    let mut neighbours = Vec::new();
    for (i, list) in lists.into_iter().enumerate() {
        if map[i] >= 0 {
            continue;
        }
        map[i] = -(neighbours.len() as i32 + 1);
        neighbours.push(list.len() as u16);
        neighbours.extend(list);
    }
    GridTables {
        dims,
        bits,
        grid,
        map,
        neighbours,
    }
}

fn iq2(dequant: &[u64], nwant: usize) -> GridTables {
    let points = dequant.iter().map(|g| g.to_le_bytes()).collect::<Vec<_>>();
    build(&points, 8, 2, KMAP_SIZE_IQ2, nwant)
}

fn iq3(dequant: &[u32], nwant: usize) -> GridTables {
    let points = dequant
        .iter()
        .map(|g| {
            let b = g.to_le_bytes();
            [b[0], b[1], b[2], b[3], 0, 0, 0, 0]
        })
        .collect::<Vec<_>>();
    build(&points, 4, 3, KMAP_SIZE_IQ3, nwant)
}

/// The tables `iq2xs_init_impl` builds for `ty`, built once.
pub(super) fn tables(ty: GgufType) -> &'static GridTables {
    static IQ2_XXS: OnceLock<GridTables> = OnceLock::new();
    static IQ2_XS: OnceLock<GridTables> = OnceLock::new();
    static IQ2_S: OnceLock<GridTables> = OnceLock::new();
    static IQ1: OnceLock<GridTables> = OnceLock::new();
    static IQ3_XXS: OnceLock<GridTables> = OnceLock::new();
    static IQ3_S: OnceLock<GridTables> = OnceLock::new();
    match ty {
        GgufType::Iq2Xxs => IQ2_XXS.get_or_init(|| iq2(&IQ2XXS_GRID, 2)),
        GgufType::Iq2Xs => IQ2_XS.get_or_init(|| iq2(&IQ2XS_GRID, 2)),
        GgufType::Iq2S => IQ2_S.get_or_init(|| iq2(&IQ2S_GRID, 1)),
        GgufType::Iq1S | GgufType::Iq1M => IQ1.get_or_init(|| iq2(&IQ1S_GRID, 3)),
        GgufType::Iq3Xxs => IQ3_XXS.get_or_init(|| iq3(&IQ3XXS_GRID, 2)),
        GgufType::Iq3S => IQ3_S.get_or_init(|| iq3(&IQ3S_GRID, 3)),
        other => unreachable!("{other:?} has no lattice tables"),
    }
}

impl GridTables {
    // the levels' grid index, or the best of its neighbours for `xval` at `scale` (whose levels go into `l`)
    fn index(&self, q: Quantizer, l: &mut [i8], xval: &[f32], weight: &[f32], scale: f32) -> usize {
        let entry = self.map[self.key(l)];
        if entry >= 0 {
            return entry as usize;
        }
        let at = (-entry - 1) as usize;
        let count = usize::from(self.neighbours[at]);
        let (mut best_d2, mut best) = (f32::MAX, usize::MAX);
        for &j in &self.neighbours[at + 1..=at + count] {
            let pg = &self.grid[usize::from(j)];
            let mut d2 = 0f32;
            for i in 0..self.dims {
                let diff = madd(q, scale, f32::from(pg[i]), -xval[i]);
                d2 = madd(q, weight[i] * diff, diff, d2);
            }
            if d2 < best_d2 {
                (best_d2, best) = (d2, usize::from(j));
            }
        }
        for (l, &p) in l.iter_mut().zip(&self.grid[best]) {
            *l = (p - 1) / 2;
        }
        best
    }

    // levels already on the grid
    fn on_grid(&self, l: &[i8]) -> usize {
        usize::try_from(self.map[self.key(l)]).expect("the levels lie on the grid")
    }

    fn key(&self, l: &[i8]) -> usize {
        l.iter()
            .enumerate()
            .map(|(i, &l)| (l as usize) << (self.bits * i))
            .sum()
    }

    fn is_on_grid(&self, l: &[i8]) -> bool {
        self.map[self.key(l)] >= 0
    }
}

fn level(q: Quantizer, id: f32, x: f32) -> i8 {
    level_of(q, id, x, KMAX_Q)
}

fn level_of(q: Quantizer, id: f32, x: f32, kmax: i32) -> i8 {
    half_level(q, id, x).clamp(0, kmax - 1) as i8
}

fn fit(q: Quantizer, xval: &[f32], weight: &[f32], l: &[i8]) -> (f32, f32) {
    let (mut sumqx, mut sumq2) = (0f32, 0f32);
    for i in 0..xval.len() {
        let lv = f32::from(2 * l[i] + 1);
        sumqx = madd(q, weight[i] * xval[i], lv, sumqx);
        sumq2 = madd(q, weight[i] * lv, lv, sumq2);
    }
    (sumqx, sumq2)
}

// a group's magnitudes and sign bits; with `even`, the least important value flips so the signs' parity is even
fn magnitudes(xb: &[f32], weight: &[f32], xval: &mut [f32], even: bool) -> u8 {
    let mut s = 0u8;
    let mut nflip = 0u32;
    for i in 0..8 {
        if xb[i] >= 0.0 {
            xval[i] = xb[i];
        } else {
            xval[i] = -xb[i];
            nflip += 1;
            s |= 1 << i;
        }
    }
    if even && !nflip.is_multiple_of(2) {
        let mut imin = 0;
        let mut min = weight[0] * xb[0] * xb[0];
        for i in 1..8 {
            let ax = weight[i] * xb[i] * xb[i];
            if ax < min {
                (min, imin) = (ax, i);
            }
        }
        xval[imin] = -xval[imin];
        s ^= 1 << imin;
    }
    s
}

fn sigma2(xbl: &[f32]) -> f32 {
    xbl.iter().map(|x| x * x).sum::<f32>() / QK_K as f32
}

/// One IQ2_XXS super-block: per 32 values four grid points with 7-bit even-parity signs and a 4-bit scale.
pub(super) fn quantize_iq2_xxs(xbl: &[f32], qw: &[f32], out: &mut [u8], q: Quantizer) {
    let t = tables(GgufType::Iq2Xxs);
    let sigma2 = sigma2(xbl);
    let mut q2 = [0u32; 2 * QK_K / 32];
    let mut scales = [0f32; QK_K / 32];
    let mut max_scale = 0f32;
    let (mut weight, mut waux, mut xval) = ([0f32; 32], [0f32; 32], [0f32; 32]);
    for ib in 0..QK_K / 32 {
        let xb = &xbl[32 * ib..][..32];
        let qw = &qw[32 * ib..][..32];
        for i in 0..32 {
            weight[i] = qw[i] * madd(q, xb[i], xb[i], sigma2).sqrt();
            waux[i] = weight[i].sqrt();
        }
        let mut signs = [0u8; 4];
        for k in 0..4 {
            let at = 8 * k..8 * k + 8;
            signs[k] = magnitudes(&xb[at.clone()], &weight[at.clone()], &mut xval[at], true) & 127;
        }
        let max = xval.iter().copied().fold(xval[0], f32::max);
        if max < GROUP_MAX_EPS {
            continue;
        }
        let mut lu = [0u8; 32];
        let mut scale = inference_tensor::quantized::utils::make_qp_quants(
            32,
            (KMAX_Q + 1) as u8,
            &xval,
            &mut lu,
            &weight,
            q,
        );
        // ggml starts from these levels, kept should no scale step beat them
        let mut l = lu.map(|l| l as i8);
        let eff_max = scale * KMAX_Q as f32;
        // ik_llama.cpp searches on from a non-positive fit too
        if eff_max <= 0.0 && q == Quantizer::LlamaCpp {
            continue;
        }
        let mut best = 0f32;
        for is in -6..=6 {
            let id = madd(q, is as f32, 0.1, (2 * KMAX_Q - 1) as f32) / eff_max;
            let this_scale = 1.0 / id;
            let mut laux = [0i8; 32];
            for k in 0..4 {
                let at = 8 * k..8 * k + 8;
                for i in at.clone() {
                    laux[i] = level(q, id, xval[i]);
                }
                t.index(
                    q,
                    &mut laux[at.clone()],
                    &xval[at.clone()],
                    &waux[at],
                    this_scale,
                );
            }
            let (sumqx, sumq2) = fit(q, &xval, &weight, &laux);
            if sumq2 > 0.0 && sumqx * sumqx > best * sumq2 {
                scale = sumqx / sumq2;
                best = scale * sumqx;
                l = laux;
            }
        }
        if scale > 0.0 {
            let id = 1.0 / scale;
            for k in 0..4 {
                let at = 8 * k..8 * k + 8;
                for i in at.clone() {
                    l[i] = level(q, id, xval[i]);
                }
                let index = t.index(
                    q,
                    &mut l[at.clone()],
                    &xval[at.clone()],
                    &waux[at.clone()],
                    scale,
                );
                for (l, &p) in l[at].iter_mut().zip(&t.grid[index]) {
                    *l = (p - 1) / 2;
                }
            }
            let (sumqx, sumq2) = fit(q, &xval, &weight, &l);
            if sumq2 > 0.0 {
                scale = sumqx / sumq2;
            }
        }
        if scale < 0.0 {
            scale = -scale;
            for s in &mut signs {
                *s = !*s & 127;
            }
        }
        for k in 0..4 {
            let index = t.on_grid(&l[8 * k..8 * k + 8]);
            q2[2 * ib] |= (index as u32) << (8 * k);
            q2[2 * ib + 1] |= u32::from(signs[k]) << (7 * k);
        }
        scales[ib] = scale;
        max_scale = max_scale.max(scale);
    }
    out.fill(0);
    if max_scale == 0.0 {
        return;
    }
    let d = max_scale / 31.0;
    out[..2].copy_from_slice(&half::f16::from_f32(d).to_le_bytes());
    let id = 1.0 / d;
    for ib in 0..QK_K / 32 {
        let l = half_level(q, id, scales[ib]).clamp(0, 15);
        q2[2 * ib + 1] |= (l as u32) << 28;
    }
    for (o, q) in out[2..].as_chunks_mut::<4>().0.iter_mut().zip(q2) {
        o.copy_from_slice(&q.to_le_bytes());
    }
}

/// IQ2_XS (`s_variant` false: even-parity 7-bit signs, an imatrix required) or IQ2_S (8-bit signs, a 1024 grid):
/// per 16 values two grid points and a 4-bit scale.
fn quantize_iq2_16(
    xbl: &[f32],
    qw: Option<&[f32]>,
    s_variant: bool,
    q: Quantizer,
) -> (f32, [u16; 32], [u8; 32], [u8; 8]) {
    let t = tables(if s_variant {
        GgufType::Iq2S
    } else {
        GgufType::Iq2Xs
    });
    let sumx2 = xbl.iter().map(|x| x * x).sum::<f32>();
    let sigma2 = if s_variant {
        2.0 * sumx2 / QK_K as f32
    } else {
        sumx2 / QK_K as f32
    };
    let eps = if s_variant {
        GROUP_MAX_EPS_IQ2_S
    } else {
        GROUP_MAX_EPS
    };
    let mut indices = [0u16; 32];
    let mut signs_out = [0u8; 32];
    let mut scales = [0f32; QK_K / 16];
    let mut max_scale = 0f32;
    let (mut weight, mut waux, mut xval) = ([0f32; 16], [0f32; 16], [0f32; 16]);
    for ib in 0..QK_K / 16 {
        let xb = &xbl[16 * ib..][..16];
        for i in 0..16 {
            weight[i] = match qw {
                Some(qw) => qw[16 * ib + i] * madd(q, xb[i], xb[i], sigma2).sqrt(),
                None => 0.25 * sigma2 + xb[i] * xb[i],
            };
            waux[i] = weight[i].sqrt();
        }
        let mut signs = [0u8; 2];
        for k in 0..2 {
            let at = 8 * k..8 * k + 8;
            let s = magnitudes(
                &xb[at.clone()],
                &weight[at.clone()],
                &mut xval[at],
                !s_variant,
            );
            signs[k] = if s_variant { s } else { s & 127 };
        }
        let max = xval.iter().copied().fold(xval[0], f32::max);
        let mut l = [0i8; 16];
        if max < eps {
            continue;
        }
        let mut best = 0f32;
        let mut scale = max / (2 * KMAX_Q - 1) as f32;
        let mut on_grid = [true; 2];
        for is in -9..=9 {
            let id = madd(q, is as f32, 0.1, (2 * KMAX_Q - 1) as f32) / max;
            let this_scale = 1.0 / id;
            let mut laux = [0i8; 16];
            let mut on_grid_aux = [true; 2];
            for k in 0..2 {
                let at = 8 * k..8 * k + 8;
                for i in at.clone() {
                    laux[i] = level(q, id, xval[i]);
                }
                let key_on = t.is_on_grid(&laux[at.clone()]);
                on_grid_aux[k] = key_on;
                if !key_on {
                    t.index(
                        q,
                        &mut laux[at.clone()],
                        &xval[at.clone()],
                        &waux[at],
                        this_scale,
                    );
                }
            }
            let (sumqx, sumq2) = fit(q, &xval, &weight, &laux);
            if sumq2 > 0.0 && sumqx * sumqx > best * sumq2 {
                scale = sumqx / sumq2;
                best = scale * sumqx;
                l = laux;
                on_grid = on_grid_aux;
            }
        }
        if on_grid.iter().any(|&g| !g) && scale > 0.0 {
            let id = 1.0 / scale;
            for k in 0..2 {
                if on_grid[k] {
                    continue;
                }
                let at = 8 * k..8 * k + 8;
                for i in at.clone() {
                    l[i] = level(q, id, xval[i]);
                }
                t.index(q, &mut l[at.clone()], &xval[at.clone()], &waux[at], scale);
            }
            let (sumqx, sumq2) = fit(q, &xval, &weight, &l);
            if sumq2 > 0.0 {
                scale = sumqx / sumq2;
            }
        }
        if scale < 0.0 {
            scale = -scale;
            for s in &mut signs {
                *s = if s_variant { !*s } else { !*s & 127 };
            }
        }
        for k in 0..2 {
            indices[2 * ib + k] = t.on_grid(&l[8 * k..8 * k + 8]) as u16;
            signs_out[2 * ib + k] = signs[k];
        }
        scales[ib] = scale;
        max_scale = max_scale.max(scale);
    }
    let mut packed = [0u8; 8];
    if max_scale == 0.0 {
        return (0.0, [0; 32], [0; 32], packed);
    }
    let d = max_scale / 31.0;
    let id = 1.0 / d;
    for ib in 0..QK_K / 16 {
        let l = half_level(q, id, scales[ib]).clamp(0, 15) as u8;
        packed[ib / 2] |= l << (4 * (ib % 2));
    }
    (d, indices, signs_out, packed)
}

/// One IQ2_XS super-block: `d`, 32 words of a 9-bit grid index under 7 sign bits, eight scale bytes.
pub(super) fn quantize_iq2_xs(xbl: &[f32], qw: &[f32], out: &mut [u8], q: Quantizer) {
    let (d, indices, signs, scales) = quantize_iq2_16(xbl, Some(qw), false, q);
    out.fill(0);
    if d == 0.0 {
        return;
    }
    out[..2].copy_from_slice(&half::f16::from_f32(d).to_le_bytes());
    for (o, (&i, &s)) in out[2..66]
        .as_chunks_mut::<2>()
        .0
        .iter_mut()
        .zip(indices.iter().zip(&signs))
    {
        o.copy_from_slice(&(i | (u16::from(s) << 9)).to_le_bytes());
    }
    out[66..74].copy_from_slice(&scales);
}

/// ik_llama.cpp's IQ2_XS (`quantize_row_iq2_xs_impl`): each 16 values start from the best split of their sorted
/// magnitudes into the levels 1, 3, 5, try 15 scales around it, then snap to the grid again at the fitted scale
/// while the points change and the fit improves; the block scale is refit last over every value as stored.
pub(super) fn quantize_iq2_xs_ik(xbl: &[f32], qw: &[f32], out: &mut [u8]) {
    let q = Quantizer::IkLlama;
    let t = tables(GgufType::Iq2Xs);
    let sigma2 = xbl.iter().map(|x| x * x).sum::<f32>() / QK_K as f32;
    let weights = |ib: usize, weight: &mut [f32; 16]| {
        let xb = &xbl[16 * ib..][..16];
        for i in 0..16 {
            weight[i] = qw[16 * ib + i] * madd(q, xb[i], xb[i], sigma2).sqrt();
        }
    };
    let mut q2 = [0u16; QK_K / 8];
    let mut scales = [0f32; QK_K / 16];
    let mut max_scale = 0f32;
    let (mut weight, mut waux, mut xval) = ([0f32; 16], [0f32; 16], [0f32; 16]);
    for ib in 0..QK_K / 16 {
        let xb = &xbl[16 * ib..][..16];
        weights(ib, &mut weight);
        for i in 0..16 {
            waux[i] = weight[i].sqrt();
        }
        let mut signs = [0u8; 2];
        for k in 0..2 {
            let at = 8 * k..8 * k + 8;
            signs[k] = magnitudes(&xb[at.clone()], &weight[at.clone()], &mut xval[at], true) & 127;
        }
        let max = xval.iter().copied().fold(xval[0], f32::max);
        if max < GROUP_MAX_EPS {
            continue;
        }
        let (_, sumx, sumw) = sorted_sums(q, &xval, &weight);
        let (mut best, mut scale) = (0f32, 0f32);
        for i1 in 0..=16 {
            for i2 in i1..=16 {
                let sumqx = madd(
                    q,
                    sumx[16] - sumx[i2],
                    5.0,
                    madd(q, sumx[i2] - sumx[i1], 3.0, sumx[i1] - sumx[0]),
                );
                let sumq2 = madd(
                    q,
                    sumw[16] - sumw[i2],
                    25.0,
                    madd(q, sumw[i2] - sumw[i1], 9.0, sumw[i1] - sumw[0]),
                );
                if sumq2 > 0.0 && sumqx * sumqx > best * sumq2 {
                    scale = sumqx / sumq2;
                    best = scale * sumqx;
                }
            }
        }
        best = 0.0;
        let eff_max = scale * (2 * KMAX_Q - 1) as f32;
        // the levels at `id`, each group snapped to the grid (its nearest neighbour at `at_scale` when off it)
        let snap = |id: f32, at_scale: f32, laux: &mut [i8; 16]| -> [usize; 2] {
            std::array::from_fn(|k| {
                let at = 8 * k..8 * k + 8;
                for i in at.clone() {
                    laux[i] = level(q, id, xval[i]);
                }
                t.index(
                    q,
                    &mut laux[at.clone()],
                    &xval[at.clone()],
                    &waux[at],
                    at_scale,
                )
            })
        };
        let mut l = [0i8; 16];
        let mut index = [0usize; 2];
        for is in -7..=7 {
            let id = madd(q, is as f32, 0.1, (2 * KMAX_Q - 1) as f32) / eff_max;
            let mut laux = [0i8; 16];
            let aux = snap(id, 1.0 / id, &mut laux);
            let (sumqx, sumq2) = fit(q, &xval, &weight, &laux);
            if sumq2 > 0.0 && sumqx * sumqx > best * sumq2 {
                scale = sumqx / sumq2;
                best = scale * sumqx;
                (l, index) = (laux, aux);
            }
        }
        if scale != 0.0 {
            for _ in 0..3 {
                let mut laux = [0i8; 16];
                let aux = snap(1.0 / scale, scale, &mut laux);
                if aux == index {
                    break;
                }
                let (sumqx, sumq2) = fit(q, &xval, &weight, &laux);
                if sumq2 > 0.0 && sumqx * sumqx > best * sumq2 {
                    scale = sumqx / sumq2;
                    best = scale * sumqx;
                    (l, index) = (laux, aux);
                } else {
                    break;
                }
            }
        }
        if scale < 0.0 {
            scale = -scale;
            for s in &mut signs {
                *s = !*s & 127;
            }
        }
        for k in 0..2 {
            q2[2 * ib + k] = t.on_grid(&l[8 * k..8 * k + 8]) as u16 | (u16::from(signs[k]) << 9);
        }
        scales[ib] = scale;
        max_scale = max_scale.max(scale);
    }
    out.fill(0);
    if max_scale == 0.0 {
        return;
    }
    let mut d = max_scale / 31.0;
    let id = 1.0 / d;
    let (mut sumqx, mut sumq2) = (0f32, 0f32);
    for ib in 0..QK_K / 16 {
        let l = half_level(q, id, scales[ib]).clamp(0, 15);
        out[66 + ib / 2] |= (l as u8) << (4 * (ib % 2));
        let lv = (2 * l + 1) as f32;
        let xb = &xbl[16 * ib..][..16];
        weights(ib, &mut weight);
        for k in 0..2 {
            let code = q2[2 * ib + k];
            let grid = IQ2XS_GRID[usize::from(code & 511)].to_le_bytes();
            let signs = KSIGNS_IQ2XS[usize::from(code >> 9)];
            for j in 0..8 {
                let sign = if signs & (1 << j) != 0 { -1.0 } else { 1.0 };
                let v = IQ2_XS_GRID_STEP * lv * f32::from(grid[j] as i8) * sign;
                let w = weight[8 * k + j];
                sumqx = madd(q, w * v, xb[8 * k + j], sumqx);
                sumq2 = madd(q, w * v, v, sumq2);
            }
        }
    }
    if sumq2 > 0.0 {
        d = IQ2_XS_FUDGE_IK * sumqx / sumq2;
    } else {
        d *= IQ2_XS_FUDGE_IK;
    }
    out[..2].copy_from_slice(&half::f16::from_f32(d).to_le_bytes());
    for (o, w) in out[2..66].as_chunks_mut::<2>().0.iter_mut().zip(q2) {
        o.copy_from_slice(&w.to_le_bytes());
    }
}

/// One IQ2_S super-block: `d`, 32 grid index low bytes, 32 sign bytes, eight bytes of the indices' top bits, eight
/// scale bytes.
pub(super) fn quantize_iq2_s(xbl: &[f32], qw: Option<&[f32]>, out: &mut [u8], q: Quantizer) {
    let (d, indices, signs, scales) = quantize_iq2_16(xbl, qw, true, q);
    out.fill(0);
    if d == 0.0 {
        return;
    }
    out[..2].copy_from_slice(&half::f16::from_f32(d * IQ2_S_SCALE_FUDGE).to_le_bytes());
    for (i8, &index) in indices.iter().enumerate() {
        out[2 + i8] = (index & 255) as u8;
        out[66 + i8 / 4] |= ((index >> 8) as u8) << (2 * (i8 % 4));
        out[34 + i8] = signs[i8];
    }
    out[74..82].copy_from_slice(&scales);
}

/// IQ3_XXS (`s_variant` false: even-parity 7-bit signs, a 256 grid) or IQ3_S (8-bit signs, a 512 grid): per 32
/// values eight 4-value grid points and a 4-bit scale. Returns `d` (0 for an all-zero block), the grid indices and
/// the sign bytes (four per 32 values) and the 4-bit scale codes.
fn quantize_iq3(
    xbl: &[f32],
    qw: Option<&[f32]>,
    s_variant: bool,
    q: Quantizer,
) -> (f32, [u16; 64], [u8; 32], [u8; 8]) {
    let t = tables(if s_variant {
        GgufType::Iq3S
    } else {
        GgufType::Iq3Xxs
    });
    let sigma2 = 2.0 * xbl.iter().map(|x| x * x).sum::<f32>() / QK_K as f32;
    let mut indices = [0u16; 64];
    let mut signs_out = [0u8; 32];
    let mut scales = [0f32; QK_K / 32];
    let mut max_scale = 0f32;
    let mut on_grid = [!s_variant; 8];
    let (mut weight, mut waux, mut xval) = ([0f32; 32], [0f32; 32], [0f32; 32]);
    for ib in 0..QK_K / 32 {
        let xb = &xbl[32 * ib..][..32];
        for i in 0..32 {
            weight[i] = match qw {
                Some(qw) => qw[32 * ib + i] * madd(q, xb[i], xb[i], sigma2).sqrt(),
                None => xb[i] * xb[i],
            };
            waux[i] = weight[i].sqrt();
        }
        let mut signs = [0u8; 4];
        for k in 0..4 {
            let at = 8 * k..8 * k + 8;
            let s = magnitudes(
                &xb[at.clone()],
                &weight[at.clone()],
                &mut xval[at],
                !s_variant,
            );
            signs[k] = if s_variant { s } else { s & 127 };
        }
        let max = xval.iter().copied().fold(xval[0], f32::max);
        let mut l = [0i8; 32];
        let empty = if s_variant {
            max == 0.0
        } else {
            max < GROUP_MAX_EPS_IQ3_XXS
        };
        if empty {
            continue;
        }
        let mut best = 0f32;
        let mut scale = max / (2 * KMAX_Q3 - 1) as f32;
        // ik_llama.cpp's IQ3_XXS keeps the last block's flags until a scale try replaces them
        if q == Quantizer::LlamaCpp || s_variant {
            on_grid = [!s_variant; 8];
        }
        let steps = if s_variant { 9 } else { 15 };
        for is in -steps..=steps {
            let id = madd(q, is as f32, 0.2, (2 * KMAX_Q3 - 1) as f32) / max;
            let this_scale = 1.0 / id;
            let mut laux = [0i8; 32];
            let mut on_grid_aux = [true; 8];
            for k in 0..8 {
                let at = 4 * k..4 * k + 4;
                for i in at.clone() {
                    laux[i] = level_of(q, id, xval[i], KMAX_Q3);
                }
                on_grid_aux[k] = t.is_on_grid(&laux[at.clone()]);
                if !on_grid_aux[k] {
                    t.index(
                        q,
                        &mut laux[at.clone()],
                        &xval[at.clone()],
                        &waux[at],
                        this_scale,
                    );
                }
            }
            let (sumqx, sumq2) = fit(q, &xval, &weight, &laux);
            if sumq2 > 0.0 && sumqx * sumqx > best * sumq2 {
                scale = sumqx / sumq2;
                best = scale * sumqx;
                l = laux;
                on_grid = on_grid_aux;
            }
        }
        if on_grid.iter().any(|&g| !g) && scale > 0.0 {
            let id = 1.0 / scale;
            for k in 0..8 {
                // IQ3_S revisits every group, IQ3_XXS only those off the grid
                if on_grid[k] && !s_variant {
                    continue;
                }
                let at = 4 * k..4 * k + 4;
                let mut lk = [0i8; 4];
                for (i, lk) in at.clone().zip(&mut lk) {
                    *lk = level_of(q, id, xval[i], KMAX_Q3);
                }
                let index = t.index(q, &mut lk, &xval[at.clone()], &waux[at.clone()], scale);
                for (l, &p) in l[at].iter_mut().zip(&t.grid[index]) {
                    *l = (p - 1) / 2;
                }
            }
            let (sumqx, sumq2) = fit(q, &xval, &weight, &l);
            if sumq2 > 0.0 {
                scale = sumqx / sumq2;
            }
        }
        if scale < 0.0 {
            scale = -scale;
            for s in &mut signs {
                *s = if s_variant { !*s } else { !*s & 127 };
            }
        }
        for k in 0..8 {
            indices[8 * ib + k] = t.on_grid(&l[4 * k..4 * k + 4]) as u16;
        }
        signs_out[4 * ib..4 * ib + 4].copy_from_slice(&signs);
        scales[ib] = scale;
        max_scale = max_scale.max(scale);
    }
    let mut codes = [0u8; 8];
    if max_scale == 0.0 {
        return (0.0, [0; 64], [0; 32], codes);
    }
    let d = max_scale / 31.0;
    let id = 1.0 / d;
    for ib in 0..QK_K / 32 {
        codes[ib] = half_level(q, id, scales[ib]).clamp(0, 15) as u8;
    }
    (d, indices, signs_out, codes)
}

/// One IQ3_XXS super-block: `d`, 64 grid index bytes, then per 32 values a word of four 7-bit sign codes under a
/// 4-bit scale.
pub(super) fn quantize_iq3_xxs(xbl: &[f32], qw: Option<&[f32]>, out: &mut [u8], q: Quantizer) {
    let (d, indices, signs, codes) = quantize_iq3(xbl, qw, false, q);
    out.fill(0);
    if d == 0.0 {
        return;
    }
    out[..2].copy_from_slice(&half::f16::from_f32(d * IQ3_XXS_SCALE_FUDGE).to_le_bytes());
    for (o, &i) in out[2..66].iter_mut().zip(&indices) {
        *o = i as u8;
    }
    for ib in 0..QK_K / 32 {
        let s = &signs[4 * ib..4 * ib + 4];
        let word = u32::from(s[0])
            | (u32::from(s[1]) << 7)
            | (u32::from(s[2]) << 14)
            | (u32::from(s[3]) << 21)
            | (u32::from(codes[ib]) << 28);
        out[66 + 4 * ib..70 + 4 * ib].copy_from_slice(&word.to_le_bytes());
    }
}

/// One IQ3_S super-block: `d`, 64 grid index low bytes, eight bytes of their ninth bits, 32 sign bytes, four bytes of
/// paired 4-bit scales.
pub(super) fn quantize_iq3_s(xbl: &[f32], qw: Option<&[f32]>, out: &mut [u8], q: Quantizer) {
    let (d, indices, signs, codes) = quantize_iq3(xbl, qw, true, q);
    out.fill(0);
    if d == 0.0 {
        return;
    }
    out[..2].copy_from_slice(&half::f16::from_f32(d * IQ3_S_SCALE_FUDGE).to_le_bytes());
    for (j, &index) in indices.iter().enumerate() {
        out[2 + j] = (index & 255) as u8;
        out[66 + j / 8] |= ((index >> 8) as u8) << (j % 8);
    }
    out[74..106].copy_from_slice(&signs);
    for ib in (0..QK_K / 32).step_by(2) {
        out[106 + ib / 2] = codes[ib] | (codes[ib + 1] << 4);
    }
}

impl GridTables {
    // `iq1_find_best_neighbour2`: the neighbour of the levels in `l` nearest `xval` at `scale` over the shifted values
    // `xx` (its fallback over the whole grid never runs, as a finite distance always beats its start)
    fn iq1_index(
        &self,
        q: Quantizer,
        l: &mut [i8],
        xval: &[f32],
        weight: &[f32],
        scale: f32,
        xx: &[f32; 3],
    ) -> usize {
        let entry = self.map[self.key(l)];
        if entry >= 0 {
            return entry as usize;
        }
        let at = (-entry - 1) as usize;
        let count = usize::from(self.neighbours[at]);
        let (mut best_d2, mut best) = (f32::MAX, usize::MAX);
        for &j in &self.neighbours[at + 1..=at + count] {
            let pg = &self.grid[usize::from(j)];
            let mut d2 = 0f32;
            for i in 0..8 {
                let diff = madd(q, scale, xx[((pg[i] - 1) / 2) as usize], -xval[i]);
                d2 = madd(q, weight[i] * diff, diff, d2);
            }
            if d2 < best_d2 {
                (best_d2, best) = (d2, usize::from(j));
            }
        }
        for (l, &p) in l.iter_mut().zip(&self.grid[best]) {
            *l = (p - 1) / 2;
        }
        best
    }

    // the scale fitting `xb` best over the chosen grid points, each group under its shifted values
    fn iq1_refit(
        &self,
        q: Quantizer,
        xb: &[f32],
        weight: &[f32],
        indices: &[usize],
        xx: impl Fn(usize) -> &'static [f32; 3],
    ) -> (f32, f32) {
        let (mut sumqx, mut sumq2) = (0f32, 0f32);
        for (k, &index) in indices.iter().enumerate() {
            let pg = &self.grid[index];
            for j in 0..8 {
                let w = weight[8 * k + j];
                let v = xx(k)[((pg[j] - 1) / 2) as usize];
                sumqx = madd(q, w * v, xb[8 * k + j], sumqx);
                sumq2 = madd(q, w * v, v, sumq2);
            }
        }
        (sumqx, sumq2)
    }
}

// the block's values in ascending order (a stable sort, as glibc's qsort of these pairs is) and the running sums of
// weight times value and of weight over them
fn sorted_sums(q: Quantizer, xb: &[f32], weight: &[f32]) -> (Vec<usize>, Vec<f32>, Vec<f32>) {
    let mut idx = (0..xb.len()).collect::<Vec<_>>();
    idx.sort_by(|&a, &b| xb[a].partial_cmp(&xb[b]).expect("finite weights"));
    let (mut sumx, mut sumw) = (vec![0f32; xb.len() + 1], vec![0f32; xb.len() + 1]);
    for (j, &i) in idx.iter().enumerate() {
        sumx[j + 1] = madd(q, weight[i], xb[i], sumx[j]);
        sumw[j + 1] = sumw[j] + weight[i];
    }
    (idx, sumx, sumw)
}

// levels 0, 1, 2 for the sorted values below `i1`, from `i1` to `i2`, and from `i2`
fn split_levels(idx: &[usize], i1: usize, i2: usize, l: &mut [i8]) {
    for (j, &i) in idx.iter().enumerate() {
        l[i] = if j < i1 {
            0
        } else if j < i2 {
            1
        } else {
            2
        };
    }
}

/// One IQ1_S super-block: `d`, per 32 values four 11-bit grid indices (low bytes in `qs`, top bits in `qh`) with a
/// 3-bit scale and the shift's sign in `qh`'s top nibble.
pub(super) fn quantize_iq1_s(xbl: &[f32], qw: &[f32], out: &mut [u8], q: Quantizer) {
    const BLOCK: usize = 32;
    let t = tables(GgufType::Iq1S);
    let sigma2 = 2.0 * xbl.iter().map(|x| x * x).sum::<f32>() / QK_K as f32;
    out.fill(0);
    let mut scales = [0f32; QK_K / BLOCK];
    let mut shifts = [1i8; QK_K / BLOCK];
    let mut qh = [0u16; QK_K / BLOCK];
    let mut max_scale = 0f32;
    let mut weight = [0f32; BLOCK];
    for ib in 0..QK_K / BLOCK {
        let xb = &xbl[BLOCK * ib..][..BLOCK];
        for i in 0..BLOCK {
            weight[i] = qw[BLOCK * ib + i] * madd(q, xb[i], xb[i], sigma2).sqrt();
        }
        let max = xb.iter().map(|x| x.abs()).fold(xb[0].abs(), f32::max);
        if max < GROUP_MAX_EPS_IQ1_S {
            continue;
        }
        let (idx, sumx, sumw) = sorted_sums(q, xb, &weight);
        let (mut best_score, mut scale) = (-f32::MAX, max);
        let (mut besti1, mut besti2, mut best_shift) = (usize::MAX, usize::MAX, 0i8);
        for i1 in 0..=BLOCK {
            for i2 in i1..=BLOCK {
                for (shift, xs) in [(1i8, &IQ1_X_P), (-1, &IQ1_X_M)] {
                    let (a, b, c) = (
                        sumx[i1] - sumx[0],
                        sumx[i2] - sumx[i1],
                        sumx[BLOCK] - sumx[i2],
                    );
                    let sumqx = madd(q, c, xs[2], madd(q, b, xs[1], a * xs[0]));
                    let (a, b, c) = (
                        sumw[i1] - sumw[0],
                        sumw[i2] - sumw[i1],
                        sumw[BLOCK] - sumw[i2],
                    );
                    let sumq2 = madd(
                        q,
                        c * xs[2],
                        xs[2],
                        madd(q, b * xs[1], xs[1], a * xs[0] * xs[0]),
                    );
                    if sumq2 > 0.0 && sumqx * sumqx > best_score * sumq2 {
                        scale = sumqx / sumq2;
                        best_score = scale * sumqx;
                        (besti1, besti2, best_shift) = (i1, i2, shift);
                    }
                }
            }
        }
        if besti1 == usize::MAX || best_shift == 0 {
            continue;
        }
        let mut l = [0i8; BLOCK];
        split_levels(&idx, besti1, besti2, &mut l);
        if scale < 0.0 {
            for l in &mut l {
                *l = 2 - *l;
            }
            scale = -scale;
            best_shift = -best_shift;
        }
        let xx: &'static [f32; 3] = if best_shift == 1 { &IQ1_X_P } else { &IQ1_X_M };
        let mut indices = [0usize; BLOCK / 8];
        let mut all_on_grid = true;
        for k in 0..BLOCK / 8 {
            let at = 8 * k..8 * k + 8;
            all_on_grid &= t.is_on_grid(&l[at.clone()]);
            indices[k] = t.iq1_index(
                q,
                &mut l[at.clone()],
                &xb[at.clone()],
                &weight[at],
                scale,
                xx,
            );
        }
        if !all_on_grid {
            let (sumqx, sumq2) = t.iq1_refit(q, xb, &weight, &indices, |_| xx);
            if sumqx > 0.0 && sumq2 > 0.0 {
                scale = sumqx / sumq2;
            }
        }
        let mut h = 0u16;
        for (k, &index) in indices.iter().enumerate() {
            out[2 + 4 * ib + k] = (index & 255) as u8;
            h |= ((index >> 8) as u16) << (3 * k);
        }
        qh[ib] = h;
        scales[ib] = scale;
        shifts[ib] = best_shift;
        max_scale = max_scale.max(scale);
    }
    if max_scale == 0.0 {
        for (o, h) in out[34..50].as_chunks_mut::<2>().0.iter_mut().zip(qh) {
            o.copy_from_slice(&h.to_le_bytes());
        }
        return;
    }
    let d = max_scale / 15.0;
    out[..2].copy_from_slice(&half::f16::from_f32(d * IQ1_S_SCALE_FUDGE).to_le_bytes());
    let id = 1.0 / d;
    for ib in 0..QK_K / BLOCK {
        let mut l = half_level(q, id, scales[ib]).clamp(0, 7) as u16;
        if shifts[ib] == -1 {
            l |= 8;
        }
        qh[ib] |= l << 12;
    }
    for (o, h) in out[34..50].as_chunks_mut::<2>().0.iter_mut().zip(qh) {
        o.copy_from_slice(&h.to_le_bytes());
    }
}

// `iq1m_process_1block`: a 16-value block's best split of its sorted values into the levels 0, 1, 2 under each of the
// four shift pairs, snapped to the grid, then (when a group left it) every shift pair refit. Returns (scale, shift
// choice, grid indices).
fn iq1_m_block_ik(t: &GridTables, xb: &[f32], weight: &[f32]) -> (f32, usize, [usize; 2]) {
    const BLOCK: usize = 16;
    let q = Quantizer::IkLlama;
    let mut idx = (0..BLOCK).collect::<Vec<_>>();
    idx.sort_by(|&a, &b| xb[a].partial_cmp(&xb[b]).expect("finite values"));
    // running sums per half, each over the sorted values in it
    let (mut sw, mut sx) = ([[0f32; BLOCK + 1]; 2], [[0f32; BLOCK + 1]; 2]);
    for (j, &i) in idx.iter().enumerate() {
        let h = usize::from(i >= BLOCK / 2);
        // both halves' branches take `weight[i] * xb[i]`, which gcc computes once before them, so it is added
        sw[h][j + 1] = sw[h][j] + weight[i];
        sx[h][j + 1] = sx[h][j] + weight[i] * xb[i];
        sw[1 - h][j + 1] = sw[1 - h][j];
        sx[1 - h][j + 1] = sx[1 - h][j];
    }
    let values = |k: usize, h: usize| iq1_m_values(k, h);
    let (mut best_score, mut scale) = (0f32, 0f32);
    let (mut besti1, mut besti2, mut best_k) = (0usize, 0usize, None);
    for i1 in 0..=BLOCK {
        for i2 in i1..=BLOCK {
            for k in 0..4 {
                let (mut sumqx, mut sumq2) = (0f32, 0f32);
                for h in 0..2 {
                    let xx = values(k, h);
                    let parts = [
                        (sx[h][i1] - sx[h][0], sw[h][i1] - sw[h][0]),
                        (sx[h][i2] - sx[h][i1], sw[h][i2] - sw[h][i1]),
                        (sx[h][BLOCK] - sx[h][i2], sw[h][BLOCK] - sw[h][i2]),
                    ];
                    // as gcc builds it: each half's first term is hoisted out of the loop (added, not fused), and
                    // the second half's x terms are shared between two shift choices (added, not fused)
                    for (n, &(x, w)) in parts.iter().enumerate() {
                        let (tx, t2) = (x * xx[n], w * xx[n] * xx[n]);
                        if h == 0 && n == 0 {
                            (sumqx, sumq2) = (tx, t2);
                        } else if n == 0 {
                            (sumqx, sumq2) = (sumqx + tx, sumq2 + t2);
                        } else {
                            sumqx = if h == 0 {
                                madd(q, x, xx[n], sumqx)
                            } else {
                                sumqx + tx
                            };
                            sumq2 = madd(q, w * xx[n], xx[n], sumq2);
                        }
                    }
                }
                if sumq2 > 0.0 && sumqx * sumqx > best_score * sumq2 {
                    scale = sumqx / sumq2;
                    best_score = scale * sumqx;
                    (besti1, besti2, best_k) = (i1, i2, Some(k));
                }
            }
        }
    }
    let mut best_k = best_k.expect("a split with a positive fit");
    let mut l = [0i8; BLOCK];
    split_levels(&idx, besti1, besti2, &mut l);
    if scale < 0.0 {
        for l in &mut l {
            *l = 2 - *l;
        }
        scale = -scale;
        best_k = 3 - best_k;
    }
    let mut all_on_grid = true;
    let indices: [usize; 2] = std::array::from_fn(|h| {
        let at = 8 * h..8 * h + 8;
        all_on_grid &= t.is_on_grid(&l[at.clone()]);
        t.iq1_index(
            q,
            &mut l[at.clone()],
            &xb[at.clone()],
            &weight[at],
            scale,
            values(best_k, h),
        )
    });
    if !all_on_grid {
        let (mut sumqx, mut sumq2) = ([0f32; 4], [0f32; 4]);
        for j in 0..BLOCK {
            let w = weight[j];
            let (qp, qm) = (IQ1_X_P[l[j] as usize], IQ1_X_M[l[j] as usize]);
            // k 1 takes the plus values in the first half, k 2 in the second
            let (k_first, k_second) = if j < BLOCK / 2 { (1, 2) } else { (2, 1) };
            for (k, v) in [(0, qp), (3, qm), (k_first, qp), (k_second, qm)] {
                sumqx[k] = madd(q, w * xb[j], v, sumqx[k]);
                sumq2[k] = madd(q, w * v, v, sumq2[k]);
            }
        }
        let mut best_score = 0f32;
        for k in 0..4 {
            if sumqx[k] > 0.0 && sumq2[k] > 0.0 && sumqx[k] * sumqx[k] > best_score * sumq2[k] {
                scale = sumqx[k] / sumq2[k];
                best_score = scale * sumqx[k];
                best_k = k;
            }
        }
    }
    (scale, best_k, indices)
}

/// ik_llama.cpp's IQ1_M (`quantize_row_iq1_m_impl`): imatrix weights over each 32 values' own spread, each 16 values
/// searched by [`iq1_m_block_ik`], the block scale refit over every value as stored.
pub(super) fn quantize_iq1_m_ik(xbl: &[f32], qw: Option<&[f32]>, out: &mut [u8]) {
    const BLOCK: usize = 16;
    let q = Quantizer::IkLlama;
    let t = tables(GgufType::Iq1M);
    out.fill(0);
    let sigma2: [f32; QK_K / 32] = std::array::from_fn(|ib| {
        IQ1_M_SIGMA_IK
            * xbl[32 * ib..32 * ib + 32]
                .iter()
                .map(|x| x * x)
                .sum::<f32>()
            / 32.0
    });
    let block_weight = |ib: usize, weight: &mut [f32; BLOCK]| {
        let xb = &xbl[BLOCK * ib..][..BLOCK];
        for i in 0..BLOCK {
            weight[i] = match qw {
                Some(qw) => qw[BLOCK * ib + i] * madd(q, xb[i], xb[i], sigma2[ib / 2]).sqrt(),
                None => xb[i] * xb[i],
            };
        }
    };
    let (qs, rest) = out.split_at_mut(32);
    let (qh, sc_bytes) = rest.split_at_mut(16);
    let mut scales = [0f32; QK_K / BLOCK];
    let mut shifts = [0usize; QK_K / BLOCK];
    let mut max_scale = 0f32;
    let mut weight = [0f32; BLOCK];
    for ib in 0..QK_K / BLOCK {
        let xb = &xbl[BLOCK * ib..][..BLOCK];
        block_weight(ib, &mut weight);
        let mut max = xb[0].abs();
        // ik_llama.cpp's sum leaves out the first value
        let mut sumwx = 0f32;
        for i in 1..BLOCK {
            max = max.max(xb[i].abs());
            sumwx = madd(q, weight[i], xb[i].abs(), sumwx);
        }
        if max < GROUP_MAX_EPS_IQ1_M {
            continue;
        }
        if sumwx == 0.0 {
            for i in 0..BLOCK {
                weight[i] = xb[i] * xb[i];
            }
        }
        let (scale, shift, indices) = iq1_m_block_ik(t, xb, &weight);
        qs[2 * ib] = (indices[0] & 255) as u8;
        qs[2 * ib + 1] = (indices[1] & 255) as u8;
        qh[ib] = ((indices[0] >> 8) | ((indices[1] >> 8) << 4)) as u8;
        scales[ib] = scale;
        shifts[ib] = shift;
        max_scale = max_scale.max(scale);
    }
    if max_scale == 0.0 {
        return;
    }
    let mut sc = [0u16; 4];
    let mut d = max_scale / 15.0;
    let id = 1.0 / d;
    let (mut sumqx, mut sumq2) = (0f32, 0f32);
    for ib in 0..QK_K / BLOCK {
        let l = half_level(q, id, scales[ib]).clamp(0, 7);
        sc[ib / 4] |= (l as u16) << (3 * (ib % 4));
        qh[ib] |= IQ1_M_SHIFT_MASKS[shifts[ib]];
        let xb = &xbl[BLOCK * ib..][..BLOCK];
        block_weight(ib, &mut weight);
        for k in 0..2 {
            let xx = iq1_m_values(shifts[ib], k);
            let index =
                usize::from(qs[2 * ib + k]) | ((usize::from(qh[ib]) << (8 - 4 * k)) & 0x700);
            let pg = &t.grid[index];
            for j in 0..8 {
                let w = weight[8 * k + j];
                let v = xx[((pg[j] - 1) / 2) as usize] * (2 * l + 1) as f32;
                sumqx = madd(q, w * v, xb[8 * k + j], sumqx);
                sumq2 = madd(q, w * v, v, sumq2);
            }
        }
    }
    if sumq2 > 0.0 {
        d = sumqx / sumq2;
    }
    let bits = half::f16::from_f32(d * IQ1_M_FUDGE_IK).to_bits();
    sc[0] |= (bits & 0x000f) << 12;
    sc[1] |= (bits & 0x00f0) << 8;
    sc[2] |= (bits & 0x0f00) << 4;
    sc[3] |= bits & 0xf000;
    for (o, s) in sc_bytes.as_chunks_mut::<2>().0.iter_mut().zip(sc) {
        o.copy_from_slice(&s.to_le_bytes());
    }
}

// IQ1_M's shift choice `k` (0 ++, 1 +-, 2 -+, 3 --) gives group `g` of a 16-value block its shifted values
fn iq1_m_values(k: usize, g: usize) -> &'static [f32; 3] {
    let plus = if g == 0 { k < 2 } else { k.is_multiple_of(2) };
    if plus { &IQ1_X_P } else { &IQ1_X_M }
}

/// One IQ1_M super-block: per 16 values two 11-bit grid indices (low bytes in `qs`, top bits and the halves' shift
/// signs in `qh`) and a 3-bit scale; the f16 block scale is spread over the scale words' top nibbles.
pub(super) fn quantize_iq1_m(xbl: &[f32], qw: Option<&[f32]>, out: &mut [u8], q: Quantizer) {
    const BLOCK: usize = 16;
    let t = tables(GgufType::Iq1M);
    let sigma2 = 2.0 * xbl.iter().map(|x| x * x).sum::<f32>() / QK_K as f32;
    out.fill(0);
    let block_weight = |ib: usize, weight: &mut [f32; BLOCK]| {
        let xb = &xbl[BLOCK * ib..][..BLOCK];
        for i in 0..BLOCK {
            weight[i] = match qw {
                Some(qw) => qw[BLOCK * ib + i] * madd(q, xb[i], xb[i], sigma2).sqrt(),
                None => xb[i] * xb[i],
            };
        }
    };
    let (qs, rest) = out.split_at_mut(32);
    let (qh, sc_bytes) = rest.split_at_mut(16);
    let mut scales = [0f32; QK_K / BLOCK];
    let mut shifts = [0usize; QK_K / BLOCK];
    let mut max_scale = 0f32;
    let mut weight = [0f32; BLOCK];
    let half = BLOCK / 2;
    let p2 = IQ1_X_P[2];
    let m2 = IQ1_X_M[2];
    let cp1 = IQ1_X_P[0] * IQ1_X_P[0] - IQ1_X_P[1] * IQ1_X_P[1];
    let cp2 = IQ1_X_P[1] * IQ1_X_P[1] - IQ1_X_P[2] * IQ1_X_P[2];
    let cm1 = IQ1_X_M[0] * IQ1_X_M[0] - IQ1_X_M[1] * IQ1_X_M[1];
    let cm2 = IQ1_X_M[1] * IQ1_X_M[1] - IQ1_X_M[2] * IQ1_X_M[2];
    for ib in 0..QK_K / BLOCK {
        let xb = &xbl[BLOCK * ib..][..BLOCK];
        block_weight(ib, &mut weight);
        let max = xb.iter().map(|x| x.abs()).fold(xb[0].abs(), f32::max);
        if max < GROUP_MAX_EPS_IQ1_M {
            continue;
        }
        let mut idx = (0..BLOCK).collect::<Vec<_>>();
        idx.sort_by(|&a, &b| xb[a].partial_cmp(&xb[b]).expect("finite weights"));
        // prefix sums per half, as each half takes its own shift
        let (mut px, mut pw) = ([[0f32; BLOCK + 1]; 2], [[0f32; BLOCK + 1]; 2]);
        for (j, &i) in idx.iter().enumerate() {
            let h = usize::from(i >= half);
            px[h][j + 1] = px[h][j] + weight[i] * xb[i];
            px[1 - h][j + 1] = px[1 - h][j];
            pw[h][j + 1] = pw[h][j] + weight[i];
            pw[1 - h][j + 1] = pw[1 - h][j];
        }
        let txs = [px[0][BLOCK], px[1][BLOCK]];
        let tws = [pw[0][BLOCK], pw[1][BLOCK]];
        let (mut best_score, mut scale) = (-f32::MAX, max);
        let (mut besti1, mut besti2, mut best_k) = (usize::MAX, usize::MAX, usize::MAX);
        for i1 in 0..=BLOCK {
            for i2 in i1..=BLOCK {
                let (mut qx_p, mut qx_m, mut q2_p, mut q2_m) =
                    ([0f32; 2], [0f32; 2], [0f32; 2], [0f32; 2]);
                for h in 0..2 {
                    let sx = px[h][i1] + px[h][i2];
                    qx_p[h] = txs[h] * p2 - sx;
                    qx_m[h] = txs[h] * m2 - sx;
                    q2_p[h] = tws[h] * p2 * p2 + pw[h][i1] * cp1 + pw[h][i2] * cp2;
                    q2_m[h] = tws[h] * m2 * m2 + pw[h][i1] * cm1 + pw[h][i2] * cm2;
                }
                let sumqx = [
                    qx_p[0] + qx_p[1],
                    qx_p[0] + qx_m[1],
                    qx_m[0] + qx_p[1],
                    qx_m[0] + qx_m[1],
                ];
                let sumq2 = [
                    q2_p[0] + q2_p[1],
                    q2_p[0] + q2_m[1],
                    q2_m[0] + q2_p[1],
                    q2_m[0] + q2_m[1],
                ];
                for k in 0..4 {
                    if sumq2[k] > 0.0 && sumqx[k] * sumqx[k] > best_score * sumq2[k] {
                        scale = sumqx[k] / sumq2[k];
                        best_score = scale * sumqx[k];
                        (besti1, besti2, best_k) = (i1, i2, k);
                    }
                }
            }
        }
        if besti1 == usize::MAX || best_k == usize::MAX {
            continue;
        }
        let mut l = [0i8; BLOCK];
        split_levels(&idx, besti1, besti2, &mut l);
        if scale < 0.0 {
            for l in &mut l {
                *l = 2 - *l;
            }
            scale = -scale;
            best_k = 3 - best_k;
        }
        let mut indices = [0usize; 2];
        let mut all_on_grid = true;
        for k in 0..2 {
            let at = 8 * k..8 * k + 8;
            all_on_grid &= t.is_on_grid(&l[at.clone()]);
            let xx = iq1_m_values(best_k, k);
            indices[k] = t.iq1_index(
                q,
                &mut l[at.clone()],
                &xb[at.clone()],
                &weight[at],
                scale,
                xx,
            );
        }
        if !all_on_grid {
            let (sumqx, sumq2) = t.iq1_refit(q, xb, &weight, &indices, |g| iq1_m_values(best_k, g));
            if sumqx > 0.0 && sumq2 > 0.0 {
                scale = sumqx / sumq2;
            }
        }
        qs[2 * ib] = (indices[0] & 255) as u8;
        qs[2 * ib + 1] = (indices[1] & 255) as u8;
        qh[ib] = ((indices[0] >> 8) | ((indices[1] >> 8) << 4)) as u8;
        scales[ib] = scale;
        shifts[ib] = best_k;
        max_scale = max_scale.max(scale);
    }
    if max_scale == 0.0 {
        return;
    }
    let mut sc = [0u16; 4];
    let mut d = max_scale / 15.0;
    let id = 1.0 / d;
    let (mut sumqx, mut sumq2) = (0f32, 0f32);
    // the block scale refit over every sub-block with its scale code, empty ones (grid point 0) included
    for ib in 0..QK_K / BLOCK {
        let l = half_level(q, id, scales[ib]).clamp(0, 7);
        sc[ib / 4] |= (l as u16) << (3 * (ib % 4));
        qh[ib] |= IQ1_M_SHIFT_MASKS[shifts[ib]];
        let xb = &xbl[BLOCK * ib..][..BLOCK];
        block_weight(ib, &mut weight);
        for k in 0..2 {
            let xx = iq1_m_values(shifts[ib], k);
            let index =
                usize::from(qs[2 * ib + k]) | ((usize::from(qh[ib]) << (8 - 4 * k)) & 0x700);
            let pg = &t.grid[index];
            for j in 0..8 {
                let w = weight[8 * k + j];
                let q = xx[((pg[j] - 1) / 2) as usize] * (2 * l + 1) as f32;
                sumqx += w * q * xb[8 * k + j];
                sumq2 += w * q * q;
            }
        }
    }
    if sumq2 > 0.0 {
        d = sumqx / sumq2;
    }
    let bits = half::f16::from_f32(d * IQ1_M_SCALE_FUDGE).to_bits();
    sc[0] |= (bits & 0x000f) << 12;
    sc[1] |= (bits & 0x00f0) << 8;
    sc[2] |= (bits & 0x0f00) << 4;
    sc[3] |= bits & 0xf000;
    for (o, s) in sc_bytes.as_chunks_mut::<2>().0.iter_mut().zip(sc) {
        o.copy_from_slice(&s.to_le_bytes());
    }
}
