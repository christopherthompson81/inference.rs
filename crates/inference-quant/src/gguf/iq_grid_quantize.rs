//! ggml's lattice quantizers for the grid types (`quantize_row_iq2_xxs_impl`, `quantize_row_iq2_xs_impl`,
//! `quantize_row_iq2_s_impl`, `quantize_row_iq3_xxs_impl`, `quantize_row_iq3_s_impl` in ggml-quants.c): each group of
//! 8 (IQ2) or 4 (IQ3) magnitudes snaps to a grid point, found through `iq2xs_init_impl` / `iq3xs_init_impl`'s map
//! from every point of the grid's levels to its grid index or its nearest grid points.
// the loops index as ggml's C does, so each line can be checked against it
#![allow(clippy::needless_range_loop)]

use std::sync::OnceLock;

use super::iq_quantize::nearest_int;
use super::iq_tables::{IQ1S_GRID, IQ2S_GRID, IQ2XS_GRID, IQ2XXS_GRID, IQ3S_GRID, IQ3XXS_GRID};
use super::kernel::GgufType;

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
// IQ1_M's qh bits for each half's shift: none, the second half's, the first's, both
const IQ1_M_SHIFT_MASKS: [u8; 4] = [0x00, 0x80, 0x08, 0x88];
pub(super) const BLOCK_IQ3_S: usize = 110;
const GROUP_MAX_EPS: f32 = 1e-15;
const GROUP_MAX_EPS_IQ2_S: f32 = 1e-8;
// three levels per coordinate, written 2 l + 1
const KMAX_Q: i32 = 3;
// ggml stores IQ2_S's scale this much below the max-based one
const IQ2_S_SCALE_FUDGE: f32 = 0.9875;
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
    fn index(&self, l: &mut [i8], xval: &[f32], weight: &[f32], scale: f32) -> usize {
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
                let diff = scale * f32::from(pg[i]) - xval[i];
                d2 += weight[i] * diff * diff;
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

fn level(id: f32, x: f32) -> i8 {
    level_of(id, x, KMAX_Q)
}

fn level_of(id: f32, x: f32, kmax: i32) -> i8 {
    nearest_int(0.5 * (id * x - 1.0)).clamp(0, kmax - 1) as i8
}

fn fit(xval: &[f32], weight: &[f32], l: &[i8]) -> (f32, f32) {
    let (mut sumqx, mut sumq2) = (0f32, 0f32);
    for i in 0..xval.len() {
        let q = f32::from(2 * l[i] + 1);
        sumqx += weight[i] * xval[i] * q;
        sumq2 += weight[i] * q * q;
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
pub(super) fn quantize_iq2_xxs(xbl: &[f32], qw: &[f32], out: &mut [u8]) {
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
            weight[i] = qw[i] * (sigma2 + xb[i] * xb[i]).sqrt();
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
        );
        // ggml starts from these levels, kept should no scale step beat them
        let mut l = lu.map(|l| l as i8);
        let eff_max = scale * KMAX_Q as f32;
        if eff_max <= 0.0 {
            continue;
        }
        let mut best = 0f32;
        for is in -6..=6 {
            let id = ((2 * KMAX_Q - 1) as f32 + is as f32 * 0.1) / eff_max;
            let this_scale = 1.0 / id;
            let mut laux = [0i8; 32];
            for k in 0..4 {
                let at = 8 * k..8 * k + 8;
                for i in at.clone() {
                    laux[i] = level(id, xval[i]);
                }
                t.index(
                    &mut laux[at.clone()],
                    &xval[at.clone()],
                    &waux[at],
                    this_scale,
                );
            }
            let (sumqx, sumq2) = fit(&xval, &weight, &laux);
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
                    l[i] = level(id, xval[i]);
                }
                let index = t.index(
                    &mut l[at.clone()],
                    &xval[at.clone()],
                    &waux[at.clone()],
                    scale,
                );
                for (l, &p) in l[at].iter_mut().zip(&t.grid[index]) {
                    *l = (p - 1) / 2;
                }
            }
            let (sumqx, sumq2) = fit(&xval, &weight, &l);
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
        let l = nearest_int(0.5 * (id * scales[ib] - 1.0)).clamp(0, 15);
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
                Some(qw) => qw[16 * ib + i] * (sigma2 + xb[i] * xb[i]).sqrt(),
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
            let id = ((2 * KMAX_Q - 1) as f32 + is as f32 * 0.1) / max;
            let this_scale = 1.0 / id;
            let mut laux = [0i8; 16];
            let mut on_grid_aux = [true; 2];
            for k in 0..2 {
                let at = 8 * k..8 * k + 8;
                for i in at.clone() {
                    laux[i] = level(id, xval[i]);
                }
                let key_on = t.is_on_grid(&laux[at.clone()]);
                on_grid_aux[k] = key_on;
                if !key_on {
                    t.index(
                        &mut laux[at.clone()],
                        &xval[at.clone()],
                        &waux[at],
                        this_scale,
                    );
                }
            }
            let (sumqx, sumq2) = fit(&xval, &weight, &laux);
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
                    l[i] = level(id, xval[i]);
                }
                t.index(&mut l[at.clone()], &xval[at.clone()], &waux[at], scale);
            }
            let (sumqx, sumq2) = fit(&xval, &weight, &l);
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
        let l = nearest_int(0.5 * (id * scales[ib] - 1.0)).clamp(0, 15) as u8;
        packed[ib / 2] |= l << (4 * (ib % 2));
    }
    (d, indices, signs_out, packed)
}

/// One IQ2_XS super-block: `d`, 32 words of a 9-bit grid index under 7 sign bits, eight scale bytes.
pub(super) fn quantize_iq2_xs(xbl: &[f32], qw: &[f32], out: &mut [u8]) {
    let (d, indices, signs, scales) = quantize_iq2_16(xbl, Some(qw), false);
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

/// One IQ2_S super-block: `d`, 32 grid index low bytes, 32 sign bytes, eight bytes of the indices' top bits, eight
/// scale bytes.
pub(super) fn quantize_iq2_s(xbl: &[f32], qw: Option<&[f32]>, out: &mut [u8]) {
    let (d, indices, signs, scales) = quantize_iq2_16(xbl, qw, true);
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
    let (mut weight, mut waux, mut xval) = ([0f32; 32], [0f32; 32], [0f32; 32]);
    for ib in 0..QK_K / 32 {
        let xb = &xbl[32 * ib..][..32];
        for i in 0..32 {
            weight[i] = match qw {
                Some(qw) => qw[32 * ib + i] * (sigma2 + xb[i] * xb[i]).sqrt(),
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
        let mut on_grid = [!s_variant; 8];
        let steps = if s_variant { 9 } else { 15 };
        for is in -steps..=steps {
            let id = ((2 * KMAX_Q3 - 1) as f32 + is as f32 * 0.2) / max;
            let this_scale = 1.0 / id;
            let mut laux = [0i8; 32];
            let mut on_grid_aux = [true; 8];
            for k in 0..8 {
                let at = 4 * k..4 * k + 4;
                for i in at.clone() {
                    laux[i] = level_of(id, xval[i], KMAX_Q3);
                }
                on_grid_aux[k] = t.is_on_grid(&laux[at.clone()]);
                if !on_grid_aux[k] {
                    t.index(
                        &mut laux[at.clone()],
                        &xval[at.clone()],
                        &waux[at],
                        this_scale,
                    );
                }
            }
            let (sumqx, sumq2) = fit(&xval, &weight, &laux);
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
                    *lk = level_of(id, xval[i], KMAX_Q3);
                }
                let index = t.index(&mut lk, &xval[at.clone()], &waux[at.clone()], scale);
                for (l, &p) in l[at].iter_mut().zip(&t.grid[index]) {
                    *l = (p - 1) / 2;
                }
            }
            let (sumqx, sumq2) = fit(&xval, &weight, &l);
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
        codes[ib] = nearest_int(0.5 * (id * scales[ib] - 1.0)).clamp(0, 15) as u8;
    }
    (d, indices, signs_out, codes)
}

/// One IQ3_XXS super-block: `d`, 64 grid index bytes, then per 32 values a word of four 7-bit sign codes under a
/// 4-bit scale.
pub(super) fn quantize_iq3_xxs(xbl: &[f32], qw: Option<&[f32]>, out: &mut [u8]) {
    let (d, indices, signs, codes) = quantize_iq3(xbl, qw, false);
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
pub(super) fn quantize_iq3_s(xbl: &[f32], qw: Option<&[f32]>, out: &mut [u8]) {
    let (d, indices, signs, codes) = quantize_iq3(xbl, qw, true);
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
                let diff = scale * xx[((pg[i] - 1) / 2) as usize] - xval[i];
                d2 += weight[i] * diff * diff;
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
                let q = xx(k)[((pg[j] - 1) / 2) as usize];
                sumqx += w * q * xb[8 * k + j];
                sumq2 += w * q * q;
            }
        }
        (sumqx, sumq2)
    }
}

// the block's values in ascending order (a stable sort, as glibc's qsort of these pairs is) and the running sums of
// weight times value and of weight over them
fn sorted_sums(xb: &[f32], weight: &[f32]) -> (Vec<usize>, Vec<f32>, Vec<f32>) {
    let mut idx = (0..xb.len()).collect::<Vec<_>>();
    idx.sort_by(|&a, &b| xb[a].partial_cmp(&xb[b]).expect("finite weights"));
    let (mut sumx, mut sumw) = (vec![0f32; xb.len() + 1], vec![0f32; xb.len() + 1]);
    for (j, &i) in idx.iter().enumerate() {
        sumx[j + 1] = sumx[j] + weight[i] * xb[i];
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
pub(super) fn quantize_iq1_s(xbl: &[f32], qw: &[f32], out: &mut [u8]) {
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
            weight[i] = qw[BLOCK * ib + i] * (sigma2 + xb[i] * xb[i]).sqrt();
        }
        let max = xb.iter().map(|x| x.abs()).fold(xb[0].abs(), f32::max);
        if max < GROUP_MAX_EPS_IQ1_S {
            continue;
        }
        let (idx, sumx, sumw) = sorted_sums(xb, &weight);
        let (mut best_score, mut scale) = (-f32::MAX, max);
        let (mut besti1, mut besti2, mut best_shift) = (usize::MAX, usize::MAX, 0i8);
        for i1 in 0..=BLOCK {
            for i2 in i1..=BLOCK {
                for (shift, xs) in [(1i8, &IQ1_X_P), (-1, &IQ1_X_M)] {
                    let sumqx = (sumx[i1] - sumx[0]) * xs[0]
                        + (sumx[i2] - sumx[i1]) * xs[1]
                        + (sumx[BLOCK] - sumx[i2]) * xs[2];
                    let sumq2 = (sumw[i1] - sumw[0]) * xs[0] * xs[0]
                        + (sumw[i2] - sumw[i1]) * xs[1] * xs[1]
                        + (sumw[BLOCK] - sumw[i2]) * xs[2] * xs[2];
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
            indices[k] = t.iq1_index(&mut l[at.clone()], &xb[at.clone()], &weight[at], scale, xx);
        }
        if !all_on_grid {
            let (sumqx, sumq2) = t.iq1_refit(xb, &weight, &indices, |_| xx);
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
        let mut l = nearest_int(0.5 * (id * scales[ib] - 1.0)).clamp(0, 7) as u16;
        if shifts[ib] == -1 {
            l |= 8;
        }
        qh[ib] |= l << 12;
    }
    for (o, h) in out[34..50].as_chunks_mut::<2>().0.iter_mut().zip(qh) {
        o.copy_from_slice(&h.to_le_bytes());
    }
}

// IQ1_M's shift choice `k` (0 ++, 1 +-, 2 -+, 3 --) gives group `g` of a 16-value block its shifted values
fn iq1_m_values(k: usize, g: usize) -> &'static [f32; 3] {
    let plus = if g == 0 { k < 2 } else { k.is_multiple_of(2) };
    if plus { &IQ1_X_P } else { &IQ1_X_M }
}

/// One IQ1_M super-block: per 16 values two 11-bit grid indices (low bytes in `qs`, top bits and the halves' shift
/// signs in `qh`) and a 3-bit scale; the f16 block scale is spread over the scale words' top nibbles.
pub(super) fn quantize_iq1_m(xbl: &[f32], qw: Option<&[f32]>, out: &mut [u8]) {
    const BLOCK: usize = 16;
    let t = tables(GgufType::Iq1M);
    let sigma2 = 2.0 * xbl.iter().map(|x| x * x).sum::<f32>() / QK_K as f32;
    out.fill(0);
    let block_weight = |ib: usize, weight: &mut [f32; BLOCK]| {
        let xb = &xbl[BLOCK * ib..][..BLOCK];
        for i in 0..BLOCK {
            weight[i] = match qw {
                Some(qw) => qw[BLOCK * ib + i] * (sigma2 + xb[i] * xb[i]).sqrt(),
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
            indices[k] = t.iq1_index(&mut l[at.clone()], &xb[at.clone()], &weight[at], scale, xx);
        }
        if !all_on_grid {
            let (sumqx, sumq2) = t.iq1_refit(xb, &weight, &indices, |g| iq1_m_values(best_k, g));
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
        let l = nearest_int(0.5 * (id * scales[ib] - 1.0)).clamp(0, 7);
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
