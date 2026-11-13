//! ik_llama.cpp's quantizers for its 2- and 3-bit table types (`quantize_row_iq2_k_impl` and its siblings in
//! iqk_quantize.cpp). g++ contracts a product into the add (or subtract) that alone uses it, so each sum here is
//! written as the fused multiply-adds its native build runs, which the byte parity against it depends on.

// the loops index as ik's C++ does, so each line can be checked against it
#![allow(clippy::needless_range_loop)]

use super::iq_quantize::nearest_int;
use super::kernel::GgufType;

const QK_K: usize = 256;
pub(super) const BLOCK_IQ2_K: usize = 76;
pub(super) const BLOCK_IQ2_KS: usize = 70;
pub(super) const F16_ROW_SCALE: usize = 2;
const IQ2_KS_EPS: f32 = 1e-14;
// the scale search's steps either side of each codebook's max-based scale, half a code apart
const IQ2_KS_TRIES: i32 = 13;
const LANES: usize = 8;
pub(super) const BLOCK_IQ2_KL: usize = 86;
pub(super) const BLOCK_IQ3_K: usize = 110;
pub(super) const BLOCK_IQ3_KS: usize = 102;
const IQ3_KS_TRIES: i32 = 5;
const IQ3_K_FUDGE: f32 = 1.01;
// the AVX2 scale search's steps either side of each codebook's max-based scale, and their spacing
const IQ3_K_TRIES: i32 = 3;
const IQ3_K_STEP: f32 = 2.0;
// `k_simple_index_iq3nl`: from `trunc(x) - values[0]` (clamped to 0..=127), the lower code of the two to compare
const SIMPLE_INDEX_IQ3NL: [u8; 128] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
];
const IQ2_KL_FUDGE: f32 = 1.025;
const IQ2_KL_SIGMA_FACTOR: f32 = 2.25;
const IQ2_KL_TRIES: i32 = 5;
// ik's 3-bit codebook (its first half), which IQ2_KL pairs
const IQ3NL_VALUES: [i8; 16] = [
    -63, -40, -23, -10, 1, 13, 28, 47, -59, -36, -19, -6, 5, 17, 32, 51,
];
// `iq3nl_index`: from `trunc(x) - values[0]`, the code (or 8 + the lower of two to compare)
const IQ3NL_INDEX: [i8; 111] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 9,
    9, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 10, 10, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 11, 11, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 12, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 13, 13, 6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 14, 14, 7, 7, 7, 7, 7, 7, 7, 7, 7,
];
// IQ2_KL's 32 pairs as code pairs `8 l1 + l2`: the pair's index, or `-(n + 1)` for the nth pair off the grid
const IQ2_KL_INDEX: [i8; 64] = [
    -1, -2, 0, -3, -4, 1, -5, -6, 2, -7, -8, 3, -9, 4, -10, 5, -11, 6, 7, -12, 8, 9, 10, -13, 11,
    -14, -15, -16, 12, 13, -17, 14, -18, -19, 15, 16, 17, 18, 19, -20, -21, 20, 21, 22, 23, 24,
    -22, -23, 25, -24, 26, -25, 27, -26, 28, 29, -27, -28, 30, -29, -30, 31, -31, -32,
];
// the grid pairs an off-grid pair tries, in ik's order
const IQ2_KL_NEIGHBOURS: [[u8; 8]; 32] = [
    [2, 0, 6, 11, 7, 3, 8, 15],
    [0, 2, 3, 6, 7, 1, 8, 4],
    [0, 1, 3, 4, 8, 7, 9, 6],
    [1, 0, 3, 4, 8, 9, 7, 10],
    [1, 4, 5, 10, 9, 3, 8, 0],
    [5, 1, 4, 10, 9, 14, 8, 3],
    [6, 2, 7, 0, 3, 11, 8, 15],
    [3, 7, 0, 6, 8, 4, 12, 9],
    [3, 4, 8, 9, 1, 7, 12, 10],
    [4, 10, 5, 9, 1, 8, 13, 14],
    [11, 2, 6, 7, 20, 15, 25, 21],
    [8, 7, 3, 12, 9, 16, 17, 13],
    [14, 5, 10, 19, 9, 13, 4, 18],
    [6, 15, 7, 11, 20, 21, 16, 2],
    [15, 7, 16, 6, 21, 12, 17, 22],
    [12, 16, 17, 8, 15, 7, 13, 22],
    [19, 10, 13, 18, 14, 9, 12, 24],
    [11, 20, 25, 6, 15, 2, 21, 7],
    [20, 15, 21, 6, 11, 7, 16, 26],
    [14, 19, 29, 10, 28, 18, 13, 24],
    [25, 11, 20, 21, 15, 6, 26, 30],
    [19, 24, 28, 18, 29, 23, 13, 17],
    [29, 19, 14, 28, 24, 18, 10, 13],
    [20, 26, 21, 25, 30, 15, 22, 16],
    [27, 26, 22, 23, 21, 30, 16, 24],
    [27, 24, 28, 31, 23, 18, 22, 17],
    [25, 30, 20, 26, 21, 11, 15, 22],
    [30, 26, 25, 20, 21, 27, 22, 15],
    [30, 27, 31, 26, 22, 23, 21, 24],
    [31, 27, 30, 26, 28, 23, 22, 24],
    [31, 28, 29, 27, 24, 23, 19, 18],
    [29, 28, 31, 24, 19, 27, 14, 18],
];
// ik's 2-bit codebook, then the same shifted for a block with its `extra` bit set
const IQ2NL_VALUES: [i8; 8] = [-31, -13, 1, 17, -26, -8, 6, 22];
// `ggml_get_quantize_fudge_factor`'s defaults
const IQ2_K_FUDGE: f32 = 1.03;
const IQ2_K_EPS: f32 = 1e-16;
// QHelper's floor for an imatrix block
const IMATRIX_EPS: f32 = 1e-9;

#[inline(always)]
fn fma(a: f32, b: f32, c: f32) -> f32 {
    a.mul_add(b, c)
}

/// `QHelper::row_weights`: an imatrix block of `block` columns weighing (or over) all but zero values is replaced by
/// a tiny constant.
pub(super) fn row_weights(x: &[f32], imatrix: &[f32], block: usize) -> Vec<f32> {
    let floor = block as f32 * IMATRIX_EPS * IMATRIX_EPS;
    let mut out = imatrix.to_vec();
    for (wb, xb) in out.chunks_exact_mut(block).zip(x.chunks_exact(block)) {
        let (mut sumw2, mut sumx2, mut sumwx) = (0f32, 0f32, 0f32);
        for (w, v) in wb.iter().zip(xb) {
            sumw2 = fma(*w, *w, sumw2);
            sumx2 = fma(*v, *v, sumx2);
            sumwx = fma(*w, v.abs(), sumwx);
        }
        if !(sumw2 > floor && sumx2 > floor && sumwx > floor) {
            wb.fill(IMATRIX_EPS);
        }
    }
    out
}

fn sum_squares(x: &[f32]) -> f32 {
    x.iter().fold(0f32, |s, v| fma(*v, *v, s))
}

/// ik's `make_qx_quants` (its own, not ggml's): the scale of `x` on signed levels `-nmax..nmax`, searched over 19
/// steps around the max-based one, and each value's level plus `nmax`.
fn make_qx_quants(nmax: i32, x: &[f32], qw: &[f32]) -> (f32, Vec<i32>) {
    let (mut max, mut amax) = (0f32, 0f32);
    for &v in x {
        if v.abs() > amax {
            (amax, max) = (v.abs(), v);
        }
    }
    if amax == 0.0 {
        return (0.0, vec![0; x.len()]);
    }
    let levels = |iscale: f32| {
        x.iter()
            .map(|&v| nearest_int(iscale * v).clamp(-nmax, nmax - 1))
            .collect::<Vec<_>>()
    };
    let sums = |l: &[i32]| {
        let (mut sumlx, mut suml2) = (0f32, 0f32);
        for ((&v, &w), &l) in x.iter().zip(qw).zip(l) {
            let l = l as f32;
            sumlx = fma(w * v, l, sumlx);
            suml2 = fma(w * l, l, suml2);
        }
        (sumlx, suml2)
    };
    let mut best_l = levels(-(nmax as f32) / max);
    let (sumlx, suml2) = sums(&best_l);
    let mut scale = if suml2 != 0.0 { sumlx / suml2 } else { 0.0 };
    let mut best = scale * sumlx;
    for is in -9..=9 {
        if is == 0 {
            continue;
        }
        let l = levels(-(nmax as f32 + 0.1 * is as f32) / max);
        let (sumlx, suml2) = sums(&l);
        if suml2 > 0.0 && sumlx * sumlx > best * suml2 {
            scale = sumlx / suml2;
            best = scale * sumlx;
            best_l = l;
        }
    }
    (scale, best_l.into_iter().map(|l| l + nmax).collect())
}

fn best_index_iq2nl(values: &[i8], x: f32) -> usize {
    let v = |i: usize| f32::from(values[i]);
    let idx = if x < v(1) {
        0
    } else if x > v(2) {
        2
    } else {
        1
    };
    if x - v(idx) < v(idx + 1) - x {
        idx
    } else {
        idx + 1
    }
}

fn iq2_k_weights(xb: &[f32], qw: Option<&[f32]>, sigma2: f32, weight: &mut [f32]) {
    for j in 0..xb.len() {
        weight[j] = match qw {
            Some(qw) => qw[j] * fma(xb[j], xb[j], sigma2).sqrt(),
            None => fma(xb[j], xb[j], 0.25 * sigma2),
        };
    }
}

// `sum over the 4 sorted groups of (group sum) * value`, as g++ fuses the chain: the first product into the second,
// then each later one over the running sum
fn four(s: [f32; 4], v: [f32; 4]) -> f32 {
    fma(s[3], v[3], fma(s[2], v[2], fma(s[0], v[0], s[1] * v[1])))
}

/// One IQ2_K super-block: per 16 values a 4-bit signed scale and a codebook pick, 2-bit indices.
pub(super) fn quantize_iq2_k(xbl: &[f32], qw: Option<&[f32]>, out: &mut [u8]) {
    const BLOCK: usize = 16;
    out.fill(0);
    let (values, shifted) = IQ2NL_VALUES.split_at(4);
    let sigma2 = 1.5 * sum_squares(xbl) / QK_K as f32;
    let mut scales = [0f32; QK_K / BLOCK];
    let mut sw = [0f32; QK_K / BLOCK];
    let mut weight = [0f32; BLOCK];
    let mut extra = 0u16;
    let mut max_abs_scale = 0f32;
    for ib in 0..QK_K / BLOCK {
        let xb = &xbl[BLOCK * ib..][..BLOCK];
        iq2_k_weights(
            xb,
            qw.map(|w| &w[BLOCK * ib..][..BLOCK]),
            sigma2,
            &mut weight,
        );
        sw[ib] = weight.iter().sum();
        let amax = xb.iter().fold(0f32, |m, v| m.max(v.abs()));
        if amax < IQ2_K_EPS {
            continue;
        }
        let mut order = (0..BLOCK).collect::<Vec<_>>();
        order.sort_by(|&a, &b| {
            xb[a]
                .partial_cmp(&xb[b])
                .expect("finite weights")
                .then(a.cmp(&b))
        });
        let (mut sumx, mut sumw) = ([0f32; BLOCK + 1], [0f32; BLOCK + 1]);
        for (j, &jj) in order.iter().enumerate() {
            sumw[j + 1] = sumw[j] + weight[jj];
            sumx[j + 1] = fma(weight[jj], xb[jj], sumx[j]);
        }
        let (mut best, mut d, mut is_shifted) = (0f32, 0f32, false);
        for i1 in 0..BLOCK {
            for i2 in i1..BLOCK {
                for i3 in i2..BLOCK {
                    let gx = [
                        sumx[i1] - sumx[0],
                        sumx[i2] - sumx[i1],
                        sumx[i3] - sumx[i2],
                        sumx[BLOCK] - sumx[i3],
                    ];
                    let gw = [
                        sumw[i1] - sumw[0],
                        sumw[i2] - sumw[i1],
                        sumw[i3] - sumw[i2],
                        sumw[BLOCK] - sumw[i3],
                    ];
                    for (codebook, reversed, shift) in [
                        (values, false, false),
                        (shifted, false, true),
                        (values, true, false),
                        (shifted, true, true),
                    ] {
                        let v: [f32; 4] = std::array::from_fn(|k| {
                            f32::from(codebook[if reversed { 3 - k } else { k }])
                        });
                        let sumqx = four(gx, v);
                        // each term is (group weight * value) * value; the outer product is the one fused
                        let sumq2 = four(std::array::from_fn(|k| gw[k] * v[k]), v);
                        if sumq2 > 0.0 && sumqx * sumqx > best * sumq2 {
                            d = sumqx / sumq2;
                            best = d * sumqx;
                            is_shifted = shift;
                        }
                    }
                }
            }
        }
        scales[ib] = d;
        if is_shifted {
            extra |= 1 << ib;
        }
        max_abs_scale = max_abs_scale.max(d.abs());
    }
    if max_abs_scale == 0.0 {
        return;
    }
    let (d, _) = make_qx_quants(8, &scales, &sw);
    if d == 0.0 {
        return;
    }
    out[2..4].copy_from_slice(&extra.to_le_bytes());
    let id = 1.0 / d;
    let (mut sumqx, mut sumq2) = (0f32, 0f32);
    for ib in 0..QK_K / BLOCK {
        let ls = nearest_int(id * scales[ib]).clamp(-8, 7);
        out[4 + ib / 2] |= ((ls + 8) as u8) << (4 * (ib % 2));
        let dl = d * ls as f32;
        if dl == 0.0 {
            continue;
        }
        let codebook = if extra & (1 << ib) != 0 {
            shifted
        } else {
            values
        };
        let xb = &xbl[BLOCK * ib..][..BLOCK];
        iq2_k_weights(
            xb,
            qw.map(|w| &w[BLOCK * ib..][..BLOCK]),
            sigma2,
            &mut weight,
        );
        let idl = 1.0 / dl;
        let ib32 = ib / 2;
        let qs = &mut out[12 + 32 * (ib32 / 4) + 16 * (ib % 2)..][..16];
        for j in 0..BLOCK {
            let ibest = best_index_iq2nl(codebook, idl * xb[j]);
            qs[j] |= (ibest as u8) << (2 * (ib32 % 4));
            let q = (i32::from(codebook[ibest]) * ls) as f32;
            sumqx = fma(weight[j] * q, xb[j], sumqx);
            sumq2 = fma(weight[j] * q, q, sumq2);
        }
    }
    let scale = if sumq2 > 0.0 { sumqx / sumq2 } else { d };
    out[..2].copy_from_slice(&half::f16::from_f32(IQ2_K_FUDGE * scale).to_le_bytes());
}

// `hsum_float_8`: lanes i and i + 4, then those halves' pairs
fn hsum8(v: &[f32; LANES]) -> f32 {
    let r: [f32; 4] = std::array::from_fn(|i| v[i] + v[i + 4]);
    (r[0] + r[2]) + (r[1] + r[3])
}

// 32 values' nearest code of `values` at scale `d` (ties to the earlier code), as the AVX2 search compares squares
fn nearest_codes(xb: &[f32], d: f32, values: &[i8], negated: bool) -> [u8; 32] {
    std::array::from_fn(|i| {
        let diff2 = |j: usize| {
            let val = d * f32::from(values[j]);
            let diff = if negated { xb[i] + val } else { xb[i] - val };
            diff * diff
        };
        let (mut best, mut code) = (diff2(0), 0u8);
        for j in 1..4 {
            let d2 = diff2(j);
            if d2 < best {
                (best, code) = (d2, j as u8);
            }
        }
        code
    })
}

// the weighted fit of `xb` to the codes' values (each times `scale`), summed in the AVX2 build's eight lanes
fn lane_fit(
    xb: &[f32],
    w: &[f32],
    values: &[i8],
    codes: &[u8; 32],
    scale: f32,
    lanes: &mut [[f32; LANES]; 2],
) {
    for k in 0..4 {
        for l in 0..LANES {
            let i = 8 * k + l;
            let q = scale * f32::from(values[usize::from(codes[i])]);
            let wq = w[i] * q;
            lanes[0][l] = fma(wq, xb[i], lanes[0][l]);
            lanes[1][l] = fma(wq, q, lanes[1][l]);
        }
    }
}

// `compute_1block_iq2ks`: the codes at `d` and at `-d`, whichever fits better, with its refit scale and score
fn iq2ks_block(d: f32, xb: &[f32], w: &[f32], values: &[i8]) -> Option<([u8; 32], f32, f32)> {
    let mut result: Option<([u8; 32], f32, f32)> = None;
    for negated in [false, true] {
        let codes = nearest_codes(xb, d, values, negated);
        let q: Vec<f32> = codes
            .iter()
            .map(|&c| f32::from(values[usize::from(c)]))
            .collect();
        let mut lanes = [[0f32; LANES]; 2];
        for k in 0..4 {
            for l in 0..LANES {
                let i = 8 * k + l;
                let wq = w[i] * q[i];
                lanes[0][l] = fma(wq, xb[i], lanes[0][l]);
                lanes[1][l] = fma(wq, q[i], lanes[1][l]);
            }
        }
        let (sumqx, sumq2) = (hsum8(&lanes[0]), hsum8(&lanes[1]));
        let better = match result {
            None => sumq2 > 0.0,
            Some((_, _, score)) => sumq2 > 0.0 && sumqx * sumqx > score * sumq2,
        };
        if better {
            let best_d = sumqx / sumq2;
            result = Some((codes, best_d, sumqx * best_d));
        }
    }
    result
}

// `compute_1block_iq2ks_rmse`: the codes at `d` and their weighted squared error
fn iq2ks_rmse(d: f32, xb: &[f32], w: &[f32], values: &[i8]) -> ([u8; 32], f32) {
    let codes = nearest_codes(xb, d, values, false);
    let mut lanes = [0f32; LANES];
    for k in 0..4 {
        for l in 0..LANES {
            let i = 8 * k + l;
            let diff = fma(-d, f32::from(values[usize::from(codes[i])]), xb[i]);
            lanes[l] = fma(w[i] * diff, diff, lanes[l]);
        }
    }
    (codes, hsum8(&lanes))
}

/// One IQ2_KS row (`quantize_row_iq2_ks_fast_impl`, what ik's quantize runs unless told `--slow-iq2ks`): per 32
/// values the best scale over both codebooks; the scales as 5-bit multiples of one row scale, each nudged a step
/// where that lowers the error; then the row scale refit to the codes.
fn quantize_iq2_ks_row(x: &[f32], qw: Option<&[f32]>, out: &mut [u8]) {
    const BLOCK: usize = 32;
    let (header, blocks) = out.split_at_mut(F16_ROW_SCALE);
    header.fill(0);
    blocks.fill(0);
    let (values, shifted) = IQ2NL_VALUES.split_at(4);
    let nblock = x.len() / QK_K;
    let mut all_scales = vec![0f32; nblock * QK_K / BLOCK];
    let mut all_sw = vec![0f32; nblock * QK_K / BLOCK];
    let mut weight = [0f32; BLOCK];
    let weights_of = |ibl: usize, ib: usize, sigma2: f32, weight: &mut [f32; BLOCK]| {
        let xb = &x[ibl * QK_K + BLOCK * ib..][..BLOCK];
        iq2_k_weights(
            xb,
            qw.map(|w| &w[ibl * QK_K + BLOCK * ib..][..BLOCK]),
            sigma2,
            weight,
        );
    };
    for ibl in 0..nblock {
        let xbl = &x[ibl * QK_K..][..QK_K];
        let sigma2 = 1.5 * sum_squares(xbl) / QK_K as f32;
        let mut extra = 0u16;
        for ib in 0..QK_K / BLOCK {
            let xb = &xbl[BLOCK * ib..][..BLOCK];
            weights_of(ibl, ib, sigma2, &mut weight);
            let (mut amax, mut max, mut sumw) = (0f32, 0f32, 0f32);
            for j in 0..BLOCK {
                if xb[j].abs() > amax {
                    (amax, max) = (xb[j].abs(), xb[j]);
                }
                sumw += weight[j];
            }
            all_sw[ibl * 8 + ib] = sumw;
            if amax < IQ2_KS_EPS {
                continue;
            }
            let mut d = max / f32::from(IQ2NL_VALUES[7]);
            let mut best = 0f32;
            if let Some((_, this_d, score)) = iq2ks_block(d, xb, &weight, values) {
                (best, d) = (score, this_d);
            }
            let mut is_shifted = false;
            for (codebook, shift) in [(values, false), (shifted, true)] {
                for itry in -IQ2_KS_TRIES..=IQ2_KS_TRIES {
                    let start = max / (f32::from(codebook[0]) + 0.5 * itry as f32);
                    if let Some((_, this_d, score)) = iq2ks_block(start, xb, &weight, codebook)
                        && score > best
                    {
                        (best, d) = (score, this_d);
                        is_shifted = shift;
                    }
                }
            }
            all_scales[ibl * 8 + ib] = d;
            if is_shifted {
                extra |= 1 << ib;
            }
        }
        blocks[ibl * BLOCK_IQ2_KS..][..2].copy_from_slice(&extra.to_le_bytes());
    }
    let (d, mut all_ls) = make_qx_quants(16, &all_scales, &all_sw);
    if d == 0.0 {
        return;
    }
    let mut lanes = [[0f32; LANES]; 2];
    for ibl in 0..nblock {
        let xbl = &x[ibl * QK_K..][..QK_K];
        let sigma2 = 1.5 * sum_squares(xbl) / QK_K as f32;
        let block = &mut blocks[ibl * BLOCK_IQ2_KS..][..BLOCK_IQ2_KS];
        let mut codes = [[0u8; 32]; 4];
        for ib in 0..QK_K / BLOCK {
            let extra = u16::from_le_bytes([block[0], block[1]]);
            let codebook = if extra & (1 << ib) != 0 {
                shifted
            } else {
                values
            };
            let xb = &xbl[BLOCK * ib..][..BLOCK];
            weights_of(ibl, ib, sigma2, &mut weight);
            let ls = &mut all_ls[ibl * 8 + ib];
            let scale = all_scales[ibl * 8 + ib];
            let dl = d * (*ls - 16) as f32;
            let (mut c, rmse1) = iq2ks_rmse(dl, xb, &weight, codebook);
            if *ls > 0 && dl > scale {
                let (c2, rmse2) = iq2ks_rmse(d * (*ls - 17) as f32, xb, &weight, codebook);
                if rmse2 < rmse1 {
                    *ls -= 1;
                    c = c2;
                }
            } else if *ls < 15 && dl < scale {
                let (c2, rmse2) = iq2ks_rmse(d * (*ls - 15) as f32, xb, &weight, codebook);
                if rmse2 < rmse1 {
                    *ls += 1;
                    c = c2;
                }
            }
            lane_fit(xb, &weight, codebook, &c, (*ls - 16) as f32, &mut lanes);
            block[2 + ib / 2] |= ((*ls & 0xf) as u8) << (4 * (ib % 2));
            let extra = extra | (((*ls >> 4) as u16) << (8 + ib));
            block[..2].copy_from_slice(&extra.to_le_bytes());
            codes[ib % 4] = c;
            if ib % 4 == 3 {
                let qs = &mut block[6 + 32 * (ib / 4)..][..32];
                for j in 0..32 {
                    qs[j] =
                        codes[0][j] | (codes[1][j] << 2) | (codes[2][j] << 4) | (codes[3][j] << 6);
                }
            }
        }
    }
    let (sumqx, sumq2) = (hsum8(&lanes[0]), hsum8(&lanes[1]));
    let scale = if sumq2 > 0.0 { sumqx / sumq2 } else { d };
    header.copy_from_slice(&half::f16::from_f32(scale).to_le_bytes());
}

// `best_index_iq3nl`: the nearest code of the 3-bit codebook
fn best_index_iq3nl(values: &[i8], x: f32) -> usize {
    let ix = x as i32 - i32::from(values[0]);
    if !(0..111).contains(&ix) {
        return if ix < 0 { 0 } else { 7 };
    }
    let ix = IQ3NL_INDEX[ix as usize] as usize;
    if ix < 8 {
        return ix;
    }
    let (lo, hi) = (f32::from(values[ix - 8]), f32::from(values[ix - 7]));
    if x - lo < hi - x { ix - 8 } else { ix - 7 }
}

// IQ2_KL's grid: pair `i` of the 3-bit codebook's values
fn iq2_kl_grid() -> [(f32, f32); 32] {
    let mut grid = [(0f32, 0f32); 32];
    for (j, &i) in IQ2_KL_INDEX.iter().enumerate() {
        if i >= 0 {
            grid[i as usize] = (
                f32::from(IQ3NL_VALUES[j / 8]),
                f32::from(IQ3NL_VALUES[j % 8]),
            );
        }
    }
    grid
}

// the pair code for `x1, x2` at inverse scale `id`, or the off-grid pair's neighbour nearest them
fn iq2_kl_index(grid: &[(f32, f32); 32], id: f32, x: [f32; 2], w: [f32; 2]) -> usize {
    let (sx1, sx2) = (id * x[0], id * x[1]);
    let l1 = best_index_iq3nl(&IQ3NL_VALUES, sx1);
    let l2 = best_index_iq3nl(&IQ3NL_VALUES, sx2);
    let i = IQ2_KL_INDEX[8 * l1 + l2];
    if i >= 0 {
        return i as usize;
    }
    let (mut best, mut ibest) = (f32::MAX, usize::MAX);
    for &n in &IQ2_KL_NEIGHBOURS[(-i - 1) as usize] {
        let (g1, g2) = grid[usize::from(n)];
        let (diff1, diff2) = (g1 - sx1, g2 - sx2);
        let score = fma(w[0] * diff1, diff1, w[1] * diff2 * diff2);
        if score < best {
            (best, ibest) = (score, usize::from(n));
        }
    }
    ibest
}

// a pair's two products onto the sum as g++ fuses them: the first into the second, the sum added after
fn pair_sum(sum: f32, a: [f32; 2], b: [f32; 2]) -> f32 {
    sum + fma(a[0], b[0], a[1] * b[1])
}

/// One IQ2_KL row (`quantize_row_iq2_kl_impl`): per 32 values a scale over pairs of the 3-bit codebook (both signs,
/// eleven steps); 6-bit multiples of the row's largest over 32, each the best of it and its neighbours; the row
/// scale refit to the pair codes.
fn quantize_iq2_kl_row(x: &[f32], qw: Option<&[f32]>, out: &mut [u8]) {
    const BLOCK: usize = 32;
    let grid = iq2_kl_grid();
    let (header, blocks) = out.split_at_mut(F16_ROW_SCALE);
    header.fill(0);
    blocks.fill(0);
    let nblock = x.len() / QK_K;
    let mut all_scales = vec![0f32; nblock * QK_K / BLOCK];
    let mut weight = [0f32; BLOCK];
    let sigma2_of = |xbl: &[f32]| sum_squares(xbl) * (IQ2_KL_SIGMA_FACTOR / QK_K as f32);
    let weights_of = |ibl: usize, ib: usize, sigma2: f32, weight: &mut [f32; BLOCK]| {
        let xb = &x[ibl * QK_K + BLOCK * ib..][..BLOCK];
        for j in 0..BLOCK {
            weight[j] = match qw {
                Some(qw) => qw[ibl * QK_K + BLOCK * ib + j] * fma(xb[j], xb[j], sigma2).sqrt(),
                None => xb[j].abs(),
            };
        }
    };
    let values0 = f32::from(IQ3NL_VALUES[0]);
    let (mut max_scale, mut max_abs_scale) = (0f32, 0f32);
    for ibl in 0..nblock {
        let xbl = &x[ibl * QK_K..][..QK_K];
        let sigma2 = sigma2_of(xbl);
        for ib in 0..QK_K / BLOCK {
            let xb = &xbl[BLOCK * ib..][..BLOCK];
            weights_of(ibl, ib, sigma2, &mut weight);
            let (mut amax, mut max) = (0f32, 0f32);
            for &v in xb {
                if v.abs() > amax {
                    (amax, max) = (v.abs(), v);
                }
            }
            if amax < IQ2_K_EPS {
                continue;
            }
            // the fit at `id` and at `-id`
            let fits = |id: f32| {
                let (mut p, mut m) = ([0f32; 2], [0f32; 2]);
                for j in (0..BLOCK).step_by(2) {
                    let (xs, w) = ([xb[j], xb[j + 1]], [weight[j], weight[j + 1]]);
                    for (sign, sums) in [(1.0, &mut p), (-1.0, &mut m)] {
                        let (q1, q2) = grid[iq2_kl_index(&grid, sign * id, xs, w)];
                        let wq = [w[0] * q1, w[1] * q2];
                        sums[0] = pair_sum(sums[0], wq, xs);
                        sums[1] = pair_sum(sums[1], wq, [q1, q2]);
                    }
                }
                (p, m)
            };
            let mut d = -max / values0;
            let (p, m) = fits(1.0 / d);
            d = p[0] / p[1];
            let mut best = d * p[0];
            if m[1] > 0.0 && m[0] * m[0] > best * m[1] {
                d = m[0] / m[1];
                best = d * m[0];
            }
            for itry in -IQ2_KL_TRIES..=IQ2_KL_TRIES {
                let (p, m) = fits((itry as f32 + values0) / max);
                for s in [p, m] {
                    if s[1] > 0.0 && s[0] * s[0] > best * s[1] {
                        d = s[0] / s[1];
                        best = d * s[0];
                    }
                }
            }
            all_scales[ibl * 8 + ib] = d;
            if d.abs() > max_abs_scale {
                (max_abs_scale, max_scale) = (d.abs(), d);
            }
        }
    }
    if max_abs_scale == 0.0 {
        return;
    }
    let mut d = -max_scale / 32.0;
    let id = 1.0 / d;
    let (mut sumqx, mut sumq2) = (0f32, 0f32);
    for ibl in 0..nblock {
        let xbl = &x[ibl * QK_K..][..QK_K];
        let sigma2 = sigma2_of(xbl);
        let block = &mut blocks[ibl * BLOCK_IQ2_KL..][..BLOCK_IQ2_KL];
        for ib in 0..QK_K / BLOCK {
            let xb = &xbl[BLOCK * ib..][..BLOCK];
            weights_of(ibl, ib, sigma2, &mut weight);
            let ls = nearest_int(id * all_scales[ibl * 8 + ib]).clamp(-32, 31);
            let (mut best_score, mut best_ls) = (f32::MAX, ls);
            for ils in (ls - 1).max(-32)..=(ls + 1).min(31) {
                let dl = d * ils as f32;
                let idl = if dl != 0.0 { 1.0 / dl } else { 0.0 };
                let mut score = 0f32;
                for j in 0..BLOCK / 2 {
                    let (xs, w) = (
                        [xb[2 * j], xb[2 * j + 1]],
                        [weight[2 * j], weight[2 * j + 1]],
                    );
                    let (g1, g2) = grid[iq2_kl_index(&grid, idl, xs, w)];
                    let diff1 = fma(dl, g1, -xs[0]);
                    let diff2 = fma(dl, g2, -xs[1]);
                    score += fma(w[0] * diff1, diff1, w[1] * diff2 * diff2);
                }
                if score < best_score {
                    (best_score, best_ls) = (score, ils);
                }
            }
            let ls = best_ls;
            let uls = (ls + 32) as u16;
            block[2 + ib % 4] |= ((uls & 0xf) as u8) << (4 * (ib / 4));
            let scales_h = u16::from_le_bytes([block[0], block[1]]) | ((uls >> 4) << (2 * ib));
            block[..2].copy_from_slice(&scales_h.to_le_bytes());
            if ls == 0 {
                continue;
            }
            let dl = d * ls as f32;
            let idl = 1.0 / dl;
            for j in 0..BLOCK / 2 {
                let (xs, w) = (
                    [xb[2 * j], xb[2 * j + 1]],
                    [weight[2 * j], weight[2 * j + 1]],
                );
                let idx = iq2_kl_index(&grid, idl, xs, w);
                block[6 + 16 * (ib / 2) + j] |= ((idx & 0xf) as u8) << (4 * (ib % 2));
                block[70 + j] |= ((idx >> 4) as u8) << ib;
                let (g1, g2) = grid[idx];
                let (q1, q2) = (ls as f32 * g1, ls as f32 * g2);
                let wq = [w[0] * q1, w[1] * q2];
                sumqx = pair_sum(sumqx, wq, xs);
                sumq2 = pair_sum(sumq2, wq, [q1, q2]);
            }
        }
    }
    if sumq2 > 0.0 {
        d = sumqx / sumq2;
    }
    header.copy_from_slice(&half::f16::from_f32(IQ2_KL_FUDGE * d).to_le_bytes());
}

// `best_index_avx2`: the value nearest `id * x` and nearest its negation, from the simple index table and the next
// code up, as each lane picks them
fn simple_values(id: f32, x: f32, values: &[i8], table: &[u8]) -> [f32; 2] {
    let pick = |v: f32| {
        let ix = ((v as i32) - i32::from(values[0])).clamp(0, table.len() as i32 - 1) as usize;
        let t0 = usize::from(table[ix]);
        let (c0, c1) = (f32::from(values[t0]), f32::from(values[t0 + 1]));
        if v - c0 < c1 - v { c0 } else { c1 }
    };
    let v = id * x;
    [pick(v), pick(-v)]
}

// `evaluate_block_avx2` over 16 values: the fit at `id` and at `-id`, keeping the better of either over `best`
fn evaluate_block(
    id: f32,
    xb: &[f32],
    w: &[f32],
    values: &[i8],
    table: &[u8],
    d: &mut f32,
    best: &mut f32,
) -> bool {
    let mut lanes = [[0f32; LANES]; 4];
    for i in 0..xb.len() / LANES {
        for l in 0..LANES {
            let j = LANES * i + l;
            let [qp, qm] = simple_values(id, xb[j], values, table);
            let (wp, wm) = (w[j] * qp, w[j] * qm);
            lanes[0][l] = fma(wp, xb[j], lanes[0][l]);
            lanes[1][l] = fma(wp, qp, lanes[1][l]);
            lanes[2][l] = fma(wm, xb[j], lanes[2][l]);
            lanes[3][l] = fma(wm, qm, lanes[3][l]);
        }
    }
    let [sqx_p, sq2_p, sqx_m, sq2_m] = lanes.map(|l| hsum8(&l));
    let mut updated = false;
    for (sqx, sq2) in [(sqx_p, sq2_p), (sqx_m, sq2_m)] {
        if sq2 > 0.0 && sqx * sqx > sq2 * *best {
            *d = sqx / sq2;
            *best = *d * sqx;
            updated = true;
        }
    }
    updated
}

// `find_best_scale_avx2` over the 3-bit codebooks: the max-based scale, then `tries` steps of `step` either side for
// each codebook; the block's scale and whether it took the shifted codebook
fn find_best_scale(xb: &[f32], w: &[f32], max: f32, tries: i32, step: f32) -> (f32, bool) {
    let (values, shifted) = IQ3NL_VALUES.split_at(8);
    let table = &SIMPLE_INDEX_IQ3NL;
    let v0 = f32::from(values[0]);
    let mut d = -max / v0;
    let mut best = 0f32;
    let mut is_shifted = false;
    evaluate_block(1.0 / d, xb, w, values, table, &mut d, &mut best);
    for itry in -tries..=tries {
        let id = (step * itry as f32 + v0) / max;
        if evaluate_block(id, xb, w, values, table, &mut d, &mut best) {
            is_shifted = false;
        }
        let id = (step * itry as f32 + f32::from(shifted[0])) / max;
        if evaluate_block(id, xb, w, shifted, table, &mut d, &mut best) {
            is_shifted = true;
        }
    }
    (d, is_shifted)
}

/// One IQ3_K super-block: per 16 values a signed 4-bit odd scale and a codebook pick (`find_best_scale_avx2`),
/// 3-bit indices.
fn quantize_iq3_k(xbl: &[f32], qw: Option<&[f32]>, out: &mut [u8]) {
    const BLOCK: usize = 16;
    out.fill(0);
    let (values, shifted) = IQ3NL_VALUES.split_at(8);
    let sigma2 = 1.5 * sum_squares(xbl) / QK_K as f32;
    let mut scales = [0f32; QK_K / BLOCK];
    let mut weight = [0f32; BLOCK];
    let mut extra = 0u16;
    let mut max_abs_scale = 0f32;
    for ib in 0..QK_K / BLOCK {
        let xb = &xbl[BLOCK * ib..][..BLOCK];
        iq2_k_weights(
            xb,
            qw.map(|w| &w[BLOCK * ib..][..BLOCK]),
            sigma2,
            &mut weight,
        );
        let (mut amax, mut max) = (0f32, 0f32);
        for &v in xb {
            if v.abs() > amax {
                (amax, max) = (v.abs(), v);
            }
        }
        if amax < IQ2_K_EPS {
            continue;
        }
        let (d, is_shifted) = find_best_scale(xb, &weight, max, IQ3_K_TRIES, IQ3_K_STEP);
        scales[ib] = d;
        if is_shifted {
            extra |= 1 << ib;
        }
        max_abs_scale = max_abs_scale.max(d.abs());
    }
    if max_abs_scale == 0.0 {
        return;
    }
    let d = max_abs_scale / 31.0;
    out[2..4].copy_from_slice(&extra.to_le_bytes());
    let id = 1.0 / d;
    let (mut sumqx, mut sumq2) = (0f32, 0f32);
    let mut scales_h = 0u16;
    for ib in 0..QK_K / BLOCK {
        let code = nearest_int(0.5 * fma(id, scales[ib].abs(), -1.0)).clamp(0, 15);
        out[6 + ib / 2] |= (code as u8) << (4 * (ib % 2));
        let negative = scales[ib] < 0.0;
        if negative {
            scales_h |= 1 << ib;
        }
        let ls = (2 * code + 1) * if negative { -1 } else { 1 };
        let dl = d * ls as f32;
        if dl == 0.0 {
            continue;
        }
        let codebook = if extra & (1 << ib) != 0 {
            shifted
        } else {
            values
        };
        let xb = &xbl[BLOCK * ib..][..BLOCK];
        iq2_k_weights(
            xb,
            qw.map(|w| &w[BLOCK * ib..][..BLOCK]),
            sigma2,
            &mut weight,
        );
        let idl = 1.0 / dl;
        let ib32 = ib / 2;
        let offset = 16 * (ib % 2);
        for j in 0..BLOCK {
            let ibest = best_index_iq3nl(codebook, idl * xb[j]);
            out[14 + 32 * (ib32 / 4) + offset + j] |= ((ibest & 3) as u8) << (2 * (ib32 % 4));
            out[78 + offset + j] |= ((ibest >> 2) as u8) << (ib32 % 8);
            let q = (i32::from(codebook[ibest]) * ls) as f32;
            sumqx = fma(weight[j] * q, xb[j], sumqx);
            sumq2 = fma(weight[j] * q, q, sumq2);
        }
    }
    out[4..6].copy_from_slice(&scales_h.to_le_bytes());
    let scale = if sumq2 > 0.0 { sumqx / sumq2 } else { d };
    out[..2].copy_from_slice(&half::f16::from_f32(IQ3_K_FUDGE * scale).to_le_bytes());
}

/// One IQ3_KS row (`quantize_row_iq3_ks_impl`, AVX2 search): per 32 values a scale and codebook pick, the scales as
/// 5-bit multiples of the row's largest over 16, the row scale refit to the indices.
fn quantize_iq3_ks_row(x: &[f32], qw: Option<&[f32]>, out: &mut [u8]) {
    const BLOCK: usize = 32;
    let (header, blocks) = out.split_at_mut(F16_ROW_SCALE);
    header.fill(0);
    blocks.fill(0);
    let (values, shifted) = IQ3NL_VALUES.split_at(8);
    let nblock = x.len() / QK_K;
    let mut all_scales = vec![0f32; nblock * QK_K / BLOCK];
    let mut weight = [0f32; BLOCK];
    let sigma2_of = |xbl: &[f32]| sum_squares(xbl) * (2.0 / QK_K as f32);
    let weights_of = |ibl: usize, ib: usize, sigma2: f32, weight: &mut [f32; BLOCK]| {
        let xb = &x[ibl * QK_K + BLOCK * ib..][..BLOCK];
        for j in 0..BLOCK {
            weight[j] = match qw {
                Some(qw) => qw[ibl * QK_K + BLOCK * ib + j] * fma(xb[j], xb[j], sigma2).sqrt(),
                None => xb[j] * xb[j],
            };
        }
    };
    let (mut amax_scale, mut max_scale) = (0f32, 0f32);
    for ibl in 0..nblock {
        let xbl = &x[ibl * QK_K..][..QK_K];
        let sigma2 = sigma2_of(xbl);
        let mut extra = 0u16;
        for ib in 0..QK_K / BLOCK {
            let xb = &xbl[BLOCK * ib..][..BLOCK];
            weights_of(ibl, ib, sigma2, &mut weight);
            let (mut amax, mut max) = (0f32, 0f32);
            for &v in xb {
                if v.abs() > amax {
                    (amax, max) = (v.abs(), v);
                }
            }
            if amax < IQ2_K_EPS {
                continue;
            }
            let (d, is_shifted) = find_best_scale(xb, &weight, max, IQ3_KS_TRIES, 1.0);
            if is_shifted {
                extra |= 1 << (8 + ib);
            }
            all_scales[ibl * 8 + ib] = d;
            if d.abs() > amax_scale {
                (amax_scale, max_scale) = (d.abs(), d);
            }
        }
        blocks[ibl * BLOCK_IQ3_KS..][..2].copy_from_slice(&extra.to_le_bytes());
    }
    let d = -max_scale / 16.0;
    header.copy_from_slice(&half::f16::from_f32(d).to_le_bytes());
    if d == 0.0 {
        return;
    }
    let id = 1.0 / d;
    let (mut sumqx, mut sumq2) = (0f32, 0f32);
    for ibl in 0..nblock {
        let xbl = &x[ibl * QK_K..][..QK_K];
        let sigma2 = sigma2_of(xbl);
        let block = &mut blocks[ibl * BLOCK_IQ3_KS..][..BLOCK_IQ3_KS];
        for ib in 0..QK_K / BLOCK {
            let mut extra = u16::from_le_bytes([block[0], block[1]]);
            let codebook = if (extra >> (8 + ib)) & 1 != 0 {
                shifted
            } else {
                values
            };
            let l = nearest_int(id * all_scales[ibl * 8 + ib]).clamp(-16, 15);
            let ul = (l + 16) as u8;
            block[2 + ib % 4] |= (ul & 0xf) << (4 * (ib / 4));
            extra |= u16::from(ul >> 4) << ib;
            block[..2].copy_from_slice(&extra.to_le_bytes());
            let dl = d * l as f32;
            let idl = if dl != 0.0 { 1.0 / dl } else { 0.0 };
            let xb = &xbl[BLOCK * ib..][..BLOCK];
            weights_of(ibl, ib, sigma2, &mut weight);
            for j in 0..BLOCK {
                let i = best_index_iq3nl(codebook, idl * xb[j]);
                block[6 + (ib / 4) * BLOCK + j] |= ((i & 3) as u8) << (2 * (ib % 4));
                block[70 + j] |= ((i >> 2) as u8) << (ib % 8);
                let q = (i32::from(codebook[i]) * l) as f32;
                sumqx = fma(weight[j] * q, xb[j], sumqx);
                sumq2 = fma(weight[j] * q, q, sumq2);
            }
        }
    }
    if sumq2 > 0.0 {
        header.copy_from_slice(&half::f16::from_f32(sumqx / sumq2).to_le_bytes());
    }
}

/// One row of `ty`, the imatrix reweighed as `QHelper` does for the type's block size.
pub(super) fn quantize_row(ty: GgufType, row: &[f32], imatrix: Option<&[f32]>, out: &mut [u8]) {
    match ty {
        GgufType::Iq2K => {
            let weights = imatrix.map(|w| row_weights(row, w, 16));
            let blocks = out.as_chunks_mut::<BLOCK_IQ2_K>().0;
            for (i, (xbl, o)) in row.as_chunks::<QK_K>().0.iter().zip(blocks).enumerate() {
                quantize_iq2_k(xbl, weights.as_deref().map(|w| &w[i * QK_K..][..QK_K]), o);
            }
        }
        GgufType::Iq2Ks => {
            let weights = imatrix.map(|w| row_weights(row, w, 32));
            quantize_iq2_ks_row(row, weights.as_deref(), out);
        }
        GgufType::Iq2Kl => {
            let weights = imatrix.map(|w| row_weights(row, w, 32));
            quantize_iq2_kl_row(row, weights.as_deref(), out);
        }
        GgufType::Iq3K => {
            let weights = imatrix.map(|w| row_weights(row, w, 16));
            let blocks = out.as_chunks_mut::<BLOCK_IQ3_K>().0;
            for (i, (xbl, o)) in row.as_chunks::<QK_K>().0.iter().zip(blocks).enumerate() {
                quantize_iq3_k(xbl, weights.as_deref().map(|w| &w[i * QK_K..][..QK_K]), o);
            }
        }
        GgufType::Iq3Ks => {
            let weights = imatrix.map(|w| row_weights(row, w, 32));
            quantize_iq3_ks_row(row, weights.as_deref(), out);
        }
        other => unreachable!("{other:?} is not an ik table type"),
    }
}
