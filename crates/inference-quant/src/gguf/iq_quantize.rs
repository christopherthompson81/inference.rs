//! The non-linear 4-bit quantizers, with the importance weights an imatrix gives: ggml's IQ4_NL / IQ4_XS
//! (`quantize_iq4_nl` / `quantize_iq4_xs` in ggml-quants.c) and ik_llama.cpp's IQ4_KS (`quantize_iq4_ks` in
//! iqk_quantize.cpp, as its AVX2 build runs it). Rows of floats to raw GGUF blocks, as the dequantizers in `raw` read
//! them.

use inference_tensor::Result;
use rayon::prelude::*;

use super::iq_grid_quantize as grid;
use super::kernel::GgufType;
use super::raw::KVALUES_IQ4NL;

const QK4_NL: usize = 32;
const QK_K: usize = 256;
const BLOCK_IQ4_NL: usize = 2 + QK4_NL / 2;
const BLOCK_IQ4_XS: usize = 2 + 2 + QK_K / 64 + QK_K / 2;
// ggml's floor below which a group is zero
const GROUP_MAX_EPS: f32 = 1e-15;
// the scale search tries this many steps either side of the max-based scale
const SCALE_TRIES: i32 = 7;
// IQ4_XS sub-block scales are 6-bit, offset by 32
const SCALE_OFFSET: i32 = 32;
const ROUNDING_MAGIC: f32 = 12_582_912.0;
// IQ4_KS: a row-wide f32 scale, then per 256 values eight 7-bit block scales (the low bit picks the shifted codebook)
// and 128 index bytes
const BLOCK_IQ4_KS: usize = QK_K / QK4_NL + QK_K / 2;
const IQ4_KS_ROW_HEADER: usize = 4;
const IQ4_KS_EPS: f32 = 1e-16;
const IQ4_KS_SCALE_MAX: i32 = 127;
// ik_llama.cpp's default fudge factor for the row scale
const IQ4_KS_FUDGE: f32 = 1.0;
// an imatrix block is replaced by this where it, or the values it weighs, are all but zero
const IMATRIX_EPS: f32 = 1e-9;
const SIMD_LANES: usize = 8;
// ik_llama.cpp's codebooks: IQ4_NL's, then the same shifted
const IQ4K_VALUES: [i8; 32] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113, -123, -100, -79, -61,
    -45, -31, -18, -6, 5, 17, 29, 42, 57, 73, 93, 117,
];
// `iq4nl_index`: from `trunc(x) - values[0]`, the index (or 16 + the lower of two to compare)
const IQ4NL_INDEX: [i8; 241] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 17, 17, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 18, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 19, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 20, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 21, 21, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 22, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 23,
    23, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 24, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 25, 10, 10, 10, 10, 10,
    10, 10, 10, 10, 10, 10, 26, 26, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 27, 27, 12, 12,
    12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 28, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
    13, 13, 13, 13, 13, 13, 29, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
    14, 14, 14, 14, 30, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
];
// `k_simple_index_iq4nl`, which the AVX2 search reads for either codebook
const SIMPLE_INDEX_IQ4NL: [u8; 256] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10,
    10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12,
    12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
    13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
    14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
    14, 14,
];

// ggml's: the float added to 1.5 * 2^23 rounds to the nearest integer, ties to even
pub(super) fn nearest_int(x: f32) -> i32 {
    let v = x + ROUNDING_MAGIC;
    (v.to_bits() & 0x007f_ffff) as i32 - 0x0040_0000
}

// the codebook entry nearest `x` (ggml's `best_index_int8`: binary search, then the closer neighbour)
fn best_index(values: &[i8; 16], x: f32) -> usize {
    if x <= f32::from(values[0]) {
        return 0;
    }
    if x >= f32::from(values[15]) {
        return 15;
    }
    let (mut lo, mut hi) = (0usize, 15usize);
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if x < f32::from(values[mid]) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    if x - f32::from(values[lo]) < f32::from(values[hi]) - x {
        lo
    } else {
        hi
    }
}

/// One super-block of `x` (`quantize_row_iq4_nl_impl`): per 32-value block a scale fit to the importance weights;
/// for a super-block of several blocks those scales then share one f16 `d` as 6-bit multiples. Returns `(d, scales
/// code)` with the block's 4-bit indices in `indices`.
fn quantize_super_block(
    x: &[f32],
    quant_weights: Option<&[f32]>,
    indices: &mut [u8],
    scales_h: &mut u16,
    scales_l: &mut [u8],
) -> f32 {
    let values = &KVALUES_IQ4NL;
    let sigma2 = 2.0 * x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let blocks = x.len() / QK4_NL;
    let mut scales = vec![0f32; blocks];
    let mut weight = [0f32; QK4_NL];
    let (mut max_scale, mut amax_scale) = (0f32, 0f32);
    for ib in 0..blocks {
        let xb = &x[ib * QK4_NL..(ib + 1) * QK4_NL];
        let lb = &mut indices[ib * QK4_NL..(ib + 1) * QK4_NL];
        match quant_weights {
            Some(qw) => {
                let qw = &qw[ib * QK4_NL..(ib + 1) * QK4_NL];
                for j in 0..QK4_NL {
                    weight[j] = qw[j] * (sigma2 + xb[j] * xb[j]).sqrt();
                }
            }
            None => {
                for j in 0..QK4_NL {
                    weight[j] = xb[j] * xb[j];
                }
            }
        }
        let (mut amax, mut max) = (0f32, 0f32);
        for &v in xb {
            if v.abs() > amax {
                amax = v.abs();
                max = v;
            }
        }
        if amax < GROUP_MAX_EPS {
            scales[ib] = 0.0;
            continue;
        }
        let fit = |inverse: f32, lb: Option<&mut [u8]>| {
            let (mut sumqx, mut sumq2) = (0f32, 0f32);
            let mut lb = lb;
            for j in 0..QK4_NL {
                let l = best_index(values, inverse * xb[j]);
                if let Some(lb) = lb.as_deref_mut() {
                    lb[j] = l as u8;
                }
                let q = f32::from(values[l]);
                sumqx += weight[j] * q * xb[j];
                sumq2 += weight[j] * q * q;
            }
            (sumqx, sumq2)
        };
        let d = -max / f32::from(values[0]);
        let (sumqx, sumq2) = fit(1.0 / d, Some(lb));
        let mut d = if sumq2 > 0.0 { sumqx / sumq2 } else { 0.0 };
        let mut best = d * sumqx;
        for itry in -SCALE_TRIES..=SCALE_TRIES {
            let inverse = (itry as f32 + f32::from(values[0])) / max;
            let (sumqx, sumq2) = fit(inverse, None);
            if sumq2 > 0.0 && sumqx * sumqx > best * sumq2 {
                d = sumqx / sumq2;
                best = d * sumqx;
            }
        }
        scales[ib] = d;
        if d.abs() > amax_scale {
            amax_scale = d.abs();
            max_scale = d;
        }
    }
    if blocks == 1 {
        // IQ4_NL: the block's own scale, its indices refit to it
        let d = scales[0];
        let inverse = if d != 0.0 { 1.0 / d } else { 0.0 };
        for (l, &v) in indices.iter_mut().zip(x) {
            *l = best_index(values, inverse * v) as u8;
        }
        return d;
    }
    let d = -max_scale / SCALE_OFFSET as f32;
    let inverse = if d != 0.0 { 1.0 / d } else { 0.0 };
    *scales_h = 0;
    scales_l.fill(0);
    for ib in 0..blocks {
        let l = nearest_int(inverse * scales[ib]).clamp(-SCALE_OFFSET, SCALE_OFFSET - 1);
        let dl = d * l as f32;
        let idl = if dl != 0.0 { 1.0 / dl } else { 0.0 };
        let xb = &x[ib * QK4_NL..(ib + 1) * QK4_NL];
        for (j, &v) in xb.iter().enumerate() {
            indices[ib * QK4_NL + j] = best_index(values, idl * v) as u8;
        }
        let code = (l + SCALE_OFFSET) as u8;
        scales_l[ib / 2] |= (code & 0x0F) << (4 * (ib % 2));
        *scales_h |= u16::from(code >> 4) << (2 * ib);
    }
    d
}

// 32 indices to 16 bytes: the first 16 in the low nibbles, the next 16 in the high ones
fn pack_indices(indices: &[u8], qs: &mut [u8]) {
    for (group, out) in indices
        .as_chunks::<QK4_NL>()
        .0
        .iter()
        .zip(qs.as_chunks_mut::<{ QK4_NL / 2 }>().0)
    {
        for j in 0..QK4_NL / 2 {
            out[j] = group[j] | (group[QK4_NL / 2 + j] << 4);
        }
    }
}

fn quantize_iq4_nl_block(x: &[f32], qw: Option<&[f32]>, out: &mut [u8]) {
    let mut indices = [0u8; QK4_NL];
    let d = quantize_super_block(x, qw, &mut indices, &mut 0, &mut []);
    out[..2].copy_from_slice(&half::f16::from_f32(d).to_le_bytes());
    pack_indices(&indices, &mut out[2..]);
}

fn quantize_iq4_xs_block(x: &[f32], qw: Option<&[f32]>, out: &mut [u8]) {
    let mut indices = [0u8; QK_K];
    let (mut scales_h, mut scales_l) = (0u16, [0u8; QK_K / 64]);
    let d = quantize_super_block(x, qw, &mut indices, &mut scales_h, &mut scales_l);
    out[..2].copy_from_slice(&half::f16::from_f32(d).to_le_bytes());
    out[2..4].copy_from_slice(&scales_h.to_le_bytes());
    out[4..8].copy_from_slice(&scales_l);
    pack_indices(&indices, &mut out[8..]);
}

// ik_llama.cpp's `best_index_iq4nl` over a 16-value codebook
fn best_index_iq4k(values: &[i8], x: f32) -> usize {
    let ix = x as i32 - i32::from(values[0]);
    if !(0..241).contains(&ix) {
        return if ix < 0 { 0 } else { 15 };
    }
    let ix = IQ4NL_INDEX[ix as usize] as usize;
    if ix < 16 {
        return ix;
    }
    if x - f32::from(values[ix - 16]) < f32::from(values[ix - 15]) - x {
        ix - 16
    } else {
        ix - 15
    }
}

// `hsum_float_8`'s order: the halves added, then pairs within them
fn lanes_sum(lanes: &[f32; SIMD_LANES]) -> f32 {
    let half = [
        lanes[0] + lanes[4],
        lanes[1] + lanes[5],
        lanes[2] + lanes[6],
        lanes[3] + lanes[7],
    ];
    (half[0] + half[2]) + (half[1] + half[3])
}

/// One scale try over a 32-value block (`evaluate_block_iq4k`): the codebook values nearest `inverse * x` and
/// `-inverse * x` (the AVX2 table search), their weighted fits summed lane by lane; `(d, best)` updated when either
/// fits better.
fn evaluate_iq4k_block(
    inverse: f32,
    values: &[i8],
    xb: &[f32],
    weight: &[f32],
    d: &mut f32,
    best: &mut f32,
) -> bool {
    let candidate = |xi: f32| -> f32 {
        let ix = ((xi as i32) - i32::from(values[0])).clamp(0, 240) as usize;
        let lo = SIMPLE_INDEX_IQ4NL[ix] as usize;
        let (c0, c1) = (f32::from(values[lo]), f32::from(values[lo + 1]));
        if xi - c0 < c1 - xi { c0 } else { c1 }
    };
    let mut sums = [[0f32; SIMD_LANES]; 4];
    for (j, (&x, &w)) in xb.iter().zip(weight).enumerate() {
        let lane = j % SIMD_LANES;
        let xi = inverse * x;
        let (qp, qm) = (candidate(xi), candidate(-xi));
        let (wp, wm) = (w * qp, w * qm);
        sums[0][lane] = wp.mul_add(x, sums[0][lane]);
        sums[1][lane] = wp.mul_add(qp, sums[1][lane]);
        sums[2][lane] = wm.mul_add(x, sums[2][lane]);
        sums[3][lane] = wm.mul_add(qm, sums[3][lane]);
    }
    let [sqx_p, sq2_p, sqx_m, sq2_m] = sums.map(|lanes| lanes_sum(&lanes));
    let mut updated = false;
    if sq2_p > 0.0 && sqx_p * sqx_p > sq2_p * *best {
        *d = sqx_p / sq2_p;
        *best = *d * sqx_p;
        updated = true;
    }
    if sq2_m > 0.0 && sqx_m * sqx_m > sq2_m * *best {
        *d = sqx_m / sq2_m;
        *best = *d * sqx_m;
        updated = true;
    }
    updated
}

// `QHelper::row_weights`: an imatrix block weighing (or over) all but zero values is replaced by a tiny constant
fn iq4_ks_row_weights(x: &[f32], imatrix: &[f32]) -> Vec<f32> {
    let floor = QK4_NL as f32 * IMATRIX_EPS * IMATRIX_EPS;
    let mut out = imatrix.to_vec();
    for (wb, xb) in out
        .as_chunks_mut::<QK4_NL>()
        .0
        .iter_mut()
        .zip(x.as_chunks::<QK4_NL>().0)
    {
        let sumw2 = wb.iter().map(|w| w * w).sum::<f32>();
        let sumx2 = xb.iter().map(|v| v * v).sum::<f32>();
        let sumwx = wb.iter().zip(xb).map(|(w, v)| w * v.abs()).sum::<f32>();
        if !(sumw2 > floor && sumx2 > floor && sumwx > floor) {
            wb.fill(IMATRIX_EPS);
        }
    }
    out
}

fn block_weights(xb: &[f32], qw: Option<&[f32]>, sigma2: f32, weight: &mut [f32; QK4_NL]) {
    match qw {
        Some(qw) => {
            for j in 0..QK4_NL {
                weight[j] = qw[j] * xb[j].mul_add(xb[j], sigma2).sqrt();
            }
        }
        None => {
            for j in 0..QK4_NL {
                weight[j] = xb[j] * xb[j];
            }
        }
    }
}

fn sigma2(x: &[f32]) -> f32 {
    x.iter().fold(0f32, |s, v| v.mul_add(*v, s)) * (2.0 / x.len() as f32)
}

/// One IQ4_KS row (`quantize_row_iq4_k_impl_bs32`): per 32-value block the best scale over both codebooks; the
/// scales as 7-bit multiples of the row's largest over 127; then the row scale refit to the indices.
fn quantize_iq4_ks_row(x: &[f32], imatrix: Option<&[f32]>, out: &mut [u8]) {
    let weights = imatrix.map(|w| iq4_ks_row_weights(x, w));
    let (header, blocks) = out.split_at_mut(IQ4_KS_ROW_HEADER);
    blocks.fill(0);
    let mut all_scales = vec![0f32; x.len() / QK4_NL];
    let mut weight = [0f32; QK4_NL];
    let mut amax_scale = 0f32;
    let (values, shifted) = IQ4K_VALUES.split_at(16);
    for (ibl, (xbl, block)) in x
        .as_chunks::<QK_K>()
        .0
        .iter()
        .zip(blocks.as_chunks_mut::<BLOCK_IQ4_KS>().0)
        .enumerate()
    {
        let s2 = sigma2(xbl);
        for ib in 0..QK_K / QK4_NL {
            let xb = &xbl[ib * QK4_NL..(ib + 1) * QK4_NL];
            let qw = weights
                .as_deref()
                .map(|w| &w[ibl * QK_K + ib * QK4_NL..ibl * QK_K + (ib + 1) * QK4_NL]);
            block_weights(xb, qw, s2, &mut weight);
            let (mut amax, mut max) = (0f32, 0f32);
            for &v in xb {
                if v.abs() > amax {
                    amax = v.abs();
                    max = v;
                }
            }
            let scale = &mut all_scales[ibl * (QK_K / QK4_NL) + ib];
            if amax < IQ4_KS_EPS {
                *scale = 0.0;
                continue;
            }
            let mut d = -max / f32::from(values[0]);
            let mut best = 0f32;
            let mut is_shifted = false;
            evaluate_iq4k_block(1.0 / d, values, xb, &weight, &mut d, &mut best);
            for itry in -SCALE_TRIES..=SCALE_TRIES {
                let inverse = (itry as f32 + f32::from(values[0])) / max;
                if evaluate_iq4k_block(inverse, values, xb, &weight, &mut d, &mut best) {
                    is_shifted = false;
                }
                let inverse = (itry as f32 + f32::from(shifted[0])) / max;
                if evaluate_iq4k_block(inverse, shifted, xb, &weight, &mut d, &mut best) {
                    is_shifted = true;
                }
            }
            if is_shifted {
                block[ib] = 0x01;
            }
            *scale = d;
            amax_scale = amax_scale.max(d.abs());
        }
    }
    let d = amax_scale / IQ4_KS_SCALE_MAX as f32;
    header.copy_from_slice(&(IQ4_KS_FUDGE * d).to_le_bytes());
    if d == 0.0 {
        return;
    }
    let id = 1.0 / d;
    let (mut sumqx, mut sumq2) = (0f32, 0f32);
    for (ibl, (xbl, block)) in x
        .as_chunks::<QK_K>()
        .0
        .iter()
        .zip(blocks.as_chunks_mut::<BLOCK_IQ4_KS>().0)
        .enumerate()
    {
        let s2 = sigma2(xbl);
        let (scales, qs) = block.split_at_mut(QK_K / QK4_NL);
        for ib in 0..QK_K / QK4_NL {
            let codebook = if scales[ib] & 0x01 != 0 {
                shifted
            } else {
                values
            };
            let scale = all_scales[ibl * (QK_K / QK4_NL) + ib];
            let l = nearest_int(0.5 * (id * scale + IQ4_KS_SCALE_MAX as f32))
                .clamp(0, IQ4_KS_SCALE_MAX)
                << 1;
            scales[ib] |= l as u8;
            let l = l - IQ4_KS_SCALE_MAX;
            let dl = d * l as f32;
            let idl = if dl != 0.0 { 1.0 / dl } else { 0.0 };
            let xb = &xbl[ib * QK4_NL..(ib + 1) * QK4_NL];
            let qw = weights
                .as_deref()
                .map(|w| &w[ibl * QK_K + ib * QK4_NL..ibl * QK_K + (ib + 1) * QK4_NL]);
            block_weights(xb, qw, s2, &mut weight);
            let half = QK4_NL / 2;
            let qs = &mut qs[ib * half..(ib + 1) * half];
            for j in 0..half {
                let i1 = best_index_iq4k(codebook, idl * xb[j]);
                let i2 = best_index_iq4k(codebook, idl * xb[j + half]);
                qs[j] = i1 as u8 | ((i2 as u8) << 4);
                let (q1, q2) = (
                    f32::from(codebook[i1]) * l as f32,
                    f32::from(codebook[i2]) * l as f32,
                );
                // as g++ contracts it: the first product fused over the second
                let (w1, w2) = (weight[j] * q1, weight[j + half] * q2);
                sumqx += w1.mul_add(xb[j], w2 * xb[j + half]);
                sumq2 += w1.mul_add(q1, w2 * q2);
            }
        }
    }
    if sumq2 > 0.0 {
        header.copy_from_slice(&(IQ4_KS_FUDGE * sumqx / sumq2).to_le_bytes());
    }
}

type BlockEncoder = fn(&[f32], Option<&[f32]>, &mut [u8]);

/// Whether [`quantize_rows`] writes `ty`.
pub fn can_quantize(ty: GgufType) -> bool {
    matches!(
        ty,
        GgufType::Iq4Nl
            | GgufType::Iq4Xs
            | GgufType::Iq4Ks
            | GgufType::Iq2Xxs
            | GgufType::Iq2Xs
            | GgufType::Iq2S
            | GgufType::Iq3Xxs
            | GgufType::Iq3S
            | GgufType::Iq1S
            | GgufType::Iq1M
    )
}

/// Whether `ty` needs an imatrix to quantize (ggml's `ggml_quantize_requires_imatrix`).
pub fn requires_imatrix(ty: GgufType) -> bool {
    matches!(ty, GgufType::Iq2Xxs | GgufType::Iq2Xs | GgufType::Iq1S)
}

/// Rows of `cols` floats as `ty` blocks, each column weighed by `imatrix` (its `cols` importances) when given.
pub fn quantize_rows(
    ty: GgufType,
    cols: usize,
    values: &[f32],
    imatrix: Option<&[f32]>,
) -> Result<Vec<u8>> {
    // IQ4_KS's scales span the row, so it encodes whole rows; the others encode block by block
    let (block, row_bytes, encode): (usize, usize, Option<BlockEncoder>) = match ty {
        GgufType::Iq4Nl => (
            QK4_NL,
            cols / QK4_NL * BLOCK_IQ4_NL,
            Some(quantize_iq4_nl_block),
        ),
        GgufType::Iq4Xs => (
            QK_K,
            cols / QK_K * BLOCK_IQ4_XS,
            Some(quantize_iq4_xs_block),
        ),
        GgufType::Iq4Ks => (QK_K, IQ4_KS_ROW_HEADER + cols / QK_K * BLOCK_IQ4_KS, None),
        GgufType::Iq2Xxs => (
            QK_K,
            cols / QK_K * grid::BLOCK_IQ2_XXS,
            Some(|x, w, o| grid::quantize_iq2_xxs(x, w.expect("an imatrix"), o)),
        ),
        GgufType::Iq2Xs => (
            QK_K,
            cols / QK_K * grid::BLOCK_IQ2_XS,
            Some(|x, w, o| grid::quantize_iq2_xs(x, w.expect("an imatrix"), o)),
        ),
        GgufType::Iq2S => (
            QK_K,
            cols / QK_K * grid::BLOCK_IQ2_S,
            Some(grid::quantize_iq2_s),
        ),
        GgufType::Iq3Xxs => (
            QK_K,
            cols / QK_K * grid::BLOCK_IQ3_XXS,
            Some(grid::quantize_iq3_xxs),
        ),
        GgufType::Iq3S => (
            QK_K,
            cols / QK_K * grid::BLOCK_IQ3_S,
            Some(grid::quantize_iq3_s),
        ),
        GgufType::Iq1S => (
            QK_K,
            cols / QK_K * grid::BLOCK_IQ1_S,
            Some(|x, w, o| grid::quantize_iq1_s(x, w.expect("an imatrix"), o)),
        ),
        GgufType::Iq1M => (
            QK_K,
            cols / QK_K * grid::BLOCK_IQ1_M,
            Some(grid::quantize_iq1_m),
        ),
        other => inference_tensor::bail!("{other:?} has no quantizer here"),
    };
    if requires_imatrix(ty) && imatrix.is_none() {
        inference_tensor::bail!("{ty:?} needs an imatrix to quantize");
    }
    if !cols.is_multiple_of(block) || !values.len().is_multiple_of(cols) {
        inference_tensor::bail!("{ty:?} rows of {cols} values are not whole blocks of {block}");
    }
    if let Some(w) = imatrix.filter(|w| w.len() != cols) {
        inference_tensor::bail!("an imatrix of {} entries for rows of {cols}", w.len());
    }
    let mut out = vec![0u8; values.len() / cols * row_bytes];
    out.par_chunks_exact_mut(row_bytes)
        .zip(values.par_chunks_exact(cols))
        .for_each(|(row_out, row)| match encode {
            None => quantize_iq4_ks_row(row, imatrix, row_out),
            Some(encode) => {
                let size = row_out.len() / (cols / block);
                for (i, (x, o)) in row
                    .chunks_exact(block)
                    .zip(row_out.chunks_exact_mut(size))
                    .enumerate()
                {
                    encode(x, imatrix.map(|w| &w[i * block..(i + 1) * block]), o);
                }
            }
        });
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn values(n: usize, seed: u32) -> Vec<f32> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((state >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 2.0
            })
            .collect()
    }

    fn rms_error(ty: GgufType, cols: usize, x: &[f32], imatrix: Option<&[f32]>) -> (f32, Vec<f32>) {
        let bytes = quantize_rows(ty, cols, x, imatrix).unwrap();
        let back = super::super::raw::dequantize_rows(ty, cols, &bytes).unwrap();
        let err = x
            .iter()
            .zip(&back)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / x.len() as f32;
        (err.sqrt(), back)
    }

    // a 4-bit codebook over uniform values keeps the error near a sixteenth of the range
    #[test]
    fn rows_round_trip_through_the_dequantizers() {
        for ty in [GgufType::Iq4Nl, GgufType::Iq4Xs, GgufType::Iq4Ks] {
            let x = values(4 * 512, 7);
            let (err, _) = rms_error(ty, 512, &x, None);
            assert!(err < 0.08, "{ty:?} round trip error {err}");
        }
    }

    #[test]
    fn an_imatrix_lowers_the_error_where_it_weighs() {
        let cols = 256;
        let x = values(64 * cols, 11);
        // the first half of the columns matters far more
        let imatrix = (0..cols)
            .map(|c| if c < cols / 2 { 100.0 } else { 0.01 })
            .collect::<Vec<_>>();
        let weighted_error = |back: &[f32]| {
            x.chunks(cols)
                .zip(back.chunks(cols))
                .map(|(a, b)| {
                    (0..cols)
                        .map(|c| imatrix[c] * (a[c] - b[c]).powi(2))
                        .sum::<f32>()
                })
                .sum::<f32>()
        };
        for ty in [GgufType::Iq4Nl, GgufType::Iq4Xs, GgufType::Iq4Ks] {
            let (_, plain) = rms_error(ty, cols, &x, None);
            let (_, weighted) = rms_error(ty, cols, &x, Some(&imatrix));
            assert!(
                weighted_error(&weighted) < weighted_error(&plain),
                "{ty:?}: the imatrix did not lower the weighted error"
            );
        }
    }

    const LOW_BIT: [GgufType; 7] = [
        GgufType::Iq1S,
        GgufType::Iq1M,
        GgufType::Iq2Xxs,
        GgufType::Iq2Xs,
        GgufType::Iq2S,
        GgufType::Iq3Xxs,
        GgufType::Iq3S,
    ];

    // uniform values have an rms of 0.58; each type's bound sits a little above what it reaches, fewer bits higher
    #[test]
    fn low_bit_rows_round_trip_through_the_dequantizers() {
        let bounds = [0.27, 0.26, 0.21, 0.18, 0.15, 0.11, 0.08];
        let cols = 512;
        let x = values(8 * cols, 3);
        let imatrix = vec![1.0f32; cols];
        for (ty, bound) in LOW_BIT.into_iter().zip(bounds) {
            let (err, _) = rms_error(ty, cols, &x, Some(&imatrix));
            assert!(err < bound, "{ty:?} round trip error {err}");
        }
    }

    // a skewed imatrix against an even one: the error it weighs falls
    #[test]
    fn a_low_bit_imatrix_lowers_the_error_where_it_weighs() {
        let cols = 256;
        let x = values(64 * cols, 11);
        // every other column: a weight that is constant over a sub-block scales its whole fit and changes nothing
        let skewed = (0..cols)
            .map(|c| if c % 2 == 0 { 10.0 } else { 1.0 })
            .collect::<Vec<_>>();
        let even = vec![1.0f32; cols];
        let weighted_error = |back: &[f32]| {
            x.chunks(cols)
                .zip(back.chunks(cols))
                .map(|(a, b)| {
                    (0..cols)
                        .map(|c| skewed[c] * (a[c] - b[c]).powi(2))
                        .sum::<f32>()
                })
                .sum::<f32>()
        };
        for ty in LOW_BIT {
            let (_, plain) = rms_error(ty, cols, &x, Some(&even));
            let (_, weighted) = rms_error(ty, cols, &x, Some(&skewed));
            assert!(
                weighted_error(&weighted) < weighted_error(&plain),
                "{ty:?}: the imatrix did not lower the weighted error"
            );
        }
    }

    #[test]
    fn the_imatrix_types_refuse_to_quantize_without_one() {
        let x = values(256, 5);
        for ty in LOW_BIT {
            assert_eq!(
                quantize_rows(ty, 256, &x, None).is_err(),
                requires_imatrix(ty),
                "{ty:?}"
            );
        }
    }

    #[test]
    fn a_zero_row_stays_zero() {
        let x = vec![0f32; 256];
        let imatrix = vec![1.0f32; 256];
        for ty in [GgufType::Iq4Nl, GgufType::Iq4Xs, GgufType::Iq4Ks]
            .into_iter()
            .chain(LOW_BIT)
        {
            let (_, back) = rms_error(ty, 256, &x, Some(&imatrix));
            assert!(back.iter().all(|v| *v == 0.0), "{ty:?}");
        }
    }
}
